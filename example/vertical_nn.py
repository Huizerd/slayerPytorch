# Other
import gym
import gym_mav
import math
import random
import argparse
from collections import deque, namedtuple
from itertools import count

# Torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Weights & Biases
import wandb
from wandb.wandb_config import Config

# Local
from vertical import (
    make_altitude_map,
    make_policy_map,
    make_value_map,
    make_divergence_map,
    make_vertspeed_map,
)

# Determinism
torch.manual_seed(0)
random.seed(0)

# GPU is to be used
# TODO: implement mixed precision in case we would be training on Tesla architectures
assert torch.cuda.is_available(), "CUDA-enabled GPU is needed!"
DEVICE = torch.device("cuda")

# Transition in the environment
# Essentially maps (state, action) to (next state, reward)
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


# Replay batches of transitions in memory for optimization
class ReplayMemory(object):
    def __init__(self, capacity: int) -> None:
        self.memory = deque(maxlen=capacity)

    def push(self, *args) -> None:
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> list:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


# Network to be used
class Network(nn.Module):
    def __init__(self, inputs, hidden, outputs):
        super(Network, self).__init__()

        # One or none hidden layers
        self.hidden = hidden

        # Define network layers
        if self.hidden > 0:
            self.fc1 = nn.Linear(inputs, self.hidden)
            self.fc2 = nn.Linear(self.hidden, outputs)
        else:
            self.fc1 = nn.Linear(inputs, outputs)

    def forward(self, x):
        # Batch norm after ReLU
        # See https://github.com/keras-team/keras/issues/1802#issuecomment-187966878
        # No batch norm after all, more stable like this
        if self.hidden > 0:
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
        else:
            x = self.fc1(x)

        # Linear activation (as in DQN) since we're approximating real-valued Q-values
        return x


def select_action(q_values, steps_done, eps_start, eps_end, eps_decay):
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * math.exp(
        -1.0 * steps_done / eps_decay
    )

    if sample > eps_threshold:
        return q_values.max(1)[1].view(1, 1), eps_threshold
    else:
        return (
            torch.tensor(
                [[random.randrange(n_actions)]], device=DEVICE, dtype=torch.long
            ),
            eps_threshold,
        )


# Optimize model by training from memory
def optimize_model(batch_size, gamma):
    # Only replay if enough experience
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)

    # Transpose the batch
    # From array of Transitions to Transition of arrays
    batch = Transition(*zip(*transitions))

    # Masks for non-terminal states
    non_terminal_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=DEVICE,
        dtype=torch.uint8,
    )
    non_terminal_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q-values: select action based on maximum
    # Action which would have been taken based on policy net
    # But Q-learning is off-policy, so we don't actually take it
    q_values = policy_net(state_batch).gather(1, action_batch)

    # Compute expected values for next states
    # Based on older target net
    # Zero in case of terminal state
    next_values = torch.zeros(batch_size, device=DEVICE, dtype=torch.float)
    with torch.no_grad():
        next_values[non_terminal_mask] = target_net(non_terminal_next_states).max(1)[0]

    # Compute expected Q-values
    expected_q_values = (next_values * gamma) + reward_batch

    # Compute loss
    # Huber loss here, which is squared for error within [-1, 1], and absolute outside
    # Might be redundant, since clipping gradients + MSE could achieve the same..
    # Yes, redundant! See https://openai.com/blog/openai-baselines-dqn/
    loss = F.smooth_l1_loss(q_values, expected_q_values[..., None])

    # Optimize model
    optimizer.zero_grad()
    loss.backward()
    # Clamp gradients to improve stability: deprecated, implemented via Huber loss
    # See https://stackoverflow.com/questions/36462962/loss-clipping-in-tensor-flow-on-deepminds-dqn
    # See DQN paper (Mnih et al., 2015)
    optimizer.step()


if __name__ == "__main__":
    # Parse for configuration file
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="vertical_nn.yaml",
        help="Select configuration file",
    )
    args = vars(parser.parse_args())

    # Config
    config = Config(config_paths=[args["config"]])
    wandb.init(config=config, project="baselines", tags=["DQN"])

    # Environment
    env = gym.make(
        config["environment"]["name"],
        obs_noise=config["environment"]["obsNoise"],
        init_rand=config["environment"]["initRand"],
        init_state=config["environment"]["initState"],
        delay=config["environment"]["delay"],
        reward_mods=config["environment"]["rewardMods"],
        state_bounds=[config["environment"]["altBounds"], None],
        total_steps=config["environment"]["steps"],
        goal_obs=config["environment"]["goalObs"],
        state_obs=config["environment"]["stateObs"],
        action_bounds=config["environment"]["actionBounds"],
        action_offset=config["environment"]["actionOffset"],
        gravity=config["environment"]["gravity"],
        timed_reward=config["environment"]["timedReward"],
    )
    env.seed(0)
    actions = config["environment"]["actions"]
    n_actions = len(actions)
    n_states = 2 if env.state_obs == "altitude" else 1

    # NN
    policy_net = Network(n_states, config["network"]["hiddenSize"], n_actions).to(
        DEVICE
    )
    target_net = Network(n_states, config["network"]["hiddenSize"], n_actions).to(
        DEVICE
    )
    target_net.load_state_dict(policy_net.state_dict())

    # Weights & Biases watching
    wandb.watch(policy_net, log="all")

    # Optimizer and replay memory
    optimizer = optim.Adam(
        policy_net.parameters(), lr=config["training"]["learningRate"], amsgrad=True
    )
    memory = ReplayMemory(config["network"]["memorySize"])

    # Tracking vars
    steps_done = 0
    accumulated_rewards_smooth = deque(maxlen=100)

    # Input observations for value/policy maps, only for divergence control
    if env.state_obs == "divergence":
        obs_space = torch.arange(-10.0, 10.0, 0.05, device=DEVICE, dtype=torch.float)[
            :, None
        ]

    for i_episode in range(config["training"]["episodes"]):
        # Initialize the environment and state
        # BCHW is torch order, but we only do BC
        state = env.reset().float().to(DEVICE).view(1, -1)

        accumulated_reward = 0.0
        max_div = (-2 * env.state[1] / env.state[0]).item()
        altitude_map = []
        divergence_map = []
        vertspeed_map = []

        for t in count():
            # Render environment
            if (
                config["environment"]["render"]
                and i_episode % config["environment"]["interval"] == 0
            ):
                env.render()

            # Feed encoded state through network
            # no_grad() here or in select_action: doesn't matter
            with torch.no_grad():
                q_values = policy_net(state)

            # Select and perform an action
            action, eps = select_action(
                q_values,
                steps_done,
                config["training"]["epsStart"],
                config["training"]["epsEnd"],
                config["training"]["epsDecay"],
            )
            next_state, reward, done, _ = env.step(actions[action.item()])
            accumulated_reward += reward
            reward = torch.tensor([reward], device=DEVICE, dtype=torch.float)

            # Log maps
            # All state observations without noise (directly from env)
            divergence = (-2 * env.state[1] / env.state[0]).item()
            if abs(divergence) > abs(max_div):
                max_div = divergence
            altitude_map.append(env.state[0].item())
            divergence_map.append(divergence)
            vertspeed_map.append(env.state[1].item())

            # Set to None if next state is terminal
            if not done:
                next_state = next_state.float().to(DEVICE).view(1, -1)
            else:
                next_state = None

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            optimize_model(config["training"]["batchSize"], config["training"]["gamma"])

            # Increment counters
            steps_done += 1

            # Episode finished
            if done:
                accumulated_rewards_smooth.append(accumulated_reward)
                wandb.log(
                    {
                        "Reward": accumulated_reward,
                        "RewardSmooth": sum(accumulated_rewards_smooth)
                        / len(accumulated_rewards_smooth),
                        "MaxDiv": max_div,
                        "Duration": t + 1,
                        "Epsilon": eps,
                        "AltitudeMap": make_altitude_map(altitude_map),
                        "DivergenceMap": make_divergence_map(divergence_map),
                        "VertSpeedMap": make_vertspeed_map(vertspeed_map),
                    },
                    step=i_episode,
                )

                if env.state_obs == "divergence":
                    wandb.log(
                        {
                            "ValueMap": make_value_map(policy_net, actions, obs_space),
                            "PolicyMap": make_policy_map(
                                policy_net, actions, obs_space
                            ),
                        },
                        step=i_episode,
                    )

                break

        # Update the target network, copying all weights etc.
        if i_episode % config["training"]["targetUpdate"] == 0:
            target_net.load_state_dict(policy_net.state_dict())

    print("Complete")
    env.close()
