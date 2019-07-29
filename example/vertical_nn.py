# Other
import gym
import gym_mav
import math
import random
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque, namedtuple
from operator import itemgetter
from itertools import count

# Torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Weights & Biases
import wandb
from wandb.wandb_config import Config

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


# Spiking network to be used
class Network(nn.Module):
    def __init__(self, inputs, hidden, outputs):
        super(Network, self).__init__()

        # Define network layers
        self.fc1 = nn.Linear(inputs, hidden)
        self.fc2 = nn.Linear(hidden, outputs)


    def forward(self, x):
        # Batch norm after ReLU
        # See https://github.com/keras-team/keras/issues/1802#issuecomment-187966878
        # No batch norm after all, more stable like this
        x = F.relu(self.fc1(x))

        # Linear activation (as in DQN) since we're approximating real-valued Q-values
        return self.fc2(x)


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
    next_values = torch.zeros(batch_size, device=DEVICE)
    next_values[non_terminal_mask] = target_net(non_terminal_next_states).max(1)[0].detach()

    # Compute expected Q-values
    expected_q_values = (next_values * gamma) + reward_batch

    # Compute loss
    # Can be computed before or after decoding of output spikes
    # We do after now, so we can use Huber loss
    # Otherwise, use the built-in snn.loss() based on # of spikes
    loss = F.smooth_l1_loss(q_values, expected_q_values[..., None])

    # Optimize model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        # Clamp to improve stability
        # See https://stackoverflow.com/questions/36462962/loss-clipping-in-tensor-flow-on-deepminds-dqn
        # See DQN paper (Mnih et al., 2015)
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def moving_average(x, window=100):
    return pd.Series(x).rolling(window=window, min_periods=1).mean().values


def make_value_map(state_values):
    fig, ax = plt.subplots()

    for i, sv in enumerate(state_values):
        sv.sort(key=itemgetter(0))
        ax.plot([s[0] for s in sv], [s[1] for s in sv], label=f"Action {i}")
    ax.set_title("Value map")
    ax.set_xlabel("Divergence")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid()

    return fig


def make_policy_map(state_actions):
    state_actions.sort(key=itemgetter(0))
    fig, ax = plt.subplots()

    ax.plot([s[0] for s in state_actions], [s[1] for s in state_actions])
    ax.set_title("Policy map")
    ax.set_xlabel("Divergence")
    ax.set_ylabel("Action")

    return fig


def make_altitude_map(altitude):
    fig, ax = plt.subplots()

    ax.plot(range(len(altitude)), altitude)
    ax.set_title("Altitude map")
    ax.set_xlabel("Step")
    ax.set_ylabel("Altitude")

    return fig


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
    wandb.init(config=config, project="vertical")

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
        gravity=config["environment"]["gravity"],
        timed_reward=config["environment"]["timedReward"],
    )
    env.seed(0)
    actions = config["environment"]["actions"]
    n_actions = len(actions)
    n_states = 2 if env.state_obs == "altitude" else 1

    # NN
    policy_net = Network(n_states, config["network"]["hiddenSize"], n_actions).to(DEVICE)
    target_net = Network(n_states, config["network"]["hiddenSize"], n_actions).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # Weights & Biases watching
    wandb.watch(policy_net, log="all")

    # Optimizer and replay memory
    optimizer = optim.Adam(
        policy_net.parameters(), lr=config["training"]["learningRate"], amsgrad=True
    )
    memory = ReplayMemory(config["network"]["memorySize"])

    # Tracking vars
    steps_done = 0
    accumulated_rewards = []

    for i_episode in range(config["training"]["episodes"]):
        # Initialize the environment and state
        # BCHW is torch order, but we only do BC
        state = env.reset().float().to(DEVICE).view(1, -1)

        accumulated_reward = 0.0
        max_div = (-2 * env.state[1] / env.state[0]).item()
        value_map = [[] for _ in range(n_actions)]
        policy_map = []
        altitude_map = []

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

            # Log value, policy and altitude map
            divergence = (-2 * env.state[1] / env.state[0]).item()
            if abs(divergence) > abs(max_div):
                max_div = divergence
            for i in range(n_actions):
                value_map[i].append((divergence, q_values[0, i].item()))
            policy_map.append((divergence, action.item()))
            altitude_map.append(env.state[0].item())

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
                accumulated_rewards.append(accumulated_reward)
                wandb.log(
                    {
                        "Reward": accumulated_reward,
                        "RewardSmooth": moving_average(accumulated_rewards)[-1],
                        "MaxDiv": max_div,
                        "Duration": t + 1,
                        "Epsilon": eps,
                        "ValueMap": make_value_map(value_map),
                        "PolicyMap": make_policy_map(policy_map),
                        "AltitudeMap": make_altitude_map(altitude_map),
                    }
                )
                break

        # Update the target network, copying all weights etc.
        if i_episode % config["training"]["targetUpdate"] == 0:
            target_net.load_state_dict(policy_net.state_dict())

    # TODO: maybe create proper value map and policy map at end, and save model etc?
    # Idea: we know the "right" policy for positive and negative divergence, so select values based on these?
    print("Complete")
    env.close()
