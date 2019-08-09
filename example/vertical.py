# SLAYER
import os
import sys

CURRENT_TEST_DIR = os.getcwd()
sys.path.append(CURRENT_TEST_DIR + "/../src")
import slayerSNN as snn

# Other
import gym
import gym_mav
import math
import random
import argparse
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
    def __init__(self, config, inputs, outputs):
        super(Network, self).__init__()

        # One or none hidden layers
        self.hidden = config["network"]["hiddenSize"]

        # Initialize SLAYER
        slayer = snn.layer(config["neuron"], config["simulation"])
        self.slayer = slayer

        # Define network layers
        if self.hidden > 0:
            self.fc1 = slayer.dense(inputs, self.hidden)
            self.fc2 = slayer.dense(self.hidden, outputs)
        else:
            self.fc1 = slayer.dense(inputs, outputs)

    def forward(self, spike_input):
        if self.hidden > 0:
            spike_layer1 = self.slayer.spike(self.slayer.psp(self.fc1(spike_input)))
            spike_layer2 = self.slayer.spike(self.slayer.psp(self.fc2(spike_layer1)))
        else:
            spike_layer2 = self.slayer.spike(self.slayer.psp(self.fc1(spike_input)))

        return spike_layer2


# With this, states closer to zero vary more, which means they excite more different cells
# Potentially making them better to differentiate
# Another way: vary place cell centers with sigmoid (closer together in center) and make them thinner
def sigmoid(
    x: torch.Tensor,
    y_min: torch.Tensor,
    y_step: torch.Tensor,
    x_mid: torch.Tensor,
    steepness: torch.Tensor,
) -> torch.Tensor:
    y = torch.where(
        x >= 0,
        y_step / (1 + torch.exp(-steepness * (x - x_mid))) + y_min,
        (y_step * torch.exp(steepness * x)) / (1 + torch.exp(steepness * (x - x_mid)))
        + y_min,
    )
    return y


# Use place cells for encoding state
def place_cell_centers(state_bounds, n):
    centers = [
        torch.linspace(*b, c, device=DEVICE, dtype=torch.float)
        for b, c in zip(state_bounds, n)
    ]
    width = torch.tensor(
        [c[1] - c[0] for c in centers], device=DEVICE, dtype=torch.float
    )
    # View: (batch, place cells, states)
    return (
        torch.functional.cartesian_prod(*centers).view(1, -1, len(state_bounds)),
        width,
    )


def place_cells(state, centers, width, max_rate):
    distance = (centers - state) ** 2
    firing_rate = max_rate * torch.exp(-(distance / (2.0 * width ** 2)).sum(-1))
    return firing_rate


# Encoding state as spike trains
def encode(
    state, centers, width, state_bounds, max_rate, steepness, time, sample_time, process
):
    # Note that quite long trains are needed to get something remotely deterministic
    steps = int(time / sample_time)

    # Clamp only in case we don't do a transform
    if process == "transform":
        low = centers[0, 0]
        high = centers[0, -1]
        mid = (high + low) / 2.0
        state_prep = sigmoid(
            state, y_min=low, y_step=(high - low), x_mid=mid, steepness=steepness / high
        )
    elif process == "clamp":
        # Singleton dimension needed for clamp
        low = centers[:, 0]
        high = centers[:, 0]
        state_prep = torch.max(torch.min(state, high), low)
    elif process == "nothing":
        state_prep = state.clone().detach()
    else:
        raise NotImplementedError("Provide a valid choice: transform, clamp, nothing.")

    # Repeat: (batch, place cells, time)
    firing_rate = place_cells(state_prep, centers, width, max_rate)[..., None].repeat(
        1, 1, steps
    )
    spikes = torch.rand(*firing_rate.size(), device=DEVICE) < firing_rate * (
        sample_time / 1000.0
    )
    # Shape: (batch, channels/place cells, height, width, time)
    return spikes[:, :, None, None, :].float(), firing_rate[:, :, None, None, :]


# Decode spikes into Q-values
def decode(q_values_enc):
    return q_values_enc.sum(-1).view(q_values_enc.size(0), -1)


# Decode output spikes into values/actions
def select_action(q_values_enc, steps_done, eps_start, eps_end, eps_decay):
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * math.exp(
        -1.0 * steps_done / eps_decay
    )

    if sample > eps_threshold:
        return decode(q_values_enc).max(1)[1].view(1, 1), eps_threshold
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

    # No need for encoding here (I think)
    # Makes more sense to store states as spike trains
    # Are actions also spikes?
    # Or does that matter less since decoding is deterministic?
    # No, actions are actions --> Q-values are spikes

    # Compute Q-values: select action based on maximum
    # Action which would have been taken based on policy net
    # But Q-learning is off-policy, so we don't actually take it
    q_values_enc = policy_net(state_batch)
    q_values = decode(q_values_enc).gather(1, action_batch)

    # Compute expected values for next states
    # Based on older target net
    # Zero in case of terminal state
    next_values = torch.zeros(batch_size, device=DEVICE, dtype=torch.float)
    with torch.no_grad():
        next_values[non_terminal_mask] = decode(
            target_net(non_terminal_next_states)
        ).max(1)[0]

    # Compute expected Q-values
    expected_q_values = (next_values * gamma) + reward_batch

    # Compute loss
    # Can be computed before or after decoding of output spikes
    # We do after now, so we can use Huber loss
    # Otherwise, use the built-in snn.loss() based on # of spikes
    # Huber loss here, which is squared for error within [-1, 1], and absolute outside
    # Might be redundant, since clipping gradients + MSE could achieve the same..
    # Yes, redundant! See https://openai.com/blog/openai-baselines-dqn/
    loss = F.smooth_l1_loss(q_values, expected_q_values[..., None])

    # Optimize model
    optimizer.zero_grad()
    loss.backward()  # TODO: try with other loss from SLAYER
    # Clamp gradients to improve stability: deprecated, implemented via Huber loss
    # See https://stackoverflow.com/questions/36462962/loss-clipping-in-tensor-flow-on-deepminds-dqn
    # See DQN paper (Mnih et al., 2015)
    optimizer.step()


def make_value_map(policy_net, actions, obs, encoded_obs=None, decode=None):
    fig, ax = plt.subplots()

    if encoded_obs is not None or decode is not None:
        assert (
            encoded_obs is not None and decode is not None
        ), "Both encoded observations and a decoder are needed."
        with torch.no_grad():
            q_values = decode(policy_net(encoded_obs))
    else:
        with torch.no_grad():
            q_values = policy_net(obs)

    for i, act in enumerate(actions):
        ax.plot(obs.squeeze().tolist(), q_values[:, i].tolist(), label=f"{act} N")
    ax.set_title("Value map")
    ax.set_xlabel("Divergence")
    ax.set_ylabel("Q-value")
    ax.legend()
    ax.grid()

    return fig


def make_policy_map(policy_net, actions, obs, encoded_obs=None, decode=None):
    fig, ax = plt.subplots()

    if encoded_obs is not None or decode is not None:
        assert (
            encoded_obs is not None and decode is not None
        ), "Both encoded observations and a decoder are needed."
        with torch.no_grad():
            policy = decode(policy_net(encoded_obs)).argmax(-1)
    else:
        with torch.no_grad():
            policy = policy_net(obs).argmax(-1)

    ax.plot(obs.squeeze().tolist(), [actions[i] for i in policy.tolist()])
    ax.set_title("Policy map")
    ax.set_xlabel("Divergence")
    ax.set_ylabel("Action")
    ax.grid()

    return fig


def make_altitude_map(altitude):
    fig, ax = plt.subplots()

    ax.plot(range(len(altitude)), altitude)
    ax.set_title("Altitude map")
    ax.set_xlabel("Step")
    ax.set_ylabel("Altitude")
    ax.grid()

    return fig


def make_divergence_map(divergence):
    fig, ax = plt.subplots()

    ax.plot(range(len(divergence)), divergence)
    ax.set_title("Divergence map")
    ax.set_xlabel("Step")
    ax.set_ylabel("Divergence")
    ax.grid()

    return fig


def make_vertspeed_map(vertspeed):
    fig, ax = plt.subplots()

    ax.plot(range(len(vertspeed)), vertspeed)
    ax.set_title("Vertical speed map")
    ax.set_xlabel("Step")
    ax.set_ylabel("Vertical speed")
    ax.grid()

    return fig


if __name__ == "__main__":
    # Parse for configuration file
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="vertical_div.yaml",
        help="Select configuration file",
    )
    args = vars(parser.parse_args())

    # Config
    config = Config(config_paths=[args["config"]])
    wandb.init(config=config, project="baselines", tags=["DQSNN"])

    # Place cells
    centers, width = place_cell_centers(
        config["placeCells"]["stateBounds"], config["placeCells"]["N"]
    )
    n_place_cells = centers.size(1)

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

    # SNN
    policy_net = Network(config, n_place_cells, n_actions).to(DEVICE)
    target_net = Network(config, n_place_cells, n_actions).to(DEVICE)
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
        # Reshape for encoding: (batch, place cells, states)
        # Output: (batch, place cells/channels, height, width, time)
        obs_space = torch.arange(
            -10.0, 10.0, 0.05, device=DEVICE, dtype=torch.float
        ).view(-1, 1, 1)
        obs_space_enc, _ = encode(
            obs_space,
            centers,
            width,
            config["placeCells"]["stateBounds"],
            config["placeCells"]["maxRate"],
            config["placeCells"]["steepness"],
            config["simulation"]["tSample"],
            config["simulation"]["Ts"],
            config["placeCells"]["process"],
        )

    for i_episode in range(config["training"]["episodes"]):
        # Initialize the environment and state
        state = env.reset().float().to(DEVICE).view(1, -1)

        # Encode state
        state_enc, _ = encode(
            state,
            centers,
            width,
            config["placeCells"]["stateBounds"],
            config["placeCells"]["maxRate"],
            config["placeCells"]["steepness"],
            config["simulation"]["tSample"],
            config["simulation"]["Ts"],
            config["placeCells"]["process"],
        )

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
                q_values_enc = policy_net(state_enc)

            # Select and perform an action
            action, eps = select_action(
                q_values_enc,
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
                next_state_enc, _ = encode(
                    next_state,
                    centers,
                    width,
                    config["placeCells"]["stateBounds"],
                    config["placeCells"]["maxRate"],
                    config["placeCells"]["steepness"],
                    config["simulation"]["tSample"],
                    config["simulation"]["Ts"],
                    config["placeCells"]["process"],
                )
            else:
                next_state_enc = None

            # Store the transition in memory
            memory.push(state_enc, action, next_state_enc, reward)

            # Move to the next state
            state = next_state
            state_enc = next_state_enc

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

                if (
                    env.state_obs == "divergence"
                    and i_episode % config["environment"]["interval"] == 0
                ):
                    wandb.log(
                        {
                            "ValueMap": make_value_map(
                                policy_net,
                                actions,
                                obs_space,
                                encoded_obs=obs_space_enc,
                                decode=decode,
                            ),
                            "PolicyMap": make_policy_map(
                                policy_net,
                                actions,
                                obs_space,
                                encoded_obs=obs_space_enc,
                                decode=decode,
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
