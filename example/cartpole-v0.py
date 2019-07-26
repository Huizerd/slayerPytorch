# SLAYER
import sys, os
from datetime import datetime

CURRENT_TEST_DIR = os.getcwd()
sys.path.append(CURRENT_TEST_DIR + "/../src")

import slayerSNN as snn

# Other
import gym
import math
import random
import pdb
import numpy as np
import pandas as pd
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

# GPU is to be used
assert torch.cuda.is_available(), "CUDA-enabled GPU is needed!"
device = torch.device("cuda")


# Transition in the environment
# Essentially maps (state, action) to (next state, reward)
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


# Replay batches of transitions in memory for optimization
class ReplayMemory(object):
    def __init__(self, capacity):
        # self.capacity = capacity
        # self.memory = []
        # self.position = 0
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        # if len(self.memory) < self.capacity:
        #     self.memory.append(None)
        # self.memory[self.position] = Transition(*args)
        # self.position = (self.position + 1) % self.capacity
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# Spiking network to be used
# 2 fully-connected layers of 512 neurons
class Network(nn.Module):
    def __init__(self, config, inputs, outputs):
        super(Network, self).__init__()

        # Initialize SLAYER
        slayer = snn.layer(config["neuron"], config["simulation"])
        self.slayer = slayer

        # Define network layers
        self.fc1 = slayer.dense(inputs, config["network"]["hiddenSize"])
        self.fc2 = slayer.dense(config["network"]["hiddenSize"], outputs)

    def forward(self, spike_input):
        spike_layer1 = self.slayer.spike(self.slayer.psp(self.fc1(spike_input)))
        spike_layer2 = self.slayer.spike(self.slayer.psp(self.fc2(spike_layer1)))

        return spike_layer2


# Use place cells for encoding state
def place_cell_centers(state_bounds, n):
    centers = [
        torch.linspace(*b, c, device=device, dtype=torch.float)
        for b, c in zip(state_bounds, n)
    ]
    width = torch.tensor(
        [c[1] - c[0] for c in centers], device=device, dtype=torch.float
    )
    return torch.functional.cartesian_prod(*centers), width


def place_cells(state, centers, width):
    distance = (centers - state) ** 2
    firing_rate = config["placeCells"]["maxRate"] * torch.exp(
        -(distance / (2.0 * width ** 2)).sum(1)
    )

    return firing_rate


# Encoding state as spike trains
def encode(state, centers, width, time):
    # TODO: add sampling time (net_params["simulation"]["Ts"]) here
    # TODO: maybe not the best method to create spike trains
    # What should've been a 95 Hz train was only an 83 Hz one
    firing_rate = place_cells(state, centers, width).repeat(time, 1).permute(1, 0)
    spikes = torch.rand(*firing_rate.size(), device=device) < firing_rate / 1000.0

    return spikes.view(1, -1, 1, 1, time).float()


# Decode spikes into Q-values
def decode(q_values_enc):
    return q_values_enc.sum(-1).view(q_values_enc.size(0), -1)


# Decode output spikes into values/actions
def select_action(q_values_enc):
    global steps_done
    sample = random.random()
    eps_threshold = config["training"]["epsEnd"] + (
        config["training"]["epsStart"] - config["training"]["epsEnd"]
    ) * math.exp(-1.0 * steps_done / config["training"]["epsDecay"])
    steps_done += 1

    if sample > eps_threshold:
        with torch.no_grad():
            return decode(q_values_enc).max(1)[1].view(1, 1)
    else:
        return torch.tensor(
            [[random.randrange(n_actions)]], device=device, dtype=torch.long
        )


# Config
config = Config(config_paths=["dqsnn.yaml"])
wandb.init(config=config, project="dqsnn")

# RL stuff
# BATCH_SIZE = 128
# GAMMA = 0.999
# EPS_START = 0.9
# EPS_END = 0.05
# EPS_DECAY = 200
# TARGET_UPDATE = 10

# Place cells
# PC_MAXRATE = 100.0
# PC_N = [5, 5, 5, 5]
# PC_STATE_BOUNDS = [
#     [-4.8, 4.8],
#     [-10.0, 10.0],
#     [-24.0 * math.pi / 360, 24.0 * math.pi / 360],
#     [-5.0 * math.pi, 5.0 * math.pi],
# ]

# TODO: convert angles to radians
config["placeCells"]["stateBounds"][2] = [
    angle * math.pi / 180 for angle in config["placeCells"]["stateBounds"][2]
]
config["placeCells"]["stateBounds"][3] = [
    angle * math.pi / 180 for angle in config["placeCells"]["stateBounds"][3]
]

# TODO: check influence of twice as large angle range
centers, width = place_cell_centers(
    config["placeCells"]["stateBounds"], config["placeCells"]["N"]
)
n_place_cells = centers.size(0)

# Create environment
env = gym.make(config["environment"]["name"])
n_actions = env.action_space.n

# SNN
# net_params = snn.params("dqsnn.yaml")
policy_net = Network(config, n_place_cells, n_actions).to(device)
target_net = Network(config, n_place_cells, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# Weights & Biases watching
wandb.watch(policy_net, log="all")

# Optimizer and replay memory
optimizer = optim.Adam(
    policy_net.parameters(), lr=config["training"]["learningRate"], amsgrad=True
)
memory = ReplayMemory(config["network"]["memorySize"])

steps_done = 0
durations = []


def optimize_model():
    # Only replay if enough experience
    if len(memory) < config["training"]["batchSize"]:
        return
    transitions = memory.sample(config["training"]["batchSize"])

    # Transpose the batch
    # From array of Transitions to Transition of arrays
    batch = Transition(*zip(*transitions))

    # Masks for non-terminal states
    non_terminal_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
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
    next_values = torch.zeros(config["training"]["batchSize"], device=device)
    next_values[non_terminal_mask] = (
        decode(target_net(non_terminal_next_states)).max(1)[0].detach()
    )

    # Compute expected Q-values
    expected_q_values = (next_values * config["training"]["gamma"]) + reward_batch

    # Compute loss
    # Can be computed before or after decoding of output spikes
    # We do after now, so we can use Huber loss
    # Otherwise, use the built-in snn.loss() based on # of spikes
    loss = F.smooth_l1_loss(q_values, expected_q_values[..., None])

    # Optimize model
    optimizer.zero_grad()
    loss.backward()  # TODO: or do we need the special loss for this to go well?
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)  # TODO: why the clamping?
    optimizer.step()


def moving_average(x, window=100):
    return pd.Series(x).rolling(window=window, min_periods=1).mean().values


#### MAIN LOOP ###

for i_episode in range(config["training"]["episodes"]):
    # Initialize the environment and state
    state = torch.from_numpy(env.reset()).float().to(device)

    # Encode state
    state_enc = encode(state, centers, width, config["simulation"]["tSample"])

    for t in count():
        # Render environment
        if config["environment"]["render"]:
            env.render()

        # Feed encoded state through network
        q_values_enc = policy_net(state_enc)

        # Select and perform an action
        action = select_action(q_values_enc)
        next_state, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        # Set to None if next state is terminal
        if not done:
            next_state = torch.from_numpy(next_state).float().to(device)
            next_state_enc = encode(
                next_state, centers, width, config["simulation"]["tSample"]
            )
        else:
            next_state_enc = None

        # Store the transition in memory
        memory.push(state_enc, action, next_state_enc, reward)

        # Move to the next state
        state_enc = next_state_enc

        # Perform one step of the optimization (on the target network)
        optimize_model()

        # Episode finished
        if done:
            durations.append(t + 1)
            wandb.log(
                {"Duration": t + 1, "Duration smooth": moving_average(durations)[-1]}
            )
            break

    # Update the target network, copying all weights etc.
    if i_episode % config["training"]["targetUpdate"] == 0:
        target_net.load_state_dict(policy_net.state_dict())

print("Complete")
env.close()
