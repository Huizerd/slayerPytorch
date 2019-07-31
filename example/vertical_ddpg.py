# Other
import gym
import gym_mav
import math
import random
import argparse
import pandas as pd
import matplotlib.pyplot as plt
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

# Determinism
torch.manual_seed(0)
random.seed(0)

# GPU is to be used
# TODO: implement mixed precision in case we would be training on Tesla architectures
assert torch.cuda.is_available(), "CUDA-enabled GPU is needed!"
DEVICE = torch.device("cuda")

# Transition in the environment
# Essentially maps (state, action) to (next state, reward)
Transition = namedtuple("Transition", ("state", "action", "reward", "next_state"))


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


# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, actions, mu=0, theta=0.15, sigma=0.2, dt=0.01):
        # TODO: some use a dt
        self.actions = actions
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.reset()

    def reset(self):
        self.x = torch.ones(self.actions, device=DEVICE, dtype=torch.float) * self.mu

    def sample(self):
        # If dt, multiply with first term
        dx = self.theta * (self.mu - self.x) * self.dt + self.sigma * math.sqrt(self.dt) * torch.randn(
            self.x.size(), device=DEVICE, dtype=torch.float
        )
        self.x += dx

        return self.x


class Critic(nn.Module):
    def __init__(self, states, hidden, actions):
        super(Critic, self).__init__()

        self.bn1 = nn.BatchNorm1d(states)
        self.fcs1 = nn.Linear(states, hidden[0])
        self.bn2 = nn.BatchNorm1d(hidden[0])
        self.fcs2 = nn.Linear(hidden[0], hidden[1])
        self.bn3 = nn.BatchNorm1d(hidden[1])
        # TODO: not sure the paper specifies a separate layer for actions!
        # self.fca1 = nn.Linear(actions, hidden[0])
        # self.fca2 = nn.Linear(hidden[0], hidden[1])
        # No action layer, so dimension is hidden + action size
        self.fc3 = nn.Linear(hidden[1] + actions, 1)

        # Gain of 0 assumes use of ReLU
        # TODO: investigate effects of uniform vs normal Kaiming init
        # TODO: DDPG paper actually uses slightly different init --> exactly torch Linear default, which compensates for Kaiming uniform sqrt(3)
        # TODO: final layer of weights should be init with [-0.003, 0.003] uniform
        # nn.init.kaiming_uniform_(self.fcs1.weight, a=0.0, nonlinearity="relu")
        # nn.init.kaiming_uniform_(self.fca1.weight, a=0.0, nonlinearity="relu")
        # nn.init.kaiming_uniform_(self.fc2.weight, a=0.0, nonlinearity="relu")
        nn.init.uniform_(self.fc3.weight, -0.003, 0.003)
        # nn.init.zeros_(self.fcs1.bias)
        # nn.init.zeros_(self.fca1.bias)
        # nn.init.zeros_(self.fc2.bias)
        nn.init.uniform_(self.fc3.bias, -0.003, 0.003)

    def forward(self, state, action):
        s = self.bn1(state)
        s = F.relu(self.fcs1(s))
        s = self.bn2(s)
        s = F.relu(self.fcs2(s))
        s = self.bn3(s)
        # a = F.relu(self.fca1(action))
        # a = F.relu(self.fca2(a))
        x = torch.cat((s, action), dim=1)
        x = self.fc3(x)

        # Real-valued, so no activation here
        return x


class Actor(nn.Module):
    def __init__(self, states, hidden, actions, action_bounds):
        super(Actor, self).__init__()

        self.action_bounds = action_bounds
        # Paper: batch norm on all layers of actor and state input
        # Suggests before layer? --> solves following comment
        # What about last layer? Seems counterintuitive, since we bound output by tanh
        self.bn1 = nn.BatchNorm1d(states)
        self.fc1 = nn.Linear(states, hidden[0])
        self.bn2 = nn.BatchNorm1d(hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.bn3 = nn.BatchNorm1d(hidden[1])
        self.fc3 = nn.Linear(hidden[1], actions)

        # TODO: again, do Kaiming inits although different from paper
        # TODO: final layer of weights should be init with [-0.003, 0.003] uniform
        # nn.init.kaiming_uniform_(self.fc1.weight, a=0.0, nonlinearity="relu")
        # nn.init.kaiming_uniform_(self.fc2.weight, a=0.0, nonlinearity="relu")
        nn.init.uniform_(self.fc3.weight, -0.003, 0.003)
        # nn.init.zeros_(self.fc1.bias)
        # nn.init.zeros_(self.fc2.bias)
        nn.init.uniform_(self.fc3.bias, -0.003, 0.003)

    def forward(self, state):
        # Batch norm after activations == batch norm on layer inputs (except for 1st layer)
        x = self.bn1(state)
        x = F.relu(self.fc1(x))
        x = self.bn2(x)
        x = F.relu(self.fc2(x))
        x = self.bn3(x)
        x = torch.tanh(self.fc3(x))  # warning says torch.tanh() instead of F.tanh()

        return sum(self.action_bounds) / 2.0 + (self.action_bounds[1] - sum(self.action_bounds) / 2.0) * x


class Trainer:
    def __init__(
        self,
        states,
        hidden,
        actions,
        action_bounds,
        learning_rate,
        weight_decay,
        memory_capacity,
        batch_size,
        tau,
        gamma,
    ):
        self.action_bounds = action_bounds
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.memory = ReplayMemory(memory_capacity)
        self.noise = OrnsteinUhlenbeckActionNoise(actions)

        self.actor = Actor(states, hidden, actions, action_bounds).to(DEVICE)
        self.target_actor = Actor(states, hidden, actions, action_bounds).to(DEVICE)
        self.target_actor.eval()
        self.actor_optim = optim.Adam(
            self.actor.parameters(), learning_rate / 10
        )

        self.critic = Critic(states, hidden, actions).to(DEVICE)
        self.target_critic = Critic(states, hidden, actions).to(DEVICE)
        self.target_critic.eval()
        self.critic_optim = optim.Adam(
            self.critic.parameters(), learning_rate, weight_decay=weight_decay
        )

        self.hard_update()

    def get_exploitation_action(self, state):
        self.actor.eval()
        action = self.actor.forward(state).detach()
        self.actor.train()
        return action

    def get_exploration_action(self, state):
        self.actor.eval()
        action = self.actor.forward(state).detach() + (self.noise.sample() * (self.action_bounds[1] - sum(self.action_bounds) / 2.0))
        self.actor.train()
        return action.clamp_(*self.action_bounds)

    def get_scaled_action(self, action):
        return (
            sum(self.action_bounds) / 2.0
            + (self.action_bounds[1] - sum(self.action_bounds) / 2.0) * action.item()
        )

    def hard_update(self):
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

    def soft_update(self):
        for target_param, param in zip(
            self.target_actor.parameters(), self.actor.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

        for target_param, param in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def optimize(self):
        # TODO: maybe implement policy/value map creation somewhere here?
        # Only replay if enough experience
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)

        # Transpose the batch
        # From array of Transitions to Transition of arrays
        batch = Transition(*zip(*transitions))

        # Masks for non-terminal states
        non_terminal_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=DEVICE,
            dtype=torch.uint8,
        )
        non_terminal_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )
        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        rewards = torch.cat(batch.reward)

        # Optimize critic
        next_actions = self.target_actor.forward(non_terminal_next_states).detach()
        next_q_values = torch.zeros_like(states)
        next_q_values[non_terminal_mask] = self.target_critic.forward(
            non_terminal_next_states, next_actions
        ).detach()
        expected_td_target = (
            rewards[..., None] + self.gamma * next_q_values
        )  # y_exp = r + gamma * Q(s', pi(s'))
        predicted_td_target = self.critic.forward(states, actions)  # y_pred = Q(s, a)
        # TODO: although MSE used in paper, Huber (smooth L1) might be better, robust to exploding grads
        critic_loss = F.mse_loss(predicted_td_target, expected_td_target)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # Optimize actor
        predicted_actions = self.actor.forward(states)
        actor_loss = -self.critic.forward(states, predicted_actions).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self.soft_update()


def moving_average(x, window=100):
    return pd.Series(x).rolling(window=window, min_periods=1).mean().values


def make_altitude_map(altitude):
    fig, ax = plt.subplots()

    ax.plot(range(len(altitude)), altitude)
    ax.set_title("Altitude map")
    ax.set_xlabel("Step")
    ax.set_ylabel("Altitude")

    return fig


def make_action_map(actions):
    fig, ax = plt.subplots()

    ax.plot(range(len(actions)), actions)
    ax.set_title("Action map")
    ax.set_xlabel("Step")
    ax.set_ylabel("Thrust")

    return fig


if __name__ == "__main__":
    # Parse for configuration file
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="vertical_ddpg.yaml",
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
    n_actions = env.action_space.shape[0]
    n_states = 2 if env.state_obs == "altitude" else 1

    # NN
    trainer = Trainer(
        n_states,
        config["network"]["hiddenSize"],
        n_actions,
        config["environment"]["actionBounds"],
        config["training"]["learningRate"],
        config["training"]["weightDecay"],
        config["network"]["memorySize"],
        config["training"]["batchSize"],
        config["training"]["tau"],
        config["training"]["gamma"],
    )

    # Weights & Biases watching
    wandb.watch((trainer.actor, trainer.critic), log="all")

    # Tracking vars
    accumulated_rewards = []
    eval_rewards = []

    for i_episode in range(config["training"]["episodes"]):
        # Initialize the environment and state
        # BCHW is torch order, but we only do BC
        state = env.reset().float().to(DEVICE).view(1, -1)

        accumulated_reward = 0.0
        duration = 0
        max_div = (-2 * env.state[1] / env.state[0]).item()
        altitude_map = []
        action_map = []

        for t in count():
            # Render environment
            if (
                config["environment"]["render"]
                and i_episode % config["environment"]["interval"] == 0
            ):
                env.render()

            # Explore or exploit (evaluate)
            if i_episode % config["environment"]["interval"] == 0:
                action = trainer.get_exploitation_action(state)
            else:
                action = trainer.get_exploration_action(state)
            # action_env = trainer.get_scaled_action(action)

            # Take action
            next_state, reward, done, _ = env.step(action.item())
            accumulated_reward += reward
            reward = torch.tensor([reward], device=DEVICE, dtype=torch.float)

            # Log value, policy and altitude map
            divergence = (-2 * env.state[1] / env.state[0]).item()
            if abs(divergence) > abs(max_div):
                max_div = divergence
            altitude_map.append(env.state[0].item())
            action_map.append(action.item())

            # Set to None if next state is terminal
            if not done:
                next_state = next_state.float().to(DEVICE).view(1, -1)

                if i_episode % config["environment"]["interval"] != 0:
                    # Store the transition in memory
                    trainer.memory.push(state, action, reward, next_state)
            else:
                next_state = None

            # Move to the next state
            state = next_state

            if i_episode % config["environment"]["interval"] != 0:
                # Optimize networks
                trainer.optimize()

            # Episode finished
            if done:
                duration = t + 1
                break

        # After-episode things
        accumulated_rewards.append(accumulated_reward)
        wandb.log(
            {
                "Reward": accumulated_reward,
                "RewardSmooth": moving_average(accumulated_rewards)[-1],
                "MaxDiv": max_div,
                "Duration": duration,
                "AltitudeMap": make_altitude_map(altitude_map),
                "ActionMap": make_action_map(action_map),
            }
        )
        trainer.noise.reset()  # described in paper

    print("Complete")
    env.close()
