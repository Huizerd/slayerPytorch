# Other
import gym
import gym_mav
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
class ReplayMemory:
    def __init__(self, capacity: int) -> None:
        self.memory = deque(maxlen=capacity)

    def push(self, *args) -> None:
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> list:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


class NormalActionNoise:
    def __init__(self, actions, sigma=0.1):
        self.actions = actions
        self.sigma = sigma

    def sample(self):
        return self.sigma * torch.randn(self.actions, device=DEVICE, dtype=torch.float)


class Actor(nn.Module):
    def __init__(self, states, hidden, actions, action_bounds):
        super(Actor, self).__init__()

        self.action_bounds = action_bounds

        # DDPG paper: batch norm on all layers of actor and state input
        # Suggests before layer? --> solves following comment
        # What about last layer? Seems counterintuitive, since we bound output by tanh
        # According to TD3 paper: no normalization
        # And no fancy weight init anymore, not even for final layer!
        self.fc1 = nn.Linear(states, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc3 = nn.Linear(hidden[1], actions)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))

        return (
            sum(self.action_bounds) / 2.0
            + (self.action_bounds[1] - sum(self.action_bounds) / 2.0) * x
        )


class Critic(nn.Module):
    def __init__(self, states, hidden, actions):
        super(Critic, self).__init__()

        # TD3 paper: no normalization, action in first layer
        self.fc1 = nn.Linear(states + actions, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc3 = nn.Linear(hidden[1], 1)

    def forward(self, state, action):
        x = F.relu(self.fc1(torch.cat([state, action], dim=1)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Real-valued, so no activation here
        return x


class DDPG:
    def __init__(
        self,
        states,
        hidden,
        actions,
        action_bounds,
        learning_rate,
        memory_capacity,
        batch_size,
        tau,
        gamma,
        sigma,
    ):
        self.action_bounds = action_bounds
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.memory = ReplayMemory(memory_capacity)
        self.noise = NormalActionNoise(actions, sigma=sigma)

        self.actor = Actor(states, hidden, actions, action_bounds).to(DEVICE)
        self.target_actor = Actor(states, hidden, actions, action_bounds).to(DEVICE)
        self.target_actor.eval()
        self.actor_optim = optim.Adam(self.actor.parameters(), learning_rate)

        self.critic = Critic(states, hidden, actions).to(DEVICE)
        self.target_critic = Critic(states, hidden, actions).to(DEVICE)
        self.target_critic.eval()
        self.critic_optim = optim.Adam(self.critic.parameters(), learning_rate)

        self.hard_update()

    def get_exploitation_action(self, state):
        self.actor.eval()
        action = self.actor.forward(state).detach()
        self.actor.train()
        return action

    def get_exploration_action(self, state):
        self.actor.eval()
        action = self.actor.forward(state).detach() + self.noise.sample()
        self.actor.train()
        return action.clamp_(*self.action_bounds)

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

    def eval(self, env, episode):
        state = env.reset().float().to(DEVICE).view(1, -1)

        accumulated_reward = 0.0
        max_div = (-2 * env.state[1] / env.state[0]).item()
        policy_map = []
        altitude_map = []
        divergence_map = []
        vertspeed_map = []

        for t in count():
            # Select and perform an action
            action = self.get_exploitation_action(state)
            next_state, reward, done, _ = env.step(action.item())
            accumulated_reward += reward

            # Log maps
            # All state observations without noise (directly from env)
            divergence = (-2 * env.state[1] / env.state[0]).item()
            if abs(divergence) > abs(max_div):
                max_div = divergence
            policy_map.append((divergence, action.item()))
            altitude_map.append(env.state[0].item())
            divergence_map.append(divergence)
            vertspeed_map.append(env.state[1].item())

            # Set to None if next state is terminal
            if not done:
                next_state = next_state.float().to(DEVICE).view(1, -1)
            else:
                next_state = None

            # Move to the next state
            state = next_state

            # Episode finished
            if done:
                wandb.log(
                    {
                        "Reward": accumulated_reward,
                        "MaxDiv": max_div,
                        "Duration": t + 1,
                        "PolicyMap": make_policy_map(policy_map),
                        "AltitudeMap": make_altitude_map(altitude_map),
                        "DivergenceMap": make_divergence_map(divergence_map),
                        "VertSpeedMap": make_vertspeed_map(vertspeed_map),
                    },
                    step=episode,
                )
                break

    def train(self, iterations):
        # Only replay if enough experience
        if len(self.memory) < self.batch_size:
            return
        else:
            for _ in range(iterations):
                # Sample experience
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
                next_actions = self.target_actor.forward(
                    non_terminal_next_states
                ).detach()
                next_q_values = torch.zeros_like(states)
                next_q_values[non_terminal_mask] = self.target_critic.forward(
                    non_terminal_next_states, next_actions
                ).detach()
                expected_td_target = (
                    rewards[..., None] + self.gamma * next_q_values
                )  # y_exp = r + gamma * Q(s', pi(s'))
                predicted_td_target = self.critic.forward(
                    states, actions
                )  # y_pred = Q(s, a)
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
    wandb.init(config=config, project="baselines", tags=["DDPG"])

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
    policy = DDPG(
        n_states,
        config["network"]["hiddenSize"],
        n_actions,
        config["environment"]["actionBounds"],
        config["training"]["learningRate"],
        config["network"]["memorySize"],
        config["training"]["batchSize"],
        config["training"]["tau"],
        config["training"]["gamma"],
        config["training"]["sigma"],
    )

    # Weights & Biases watching
    wandb.watch((policy.actor, policy.critic), log="all")

    # Initial evaluation
    policy.eval(env, 0)

    for i_episode in range(config["training"]["episodes"]):
        # Initialize the environment and state
        # BCHW is torch order, but we only do BC
        state = env.reset().float().to(DEVICE).view(1, -1)

        for t in count():
            # Render environment
            if (
                config["environment"]["render"]
                and i_episode % config["environment"]["interval"] == 0
            ):
                env.render()

            # Select and perform an action
            # Exploratory random uniform action for first couple eps
            if i_episode < config["training"]["explEps"]:
                action = (
                    torch.from_numpy(env.action_space.sample())
                    .float()
                    .to(DEVICE)
                    .view(1, -1)
                )
            else:
                action = policy.get_exploration_action(state)
            next_state, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], device=DEVICE, dtype=torch.float)

            # Set to None if next state is terminal
            if not done:
                next_state = next_state.float().to(DEVICE).view(1, -1)
            else:
                next_state = None

            # Store the transition in memory
            policy.memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Episode finished
            if done:
                if i_episode > 0:
                    policy.train(t)

                    # Only for episode > 0 since we did initial eval above
                    if i_episode % config["training"]["evalInterval"] == 0:
                        policy.eval(env, i_episode)

                break

    print("Complete")
    env.close()
