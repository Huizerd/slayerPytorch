# Other
import gym
import gym_mav
import random
import argparse
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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


def make_value_map(critic, acts, obs, act_offset=0.0, encoded_obs=None, decode=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    obs, acts = torch.meshgrid(obs, acts)

    if encoded_obs is not None or decode is not None:
        assert (
            encoded_obs is not None and decode is not None
        ), "Both encoded observations and a decoder are needed."
        with torch.no_grad():
            # TODO: fix this: do we need actor here? Do we need deterministic mapping both ways for spikes <-> actions?
            q_values = decode(critic(encoded_obs))
    else:
        with torch.no_grad():
            q_values = critic(
                obs.contiguous().view(-1, 1), acts.contiguous().view(-1, 1)
            ).view(obs.size())

    ax.plot_surface(
        obs.cpu().numpy(), acts.cpu().numpy() + act_offset, q_values.cpu().numpy()
    )
    ax.set_title("Value map")
    ax.set_xlabel("Divergence")
    ax.set_ylabel("Thrust")
    ax.set_zlabel("Q-value")
    ax.grid()

    return fig


def make_value_map_plotly(
    critic, acts, obs, act_offset=0.0, encoded_obs=None, decode=None
):
    fig = go.Figure()
    obs, acts = torch.meshgrid(obs, acts)

    if encoded_obs is not None or decode is not None:
        assert (
            encoded_obs is not None and decode is not None
        ), "Both encoded observations and a decoder are needed."
        with torch.no_grad():
            # TODO: fix this: do we need actor here? Do we need deterministic mapping both ways for spikes <-> actions?
            q_values = decode(critic(encoded_obs))
    else:
        with torch.no_grad():
            q_values = critic(
                obs.contiguous().view(-1, 1), acts.contiguous().view(-1, 1)
            ).view(obs.size())

    fig.add_trace(
        go.Surface(
            x=obs.cpu().numpy(),
            y=acts.cpu().numpy() + act_offset,
            z=q_values.cpu().numpy(),
        )
    )
    fig.update_layout(
        title="Value map",
        scene=dict(
            xaxis_title="Divergence", yaxis_title="Thrust", zaxis_title="Q-value"
        ),
    )

    return fig


def make_policy_map(actor, obs, act_offset=0.0, encoded_obs=None, decode=None):
    fig, ax = plt.subplots()

    if encoded_obs is not None or decode is not None:
        assert (
            encoded_obs is not None and decode is not None
        ), "Both encoded observations and a decoder are needed."
        with torch.no_grad():
            policy = decode(actor(encoded_obs))
    else:
        with torch.no_grad():
            policy = actor(obs[:, None])

    ax.plot(obs.cpu().numpy(), policy.squeeze().cpu().numpy() + act_offset)
    ax.set_title("Policy map")
    ax.set_xlabel("Divergence")
    ax.set_ylabel("Thrust")
    ax.grid()

    return fig


def make_altitude_map(altitudes):
    fig, ax = plt.subplots()

    for altitude in altitudes:
        ax.plot(range(len(altitude)), altitude)
    ax.set_title("Altitude map")
    ax.set_xlabel("Step")
    ax.set_ylabel("Altitude")
    ax.grid()

    return fig


def make_divergence_map(divergences):
    fig, ax = plt.subplots()

    for divergence in divergences:
        ax.plot(range(len(divergence)), divergence)
    ax.set_title("Divergence map")
    ax.set_xlabel("Step")
    ax.set_ylabel("Divergence")
    ax.grid()

    return fig


def make_action_map(actions):
    fig, ax = plt.subplots()

    for action in actions:
        ax.plot(range(len(action)), action)
    ax.set_title("Action map")
    ax.set_xlabel("Step")
    ax.set_ylabel("Thrust")
    ax.grid()

    return fig


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
        self.actor_optim = optim.Adam(self.actor.parameters(), learning_rate)

        self.critic = Critic(states, hidden, actions).to(DEVICE)
        self.target_critic = Critic(states, hidden, actions).to(DEVICE)
        self.critic_optim = optim.Adam(self.critic.parameters(), learning_rate)

        self.hard_update()

    def get_exploitation_action(self, state):
        with torch.no_grad():
            action = self.actor(state)
        return action

    def get_exploration_action(self, state):
        with torch.no_grad():
            action = self.actor(state) + self.noise.sample()
        return action.clamp_(*self.action_bounds)

    def hard_update(self):
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

    def soft_update(self):
        # copy_ is not recorded in computation graph!
        # TODO: test with and without .data --> seems to be useless, except that it points to a different/copied tensor?
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

    def eval(self, env, episode, obs_space, act_space, iterations=3):
        accumulated_reward = 0.0
        duration = 0
        max_div = 0.0
        altitude_maps = []
        divergence_maps = []
        action_maps = []

        for i in range(iterations):
            altitude_map = []
            divergence_map = []
            action_map = []

            state = env.reset().float().to(DEVICE).view(1, -1)

            for t in count():
                # Select action
                action = self.get_exploitation_action(state)

                # Log maps
                # All state observations without noise (directly from env)
                divergence = (-2 * env.state[1] / env.state[0]).item()
                if abs(divergence) > abs(max_div):
                    max_div = divergence
                altitude_map.append(env.state[0].item())
                divergence_map.append(divergence)
                action_map.append(action.item() + env.action_offset)

                # Take action
                next_state, reward, done, _ = env.step(action.item())
                accumulated_reward += reward

                # Set to None if next state is terminal
                if not done:
                    next_state = next_state.float().to(DEVICE).view(1, -1)
                else:
                    next_state = None

                # Move to the next state
                state = next_state

                # Episode finished
                if done:
                    duration += t + 1
                    altitude_maps.append(altitude_map)
                    divergence_maps.append(divergence_map)
                    action_maps.append(action_map)

                    break

            wandb.log(
                {
                    "Reward": accumulated_reward / iterations,
                    "MaxDiv": max_div,
                    "Duration": duration / iterations,
                    "ValueMap": make_value_map(
                        self.critic, act_space, obs_space, env.action_offset
                    ),
                    "PolicyMap": make_policy_map(
                        self.actor, obs_space, env.action_offset
                    ),
                    "AltitudeMap": make_altitude_map(altitude_maps),
                    "DivergenceMap": make_divergence_map(divergence_maps),
                    "ActionMap": make_action_map(action_maps),
                },
                step=episode,
            )
            plt.close("all")

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
                next_q_values = torch.zeros_like(states)
                with torch.no_grad():
                    next_actions = self.target_actor(non_terminal_next_states)
                    next_q_values[non_terminal_mask] = self.target_critic(
                        non_terminal_next_states, next_actions
                    )

                expected_td_target = (
                    rewards[..., None] + self.gamma * next_q_values
                )  # y_exp = r + gamma * Q(s', pi(s'))
                predicted_td_target = self.critic(states, actions)  # y_pred = Q(s, a)

                critic_loss = F.mse_loss(predicted_td_target, expected_td_target)
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                # Optimize actor
                predicted_actions = self.actor(states)
                actor_loss = -self.critic(states, predicted_actions).mean()
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
        punish_crash=config["environment"]["punishCrash"],
        reward_finish=config["environment"]["rewardFinish"],
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
    # Input observations for value/policy maps, only for divergence control
    if env.state_obs == "divergence":
        obs_space = torch.arange(
            -10.0, 10.0, 0.1, device=DEVICE, dtype=torch.float
        )  # [:, None]
        act_space = torch.arange(
            *config["environment"]["actionBounds"],
            0.1,
            device=DEVICE,
            dtype=torch.float
        )  # [:, None]
    policy.eval(env, 0, obs_space, act_space)

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
                    if (
                        i_episode % config["training"]["evalInterval"] == 0
                        or i_episode == config["training"]["episodes"] - 1
                    ):
                        policy.eval(env, i_episode, obs_space, act_space)

                break

    print("Complete")
    env.close()
