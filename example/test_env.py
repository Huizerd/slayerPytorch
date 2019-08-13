import gym
import gym_mav
import torch
import matplotlib.pyplot as plt

env = gym.make("Mav-v0", gravity=9.81, action_bounds=[-10.0, 10.0], state_bounds=[[0.1, 2.0], None], init_state=[2.0, 0.0], init_rand=[0.0, 0.0])
env.reset()
done = False
div = []

while not done:
    # env.render()
    div.append(-2.0 * (env.state[1] / env.state[0]).item())
    _, _, done, _ = env.step(0.0)

env.close()
# Difference with logdiv: rounding/floating point errors!
plt.plot(torch.arange(0.0, env.steps * env.step_duration, env.step_duration).numpy(), div)
plt.title("Divergence")
plt.grid()
plt.show()
