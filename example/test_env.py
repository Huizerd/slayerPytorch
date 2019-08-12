import gym
import gym_mav
import matplotlib.pyplot as plt

env = gym.make("Mav-v0", gravity=9.81, action_bounds=[-10.0, 10.0], state_bounds=[[0.1, 2.0], None], init_state=[2.0, 0.0], init_rand=[0.0, 0.0])
env.reset()
done = False
div = [-2.0 * (env.state[1] / env.state[0]).item()]

while not done:
    # env.render()
    _, _, done, _ = env.step(0.0)
    div.append(-2.0 * (env.state[1] / env.state[0]).item())

env.close()
plt.plot(range(len(div)), div)
plt.title("Divergence")
plt.grid()
plt.show()
