import gym
import gym_mav
import matplotlib.pyplot as plt

env = gym.make("Mav-v0", gravity=0.0, action_bounds=[-10.0, 10.0], state_bounds=[[0.0, 20.0], None])
env.reset()
done = False
altitude = [env.state[0].item()]

while not done:
    # env.render()
    _, _, done, _ = env.step(1.0)
    altitude.append(env.state[0].item())

env.close()
plt.plot(range(len(altitude)), altitude)
plt.grid()
plt.show()
