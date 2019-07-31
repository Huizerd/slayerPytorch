from vertical_ddpg import OrnsteinUhlenbeckActionNoise
import torch
import matplotlib.pyplot as plt

if __name__ == "__main__":
    t = range(10000)
    noise = OrnsteinUhlenbeckActionNoise(1)
    noise_list = []
    for i in t:
        noise_list.append(noise.sample().item())

    plt.plot(t, noise_list)
    plt.show()
