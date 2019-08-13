import torch
import matplotlib.pyplot as plt

torch.manual_seed(1)

def add_noise(x, scale):
    return x + torch.randn_like(x) * scale


if __name__ == "__main__":
    acc = 9.81
    time = torch.arange(0.0, 0.6 + 0.01, 0.01)
    alt = (2.0 - 0.5 * acc * time ** 2)
    vs = acc * time
    div = 2.0 * vs / alt

    std = 0.2
    div_x = add_noise(div, scale=std)
    div_1x = 2.0 * vs / add_noise(alt, scale=std)

    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(10, 10), dpi=120)
    ax[0].plot(time.numpy(), alt.numpy())
    ax[1].plot(time.numpy(), div.numpy())
    ax[2].plot(time.numpy(), div_x.numpy(), label="x noise (div)")
    ax[2].plot(time.numpy(), div_1x.numpy(), label="1/x noise (alt)")
    ax[3].plot(time.numpy(), (div - div_x).abs().numpy(), label="x noise (div)")
    ax[3].plot(time.numpy(), (div - div_1x).abs().numpy(), label="1/x noise (alt)")
    ax[0].set_ylabel("Altitude")
    ax[1].set_ylabel("Clean div")
    ax[2].set_ylabel("Noisy div")
    ax[2].legend()
    ax[3].set_ylabel("Noise")
    ax[3].set_xlabel("Time")
    ax[3].legend()
    for a in ax: a.grid()
    plt.tight_layout()
    plt.show()
