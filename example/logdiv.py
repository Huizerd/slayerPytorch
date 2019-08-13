import torch
import matplotlib.pyplot as plt


def take_log(divs, offset=0.0):
    return torch.log(divs + offset)


if __name__ == "__main__":
    acc = 9.81
    time = torch.arange(0.0, 0.6 + 0.01, 0.01)
    alt = (2.0 - 0.5 * acc * time ** 2)
    vs = acc * time
    div = 2.0 * vs / alt
    log_div = take_log(div, offset=0.0)

    fig, ax = plt.subplots(3, 1, sharex=True, dpi=120)
    ax[0].plot(time.numpy(), alt.numpy())
    ax[1].plot(time.numpy(), div.numpy())
    ax[2].plot(time.numpy(), log_div.numpy())
    ax[0].set_ylabel("Altitude")
    ax[1].set_ylabel("Div")
    ax[2].set_ylabel("Log div")
    ax[2].set_xlabel("Time")
    for a in ax: a.grid()
    plt.tight_layout()
    plt.show()
