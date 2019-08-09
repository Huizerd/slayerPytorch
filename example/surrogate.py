import numpy as np
import matplotlib.pyplot as plt


def surrogate_exp(u, theta, alpha, tau):
    return alpha / tau * np.exp(-np.abs(u - theta) / tau)


def surrogate_pwl(u, theta):
    return np.maximum(np.zeros_like(u), 1.0 - 2.0 * np.abs(u - theta))


if __name__ == "__main__":
    # Hypers as in slayer.yaml
    scaleRho = 1.0
    tauRho = 1.0
    theta = 10.0

    # Hypers as we use them
    alpha = scaleRho
    theta = theta
    tau = tauRho * theta

    # Inputs
    u = np.linspace(0.0, theta * 2, 1001)

    # Original paper function
    original = surrogate_exp(u, theta, alpha, tau)

    # Our proposed one
    tauRho = 0.05
    tau = tauRho * theta
    ours = surrogate_exp(u, theta, alpha, tau)

    # Piecewise linear that sums to 1 (like proper PDF)
    proper = surrogate_pwl(u, theta)

    # Plot
    plt.plot(u - theta, original, label="Original")
    plt.plot(u - theta, ours, label="Ours")
    plt.plot(u - theta, proper, label="Proper")
    plt.legend()
    plt.show()
