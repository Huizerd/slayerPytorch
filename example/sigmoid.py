import torch
import matplotlib.pyplot as plt
from vertical import sigmoid

if __name__ == "__main__":
    low = -2.0
    high = 2.0
    x = torch.arange(low, high + 0.01, 0.05)
    plt.plot(
        x.tolist(),
        sigmoid(
            x,
            y_min=torch.tensor(low),
            y_step=torch.tensor(high - low),
            x_mid=torch.tensor(0.0),
            steepness=10.0 / high,
        ).tolist(),
    )
    plt.grid()
    plt.show()
