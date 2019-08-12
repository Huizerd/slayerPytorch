import torch, pdb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from vertical import place_cell_centers, encode, DEVICE

torch.manual_seed(0)

low = -10.0
high = 10.0
states = torch.arange(-10.0, 10.0, 0.05, device=DEVICE).view(-1, 1, 1)
centers, width = place_cell_centers([[low, high]], [11])
ts = 1.0  # ms
time = 500  # ms
steepness = 10.0
process = "transform"
max_rate = 800.0

spikes, rates = encode(states, centers, width, max_rate, steepness, time, ts, process)
fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(111, projection="3d")

ax.plot_wireframe(
    centers.repeat(states.size(0), 1, 1).squeeze().cpu().numpy(),
    states.repeat(1, centers.size(1), 1).squeeze().cpu().numpy(),
    spikes.mean(-1).squeeze().cpu().numpy() * 1000,
    label="Counts",
    rstride=10000,
    color="b",
)
ax.plot_wireframe(
    centers.repeat(states.size(0), 1, 1).squeeze().cpu().numpy(),
    states.repeat(1, centers.size(1), 1).squeeze().cpu().numpy(),
    rates[..., 0].squeeze().cpu().numpy(),
    label="Rates",
    rstride=10000,
    color="r",
)
ax.set_ylabel("Divergence")
ax.set_xlabel("Centers")
ax.set_zlabel("Firing rate")

plt.show()
pdb.set_trace()
