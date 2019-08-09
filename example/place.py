import torch, pdb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from vertical import place_cell_centers, encode, DEVICE

torch.manual_seed(0)

low = -10.0
high = 10.0
states = torch.arange(-40.0, 40.0, 0.05, device=DEVICE)
centers, width = place_cell_centers([[low, high]], [41])
ts = 1.0  # ms
time = 200  # ms
steepness = 10.0
process = "transform"
max_rate = 1000.0
spikes = torch.zeros(states.size(0), centers.size(0), device=DEVICE)
rates = torch.zeros(states.size(0), centers.size(0), device=DEVICE)

for i in range(states.size(0)):
    spike, rate = encode(
        states[i], centers, width, [[low, high]], max_rate, steepness, time, ts, process
    )
    spikes[i] = spike.squeeze().mean(-1) * 1000
    rates[i] = rate.squeeze()[:, 0]

fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(111, projection="3d")

ax.plot_wireframe(
    centers.repeat(1, states.size(0)).permute(1, 0).cpu().numpy(),
    states[:, None].repeat(1, centers.size(0)).cpu().numpy(),
    spikes.cpu().numpy(),
    label="Counts",
    rstride=10000,
    color="b",
)
ax.plot_wireframe(
    centers.repeat(1, states.size(0)).permute(1, 0).cpu().numpy(),
    states[:, None].repeat(1, centers.size(0)).cpu().numpy(),
    rates.cpu().numpy(),
    label="Rates",
    rstride=10000,
    color="r",
)
ax.set_ylabel("State")
ax.set_xlabel("Centers")
ax.set_zlabel("Firing rate")

plt.show()
pdb.set_trace()
