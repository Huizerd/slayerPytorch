import timeit

setup = """
import torch
import numpy as np
import math

TSAMPLE = 500.0
TS = 1.0
TAU = 10.0
"""

naive = """
def naive_alpha_kernel(mult=1, eps_thresh=0.01):
    eps = []
    for t in np.arange(0, TSAMPLE, TS):
        eps_val = mult * t / TAU * math.exp(1 - t / TAU)
        if abs(eps_val) < eps_thresh and t > tau:
            break
        eps.append(eps_val)
    return eps
"""

fast = """
def fast_alpha_kernel(mult=1, eps_thresh=0.01):
    time = torch.arange(TS, TSAMPLE, TS)
    eps_val = mult * time / TAU * torch.exp(1 - time / TAU)
    eps_val_cut = [0.0]
    eps_val_cut.extend(eps_val[eps_val.abs() >= eps_thresh].tolist())
    return eps_val_cut
"""

if __name__ == "__main__":
    print(timeit.timeit(setup=setup, stmt=naive, number=10000))
    print(timeit.timeit(setup=setup, stmt=fast, number=10000))
