import timeit

setup = """
import torch

tens = torch.arange(0.0, 1000.0, 0.01)
"""

numpy = """
tens2 = tens.numpy()
"""

lst = """
tens2 = tens.tolist()
"""

if __name__ == "__main__":
    print(timeit.timeit(setup=setup, stmt=numpy, number=10000))
    print(timeit.timeit(setup=setup, stmt=lst, number=10000))
