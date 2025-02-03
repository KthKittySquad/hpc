import numpy as np
import matplotlib.pyplot as plt
import timeit


def gauss_seidel(f):
    newf = f.copy()

    for i in range(1, newf.shape[0] - 1):
        for j in range(1, newf.shape[1] - 1):
            newf[i, j] = 0.25 * (
                newf[i, j + 1] + newf[i, j - 1] + newf[i + 1, j] + newf[i - 1, j]
            )

    return newf


def gen_matrix(n):
    f = np.random.rand(n, n)
    f[0, :] = 0
    f[-1, :] = 0
    f[:, 0] = 0
    f[:, -1] = 0

    return f


grid_size = 100
x = gen_matrix(grid_size)

for i in range(1000):
    x = gauss_seidel(x)
