""" Compares naive and numpy (BLAS based) matrix-matrix multiplication. """

import numpy as np
import timeit
import matplotlib.pyplot as plt

N = 200

def time(stmt, number = 1):
    setup = "from __main__ import A, B, naive_multiply, numpy_multiply"
    tg = timeit.timeit(stmt, setup, number = number)
    return tg / number

def naive_multiply(A, B):
    """ Naive matrix matrix multiplication """
    C = np.zeros_like(A)
    n = C.shape[0]
    for row in range(n):
        for col in range(n):
            for i in range(n):
                C[row, col] += A[row,i]*B[i,col]
    return C

def numpy_multiply(A, B):
    """ Multiply using NumPy """
    return np.dot(A, B)

def get_matrices(n):
    A = np.array(range(n*n)).reshape(n, n)
    B = np.array(list(reversed(range(n*n)))).reshape(n, n)
    return A, B



sizes = np.linspace(1, N, 50)
times_naive = []
times_numpy = []

for s in sizes:
    size = int(s)
    A, B = get_matrices(size)
    print("Size = ", size)
    times_naive.append(time("naive_multiply(A, B)", number = 1))
    times_numpy.append(time("numpy_multiply(A, B)", number = 1))

# plt.semilogy(sizes, times_naive, label = "Naive")
# plt.semilogy(sizes, times_numpy, label = "Numpy")
plt.loglog(sizes, times_naive, "d-", label = "Naive")
plt.loglog(sizes, times_numpy, "d-", label = "Numpy")
plt.legend()
plt.grid()
plt.show()
