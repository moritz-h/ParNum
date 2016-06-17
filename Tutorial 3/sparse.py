import numpy as np
import scipy.sparse
import timeit
import matplotlib.pyplot as plt

N = 150

def time(stmt, number = 1):
    setup = "from __main__ import A, CSR, CSC, COO, DIA, b, multiply"
    tg = timeit.timeit(stmt, setup, number = number)
    return tg / number

def multiply(A, b):
    """ Multiply using NumPy """
    return A.dot(b)

def get_matrices(n):
    """ Creates matrix of size n**2 """
    A = np.diag(np.full(n**2, 4)) + np.diag(np.full(n**2-1, 1), k=1) + np.diag(np.full(n**2-1, 1), k=-1) + np.diag(np.full(n**2 -n, 1), k=n) + np.diag(np.full(n**2 -n, 1), k=-n)
    return A, scipy.sparse.csr_matrix(A), scipy.sparse.csc_matrix(A), scipy.sparse.coo_matrix(A), scipy.sparse.dia_matrix(A)


sizes = np.linspace(1, N, 50)
times_dense = []
times_csr = []
times_csc = []
times_coo = []
times_dia = []

for s in sizes:
    size = int(s)
    A, CSR, CSC, COO, DIA = get_matrices(size)
    b = np.array(range(size**2))
    print("Size = ", size**2)
    print("Number of non-zeros = ", CSR.nnz)
    print("Ratio  of non-zeros = ", size**2 / CSR.nnz)
    times_dense.append(time("multiply(A, b)", number = 1))
    times_csr.append(time("multiply(CSR, b)", number = 1))
    times_csc.append(time("multiply(CSC, b)", number = 1))
    times_coo.append(time("multiply(COO, b)", number = 1))
    times_dia.append(time("multiply(DIA, b)", number = 1))

sizes = sizes**2 # Correct sizes for display
    
plt.loglog(sizes, times_dense, label = "Dense")
plt.loglog(sizes, times_csr, label = "Sparse (CSR)")
plt.loglog(sizes, times_csc, label = "Sparse (CSC)")
plt.loglog(sizes, times_coo, label = "Sparse (COO)")
plt.loglog(sizes, times_dia, label = "Sparse (DIA)")
plt.legend()
plt.grid()
plt.show()
