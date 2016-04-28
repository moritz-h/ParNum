import numpy as np
from math import ceil, floor
import copy
import timeit, sys
import matplotlib.pyplot as plt
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
psqrt = int(np.sqrt(size))
N = 9

def Print(*s):
    ''' Prints with MPI rank and blocks to keep the output together. Be careful when used inside loops.'''
    PrintNB(*s)
    MPI.COMM_WORLD.Barrier() # Just to keep the output together

def PrintNB(*s):
    out = " ".join( [ str(i) for i in s] )
    print("[%s] %s" % (rank, out))

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

'''
Matrix in numpy notation: mat[y][x]
enumerate like that:
    y0
    |
    |
    ymax
        x0-----xmax
'''
def rank_pos():
    """ Returns (row, col) position in the matrix for our rank. """
    return (int(rank % psqrt), floor(rank/psqrt))

def get_rank(x, y):
    """ Inverse operation of rank_pos. """
    return x + y*psqrt

def get_left_rank():
    if rank % psqrt == 0:
        return rank + psqrt - 1
    else:
        return rank-1

def get_right_rank():
    if (rank+1) % psqrt == 0:
        return rank - psqrt + 1
    else:
        return rank+1

def get_upper_rank():
    row = rank // psqrt
    if row == 0:
        return psqrt*psqrt - (psqrt - rank)
    else:
        return rank - psqrt

def get_below_rank():
    row = rank // psqrt
    if row == psqrt-1: # last row
        return rank % psqrt
    else:
        return rank + psqrt

# Create matrices.
A, B = get_matrices(N)

# Ensure that both matrices can be distributed equally among the number of processors
assert(N % psqrt == 0)

# Ensure that the number of procs is a squared number
assert(np.sqrt(size) == psqrt)

# Calculate the blocksize. In fact we mean block length here.
blockSize = int(N / psqrt)

# Find out which slice of the matrix is ours w.r.t. the current process
x, y = rank_pos()
assert(rank == get_rank(x, y))
# print ("rank:", rank, "- calculated rank: ", get_rank(x, y))
# print ("rank:", rank,"- coords: (x =", x, "- y =", y, ") - left", get_left_rank(), "- right", get_right_rank(), "- top", get_upper_rank(), "- bottom", get_below_rank())

# print some informations on master node
if rank == 0:
    print("Size of one sub block = ", blockSize, "- Number of blocks:", psqrt)
    print("Matrix A:\n", A)
    print("Matrix B:\n", B)
    print("Result of AxB using numpy.dot:\n", np.dot(A, B))
comm.Barrier()

requests = []

# Local matrices
lA = np.zeros(blockSize*blockSize, dtype=np.int).reshape(blockSize, blockSize)
lB = np.zeros(blockSize*blockSize, dtype=np.int).reshape(blockSize, blockSize)

# Distribute matrices
if rank == 0:
    tmpA = np.zeros(blockSize*blockSize).reshape(blockSize, blockSize)
    tmpB = np.zeros(blockSize*blockSize).reshape(blockSize, blockSize)

    # Iterate through procs, skew before sending
    for xIdx in range(psqrt):
        for yIdx in range(psqrt):
            k = (xIdx + yIdx) % psqrt
            print ("x=", xIdx, "y=", yIdx, "k=", k, "dest=", get_rank(xIdx, yIdx))
            tmpA = A[blockSize*xIdx : blockSize*(xIdx+1), blockSize*k : blockSize*(k+1)]
            tmpB = B[blockSize*k : blockSize*(k+1),       blockSize*yIdx : blockSize*(yIdx+1)]
            # perform some shifting such that we can use MPI Scatter
            dst = get_rank(yIdx, xIdx)
            if dst != 0:
                requests.append(comm.Isend(buf=np.ascontiguousarray(tmpA), dest=dst, tag=1))
                requests.append(comm.Isend(buf=np.ascontiguousarray(tmpB), dest=dst, tag=2))
            else:
                lA = np.copy(tmpA)
                lB = np.copy(tmpB)
else:
    comm.Recv(lA, source=0, tag=1)
    comm.Recv(lB, source=0, tag=2)

MPI.Request.Waitall(requests)
comm.Barrier()

lC = np.zeros_like(lA)
# Print ("a:\n", lA, "\nb:\n", lB, "\n")#, "\nc:\n", lC, "\n")
for block in range(psqrt -1):
    lC += np.dot(lA, lB)
    # PrintNB("lC\n", lC,"\n \n")
    temp = np.zeros_like(lC)
    # PrintNB("Communicate local A to", get_left_rank(), "receiving from ", get_right_rank())
    comm.Sendrecv(sendbuf=np.ascontiguousarray(lA), dest=get_left_rank(), recvbuf=temp, source=get_right_rank())
    lA = np.copy(temp)
    # PrintNB("lA\n", lA,"\n \n")
    # PrintNB("Finished sending")
    # PrintNB("Communicate local B to", get_upper_rank(), "receiving from ", get_below_rank())
    comm.Sendrecv(sendbuf=np.ascontiguousarray(lB), dest=get_upper_rank(), recvbuf=temp, source=get_below_rank())
    lB = np.copy(temp)
    # PrintNB("lB\n", lB,"\n \n")

# PrintNB("lAneu", lA,"\n \n")
# PrintNB("lBneu", lB,"\n \n")
# PrintNB("lCneu", np.dot(lA, lB),"\n \n")
lC += np.dot(lA, lB)

Print (lC, "\n")
