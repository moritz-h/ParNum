import numpy as np
from numpy.linalg import inv, qr

def fnorm(A):
    return np.linalg.norm(A, ord="fro")

def get_tridiag_matrix(size):
    """ Returns a matrix with the diagonals filled with 1, -2, 1. """
    return np.diagflat(np.full(size,-2, np.double), 0 ) + \
        np.diagflat(np.full(size-1,1,np.double), 1) + \
        np.diagflat(np.full(size-1,1,np.double), -1 )

def reduce(A, Mk):
    """ Reduces matrix A and vector Mk according to the pattern of Mk. """
    # Selects all cols from A where Mk is not zero
    # Selects all values from Mk which are not zero
    return A[:, Mk != 0], Mk[Mk != 0]

def get_pattern(A):
    """ Returns matrix M defining the pattern, here it's the same pattern as A. """
    M = A.copy()
    M[ M !=0 ] = 1 # Selects all values not equal 0 and sets them to 1.
    return M

def add_to_reconstructed(M, Mkt, col):
    """ Adds Mkt to the reconstruct"""
    M[M[:,col] != 0, col] = Mkt # Replaces the non-zero entries in the pattern with the actual values from Mkt

def solve_for_Mk(A, Mk, ek):
    Q, R = qr(A)
    return inv(R) @ Q.T @ ek

def spai(A):
    M = get_pattern(A)
    for i, Mk in enumerate(M.T): # Iterate through all columns
        At, Mkt = reduce(A, Mk)
        ek = np.zeros(A.shape[0])
        ek[i] = 1
        Mkt = solve_for_Mk(At, Mkt, ek)
        add_to_reconstructed(M, Mkt, i)

    return M

def main():
    size = 1000
    A = get_tridiag_matrix(size)
    M = spai(A)

    print("||A-M|| = ", fnorm(A - M))
    print("||A-I|| = ", fnorm(A - np.identity(size)))
    print("||A M - I|| = ", fnorm(A @ M - np.identity(size)))
    print("k(A) = ", np.linalg.cond(A))
    print("k(A M) = ", np.linalg.cond(A@M))

    if size < 10:
        print("====== A ======\n", A)
        print("====== M ======\n", M)

if __name__ == "__main__":
    main()

