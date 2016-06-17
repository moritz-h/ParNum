import numpy as np
import scipy.linalg
from numpy import dot
import matplotlib.pyplot as plt

tolerance = 1e-3
max_iter = 100
N = 40
cg_residuals = []
gr_residuals = []

def CG(A, x0, b):
    iter = 1
    # P = np.identity(N) # Preconditioner
    P = np.linalg.inv(np.diagflat(np.diag(A))) # Jacobi Preconditoner P = diag(A)^-1
    print(P)
    print(A)

    r0 = b - dot(A, x0) # For the first iteration search direction is like with gradient methods
    h0 = dot(P, r0)
    p0 = h0.copy()
        
    while True:
        a0 = (dot(r0, h0)) / (dot(p0, dot(A, p0))) # Search direction
        x1 = x0 + dot(a0, p0)
        r1 = r0 - dot(a0, dot(A, p0))
        h1 = dot(P, r1) # Apply preconditioner
        cg_residuals.append(np.linalg.norm(r1))
        if np.linalg.norm(b-dot(A,x1)) < tolerance or iter > max_iter:
            break
        b0 = dot(r1, h1) / dot(r0, h0)
        p1 = h1 + dot(b0, p0)        
        iter += 1
        r0 = r1.copy()
        p0 = p1.copy()
        x0 = x1.copy()
        h0 = h1.copy()
    return x1

def gradient(A, x0, b):
    iter = 1
    r0 = b - dot(A, x0)
    while True:
        a0 = dot(r0, np.transpose(r0)) / dot(np.transpose(r0), dot(A, r0))
        x1 = x0 + dot(a0, r0)
        r1 = b - dot(A,x1) 
        gr_residuals.append(np.linalg.norm(r1))
        if np.linalg.norm(r1) < tolerance or iter > max_iter:
            break
        iter += 1
        r0 = r1.copy()
        x0 = x1.copy()
    return x1


A = np.random.rand(N, N)
A = np.dot(A, np.transpose(A)) + 4*np.identity(N) # A*A^T will get us a SPD matrix
x0 = np.zeros(N)
b = np.array(range(1, N+1))

ev = scipy.linalg.eig(A)[0]
if np.any(np.real(ev) < 0):
    print("Negative Eigenvalues, matrix is not semi-positive definite!")
    
sp_solution = scipy.linalg.solve(A,b)
cg_solution = CG(A, x0, b)
gr_solution = gradient(A, x0, b)

print("Delta Conjugate Gradient = ", np.linalg.norm(cg_solution - sp_solution))
print("Delta Simple Gradient = ", np.linalg.norm(gr_solution - sp_solution))

plt.plot(range(len(cg_residuals)), cg_residuals, "rx-")
plt.plot(range(len(gr_residuals)), gr_residuals, "bx-")
plt.show()
