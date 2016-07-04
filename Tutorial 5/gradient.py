import numpy as np
import scipy.linalg
import scipy.sparse.linalg
import time
import matplotlib.pyplot as plt
import spai

tolerance = 1e-5
max_iter = 4000
N = 100


def CG(A, x, b, P):
    """ Preconditioned Conjugate Gradients algorithm, see p. 94 in script. """
    iter = 1
    residuals = []
    # P = np.linalg.inv(np.diagflat(np.diag(A))) # Jacobi Preconditoner P = diag(A)^-1

    r0 = b - A @ x # For the first iteration search direction is like with gradient methods
    h0 = P @ r0
    p = h0.copy()
        # Klammerung expliziter
    while True:
        a = (r0 @ h0) / (p @ A @ p) # Step size
        x = x + a * p # New solution
        r1 = r0 - (a * A @ p)
        h1 = P @ r1 # Apply preconditioner
        residuals.append(np.linalg.norm(r1))
        if np.linalg.norm(b - A @ x) < tolerance:
            print("CG: Converged after %i iterations." % iter)
            break
        elif iter > max_iter:
            print("CG: Maximum iterations reached. Last residual: %f" % np.linalg.norm(b - A @ x))
            break
        beta = (r1 @ h1) / (r0 @ h0)
        p = h1 + beta * p # Search direction
        iter += 1
        r0 = r1.copy()
        h0 = h1.copy()
    return x, residuals


def gradient(A, x, b):
    iter = 1
    residuals = []
    r = b - A @ x # Residual
    while True:
        a = (r @ np.transpose(r)) / (np.transpose(r) @ A @ r) # Step size
        x = x + a * r # New solution = old solution + step size * residual
        r = b - A @ x # New residual
        residuals.append(np.linalg.norm(r))
        if np.linalg.norm(r) < tolerance:
            print("Gradient: Converged after %i iterations." % iter)
            break
        elif iter > max_iter:
            print("Gradient: Maximum iterations reached. Last residual: %f" % np.linalg.norm(b - A @ x))
            break
        iter += 1
    return x, residuals


# A = np.random.rand(N, N)
# A = A @ np.transpose(A) + 4*np.identity(N) # A*A^T will get us a SPD matrix which is diagonal-dominant
A = - spai.get_tridiag_matrix(N)
print(A)

x0 = np.zeros(N) # Start vector
b = np.array(range(1, N+1))

ev = scipy.linalg.eig(A)[0]
if np.any(np.real(ev) < 0):
    print("Negative Eigenvalues, matrix is not semi-positive definite!")

# === SCIPY ===
t = time.perf_counter()
sp_solution = scipy.linalg.solve(A,b) # Hint: sym_pos=True
t_sp = time.perf_counter()-t

# === CG ===
t = time.perf_counter()
cg_solution, cg_residuals = CG(A, x0, b, np.identity(N))
t_cg =  time.perf_counter()-t

# === SPAI CG ====
t = time.perf_counter()
M = spai.spai(A)
t_onlyspai = time.perf_counter()-t
spai_solution, spai_residuals = CG(A, x0, b, M)
t_spai =  time.perf_counter()-t

# === Perfect CG (inverse is preconditioner) ====
t = time.perf_counter()
M = scipy.linalg.inv(A)
perfect_solution, perfect_residuals = CG(A, x0, b, M)
t_perfect =  time.perf_counter()-t

# === Gradient ===
t = time.perf_counter()
gr_solution, gr_residuals = gradient(A, x0, b)
t_gr = time.perf_counter()-t

print()
print("Time  SP           = ", t_sp)
print("Time  CG           = ", t_cg)
print("Time SPAI CG       = ", t_spai)
print("  Time SPAI        = ", t_onlyspai)
print("Time PERFECT CG    = ", t_perfect)
print("Time  GR           = ", t_gr)
print()
print("Delta Conjugate Gradient                        = ", np.linalg.norm(cg_solution - sp_solution))
print("Delta SPAI Preconditioned Conjugate Gradient    = ", np.linalg.norm(spai_solution - spai_solution))
print("Delta PERFECT Preconditioned Conjugate Gradient = ", np.linalg.norm(perfect_solution - perfect_solution))
print("Delta Gradient                                  = ", np.linalg.norm(gr_solution - sp_solution))

plt.plot(range(len(cg_residuals)), cg_residuals, "rx-", label="CG Residuals")
plt.plot(range(len(spai_residuals)), spai_residuals, "gx-", label="SPAI Preconditioned Gradient Residuals")
plt.plot(range(len(perfect_residuals)), perfect_residuals, "yx-", label="PERFECT Preconditioned Gradient Residuals")
plt.plot(range(len(gr_residuals)), gr_residuals, "bx-", label="Gradient Residuals")
plt.legend()
plt.grid()
plt.show()
