import numpy as np
import scipy


def theta_method(u0, dx, dt, steps, theta, left_bc, right_bc):
    """Solves the diffusion equation using the theta method.

    Parameters
    ----------
    u0 : np.ndarray
        Initial conditions
    dx : float
        spatial step
    dt : float
        time step
    steps : int
        Number of steps to take
    theta : float
        Controls interpolation between implicit and explicit euler, must be in [0,1]
    left_bc : Callable
        Function for left BC
    right_bc : Callable
        Function for the right BC

    Returns
    -------
    u
        Array of solution
    t
        Times solution is given at
    """
    N = len(u0) - 2
    u = np.empty((steps + 1, N))
    u[0, :] = u0[1:-1]
    t = [0]

    weights = [1, -2, 1]
    offsets = [-1, 0, 1]
    diags = [weights[i] * np.ones(N - abs(offset)) for i, offset in enumerate(offsets)]
    A = scipy.sparse.diags(diags, offsets, format="csr")
    b = np.zeros_like(u0[1:-1])
    I = scipy.sparse.eye(N)
    C = dt / dx**2

    lhs = I - theta * C * A

    for i in range(steps):
        t_new = t[-1] + dt
        b[0] = left_bc(t_new)
        b[-1] = right_bc(t_new)

        rhs = (I + (1 - theta) * C * A) @ u[i, :] + C * b
        u[i + 1, :] = scipy.sparse.linalg.spsolve(lhs, rhs)
        t.append(t_new)

    return u, t
