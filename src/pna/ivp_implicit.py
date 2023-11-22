import numpy as np
from scipy.optimize import root


def implicit_solve_to(f, y0, t_span, dt, method):
    """Solves ODE using the implicit euler method / Crank Nicolson.

    Parameters
    ----------
    f : Callable
        ODE to solve. Must have signature f(t, y) -> array_like
    y0 : np.ndarray
        Initial condition
    t_span : tuple[float, float]
        Time span to solve over
    dt : float
        Step size to use
    method : str
        method to use. Options are "euler", "CN"

    Returns
    -------
    y
        Solution to f
    t
        Time corresponding to solution y
    """
    t = [t_span[0]]
    y = [np.asarray(y0)]

    # define function to find root of
    if method == "euler":

        def f_root(y_next):
            return y_next - y[-1] - dt * f(t[-1] + dt, y_next)

    elif method == "CN":

        def f_root(y_next):
            return y_next - y[-1] - 0.5 * dt * (f(t[-1] + dt, y_next) + f(t[-1], y[-1]))

    else:
        raise ValueError("Invalid Method")

    # Check if integration is finished
    while (t[-1] - t_span[-1]) < 0:
        # check if the step is going to overshoot and adjust accordingly
        if (t[-1] + dt - t_span[-1]) > 0:
            dt = t_span[-1] - t[-1]

        # compute next value at the next time step
        res = root(f_root, x0=y[-1], tol=1e-12)

        y.append(res.x)
        t.append(t[-1] + dt)

    return np.asarray(y).T, np.asarray(t)
