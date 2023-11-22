import numpy as np

# define the explicit methods in terms of their butch tableau


class _EulerStep:
    """Defines Butcher Tableau for the Forward Euler method."""

    A = np.array([[0]])
    B = np.array([1])
    C = np.array([0])


class _HeunsStep:
    """Defines the Butcher Tableau for Huan's method."""

    A = np.array([[0, 0], [1, 0]])
    B = np.array([0.5, 0.5])
    C = np.array([0, 1])


class _RK4Step:
    """Butcher Tableau for the classic fourth-order Rung-Kutta method."""

    A = np.array([[0, 0, 0, 0], [0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 1, 0]])
    B = np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6])
    C = np.array([0, 0.5, 0.5, 1])


_fixed_step_methods = {
    "euler": _EulerStep(),
    "heun": _HeunsStep(),
    "rk4": _RK4Step(),
}


def solve_to_fixed_step(f, y0, t_span, dt, method):
    """Solves ODE using a fixed timestep.

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
        method to use. Options are "euler", "huen", "rk4"

    Returns
    -------
    y
        Solution to f
    t
        Time corresponding to solution y
    """
    t = [t_span[0]]
    y = [np.asarray(y0)]
    method = _fixed_step_methods[method]

    # Check if integration is finished
    while (t[-1] - t_span[-1]) < 0:
        # check if the step is going to overshoot and adjust accordingly
        if (t[-1] + dt - t_span[-1]) > 0:
            dt = t_span[-1] - t[-1]

        # compute next value at the next time step
        s = method.B.size
        k = np.empty((y[-1].size, s))
        # calculate k values
        for i in range(s):
            # k_i = f(t + c_i*h, y + h * sum_{j=1}^{i-1} a_ij*k_j)
            k[:, i] = f(
                t[-1] + method.C[i] * dt,
                y[-1]
                + dt
                * np.sum(method.A[i, np.newaxis, : i + 1] * k[:, : i + 1], axis=-1),
            )

        # y_{n+1} = y_n + h sum_{i=1}^s b_i*k_i
        y.append(y[-1] + dt * np.inner(method.B, k))
        t.append(t[-1] + dt)

    return np.asarray(y).T, np.asarray(t)
