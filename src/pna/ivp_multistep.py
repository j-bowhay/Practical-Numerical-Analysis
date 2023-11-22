import warnings

import numpy as np
import scipy

from pna.ivp_explicit import solve_to_fixed_step


def solve_to_multistep(f, y0, t_span, dt, method, initial_method):
    """Solves ODE using a linear multistep method.

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
        method to use. Options are "AB" for (explicit) Adams Bashforth schemes
        and "AM" for (implicit) Adams Moulton schemes
    initial_method : str
        Method to use to start the method. Options are "euler", "huen", "rk4".

    Returns
    -------
    y
        Solution to f
    t
        Time corresponding to solution y
    """
    num_initial = 3 if method == "AB" else 2
    # check we aren't going to over shoot in the setup
    if t_span[0] + num_initial * dt < t_span[1]:
        t_span_initial = (t_span[0], t_span[0] + num_initial * dt)
    else:
        t_span_initial = t_span
        warnings.warn("With specified time step problem is solved entirely in setup")

    y, t = solve_to_fixed_step(f, y0, t_span_initial, dt=dt, method=initial_method)

    # Check if integration is finished
    while (t[-1] - t_span[-1]) < 0:
        # check if the step is going to overshoot and adjust accordingly
        if (t[-1] + dt - t_span[-1]) > 0:
            dt = t_span[-1] - t[-1]

        if method == "AB":
            y_next = y[:, -1] + (dt / 24) * (
                55 * f(t[-1], y[:, -1])
                - 59 * f(t[-2], y[:, -2])
                + 37 * f(t[-3], y[:, -3])
                - 9 * f(t[-4], y[:, -4])
            )

        elif method == "AM":

            def func(y_guess):
                return np.squeeze(
                    y_guess
                    - (
                        y[:, -1]
                        + (dt / 24)
                        * (
                            9 * f(t[-1] + dt, y_guess)
                            + 19 * f(t[-1], y[:, -1])
                            - 5 * f(t[-2], y[:, -2])
                            + f(t[-3], y[:, -3])
                        )
                    )
                )

            y_next = scipy.optimize.root(func, x0=y[:, -1]).x

        y = np.hstack([y, np.atleast_2d(y_next)])
        t = np.hstack([t, t[-1] + dt])

    return np.asarray(y).T, np.asarray(t)
