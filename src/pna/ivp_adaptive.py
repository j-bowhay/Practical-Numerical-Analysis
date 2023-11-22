import math

import numpy as np

from pna.ivp_explicit import solve_to_fixed_step


class _RKF45Step:
    """Butcher Tableau for the Runge-Kutta-Fehlberg method
    https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta%E2%80%93Fehlberg_method
    """

    A = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [1 / 4, 0, 0, 0, 0, 0],
            [3 / 32, 9 / 32, 0, 0, 0, 0],
            [1932 / 2197, -7200 / 2197, 7296 / 2197, 0, 0, 0],
            [439 / 216, -8, 3680 / 513, -845 / 4104, 0, 0],
            [-8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40, 0],
        ]
    )
    B_hat = np.array([16 / 135, 0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55])
    B = np.array([25 / 216, 0, 1408 / 2565, 2197 / 4104, -1 / 5, 0])
    C = np.array([0, 1 / 4, 3 / 8, 12 / 13, 1, 1 / 2])
    order = 4


class _DomandPrinceStep:
    """Butcher Tableau for the Dormand-Prince method
    https://en.wikipedia.org/wiki/Dormand%E2%80%93Prince_method
    """

    A = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [1 / 5, 0, 0, 0, 0, 0, 0],
            [3 / 40, 9 / 40, 0, 0, 0, 0, 0],
            [44 / 45, -56 / 15, 32 / 9, 0, 0, 0, 0],
            [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0, 0],
            [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0, 0],
            [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0],
        ]
    )
    B_hat = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0])
    B = np.array(
        [5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40]
    )
    C = np.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1])
    order = 4


_embedded_methods = {
    "rkf45": _RKF45Step,
    "dopri45": _DomandPrinceStep,
}


def _scale(r_tol, a_tol, *args):
    """Calculates the scale term for the error norm. Based on eq 4.10 from Hairer.

    Parameters
    ----------
    r_tol : float
        The desired relative tolerance
    a_tol : float
        The desired absolute tolerance

    Returns
    -------
    np.ndarray
        Array containing the scale term for each component of the solution.
    """
    if len(args) > 1:
        # Take maximum of input args to avoid near division by zero later on
        y = np.amax(np.abs([np.squeeze(arg) for arg in args]), axis=0)
    else:
        y = np.abs(args[0])
    return a_tol + y * r_tol


def _error_norm(x: np.ndarray):
    """Calculates the error norm based on eq 4.11 from Hairer.

    Parameters
    ----------
    x : np.ndarray
        Vector to compute the norm of

    Returns
    -------
    float
        Error norm of `x`
    """
    return np.linalg.norm(x) / (x.size**0.5)


def _estimate_initial_step_size(f, y0, t0, method, r_tol, a_tol, max_step):
    """Private function to estimate a suitable initial step size. Algorithm described
    on page 169 of Hairer Solving Ordinary Differential Equations 1.

    Parameters
    ----------
    f : Callable
        RHS function of the ODE
    y0 : np.ndarray
        Initial conditions
    t0 : float
        Initial Time
    method : _RungeKuttaStep
        Integrator being used
    r_tol : float
        Relative error tolerance
    a_tol : float
        Absolute error tolerance
    max_step : float
        Maximum allowable stepsize

    Returns
    -------
    float
        Estimate for the initial step size to use
    """
    # step a
    f0 = f(t0, y0)
    scale = _scale(r_tol, a_tol, f0, y0)
    d0 = _error_norm(y0 / scale)
    d1 = _error_norm(f0 / scale)

    # step b
    if d0 < 1e-5 or d1 < 1e-5 or math.isnan(d0) or math.isnan(d1):
        h0 = 1e-6
    else:
        h0 = 0.01 * (d0 / d1)

    # step c
    y1 = y0 + h0 * f(t0, y0)

    # step d
    diff = f(t0 + h0, y1) - f(t0, y0)
    scale = _scale(r_tol, a_tol, diff)
    d2 = _error_norm(diff / scale)

    # step e
    if max(d1, d2) <= 1e-15 or math.isnan(d1) or math.isnan(d2):
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** (1 / (method.order + 1))

    # step d
    h = max(100 * h0, h1)
    return h if h < max_step else max_step


def _step(f, t, y, h, method):
    r"""Computes one step of the ode ``y' = f(t,y)``.

    Based on the following equation,

    .. math::

        y_{n+1} = y_n + h \sum_{i=1}^s b_i k_i,

    where

    .. math::

        k_i = f(t_n + c_i h, y_n + h \sum_{i=1}^s a_{ij}k_j).

    Parameters
    ----------
    f : Callable
        The rhs function.
    t : float
        The current time.
    y : np.ndarray
        The current state.
    h : float
        The step size.
    method
        Butcher tableau for the method being used

    Returns
    -------
    y1 : np.ndarray
        Result object containing the next value for `y`
    err: float
        error estimate.
    """
    s = method.B.size
    k = np.empty((y.size, s))
    # calculate k values
    for i in range(s):
        # k_i = f(t + c_i*h, y + h * sum_{j=1}^{i-1} a_ij*k_j)
        k[:, i] = f(
            t + method.C[i] * h,
            y + h * np.sum(method.A[i, np.newaxis, : i + 1] * k[:, : i + 1], axis=-1),
        )

    # y_{n+1} = y_n + h sum_{i=1}^s b_i*k_i
    y1 = y + h * np.inner(method.B, k)

    return y1, h * np.inner(method.B - method.B_hat, k)


def solve_to_adaptive(f, y0, t_span, method, r_tol=1e-3, a_tol=1e-6, max_step=np.inf):
    """Function for solving ODE using an adaptive timestep.

    Parameters
    ----------
    f : Callable
        RSH of the ODE.
    y0 : np.ndarray
        Initial conditions
    t_span : tuple[float, float]
        Time span to solve over
    method : str
        Stepper to use
    r_tol : float
        The relative error tolerance
    a_tol : float
        The absolute error tolerance
    max_step : float
        Maximum allowable step size to take

    Returns
    -------
    y
        Solution to f
    t
        Time corresponding to solution y
    """

    if method in _embedded_methods.keys():
        method = _embedded_methods[method]()
    else:
        raise ValueError(f"{method} is not a valid option for 'method'")

    h = _estimate_initial_step_size(f, y0, t_span[0], method, r_tol, a_tol, max_step)

    fac_max = 1.5
    fac_min = 0.5
    safety_fac = 0.9

    t = [t_span[0]]
    y = [np.asarray(y0)]

    final_step = False

    # Check if integration is finished
    while (t[-1] - t_span[-1]) < 0:
        step_accepted = False
        while not step_accepted:
            y1, local_err = _step(f, t[-1], y[-1], h, method)

            scale = _scale(r_tol, a_tol, y1, y[-1])
            err = _error_norm(local_err / scale)

            # adjust step size
            # eq 4.13 Hairer
            if err == 0:
                h_new = h * fac_max
            else:
                h_new = h * min(
                    fac_max,
                    max(fac_min, safety_fac * (1 / err) ** (1 / (method.order + 1))),
                )
            h_new = max_step if h_new > max_step else h_new

            # accept the step
            if (err <= 1 and h <= max_step) or final_step:
                if (t[-1] + h - t_span[-1]) > 0 and not final_step:
                    final_step = True
                    h_new = t_span[-1] - t[-1]
                else:
                    step_accepted = True
                    t.append(t[-1] + h)
                    y.append(y1)
            h = h_new

    return np.asarray(y).T, np.asarray(t)
