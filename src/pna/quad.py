import numpy as np
import scipy


def composite_trapezium(f, a, b, n):
    x = np.linspace(a, b, n)
    y = f(x)
    h = (b - a) / n
    return (h / 2) * (y[0] + 2 * np.sum(y[1:-1]) + y[-1])


def clenshaw_curtis(f, a, b, n):
    # Generate Chebyshev spaced points
    x = 0.5 * ((b - a) * np.cos(np.pi * np.arange(0, n + 1) / n) + (b + a))

    # Evaluate function at Chebyshev points
    fx = f(x) / (2 * n)

    # Compute weights
    g = np.real(np.fft.fft(np.concatenate([fx, fx[-2:0:-1]])))
    w = np.concatenate([[g[0]], g[1 : n - 1] + g[-1 : n + 1 : -1], [g[-1]]])

    # Compute integral of Chebyshev basis over [-1,1]
    I = np.zeros_like(w)
    # Only even terms are non-zero
    I[::2] = 2 / (1 - np.arange(0, n, 2) ** 2)

    return 0.5 * (b - a) * I @ w


def guass_legendre(f, a, b, n):
    gamma = 0.5 / np.sqrt(1 - (2 * np.arange(1, n + 1, dtype=float)) ** (-2))
    T = np.diag(gamma, -1) + np.diag(gamma, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(T)
    w = 2 * eigenvectors[0, :] ** 2
    return 0.5 * (b - a) * w @ f(0.5 * (b - a) * eigenvalues + 0.5 * (a + b))


def _second_order_central(f, x, h=np.finfo(float).eps ** (1 / 3)):
    return (f(x + h) - 2 * f(x) + f(x - h)) / h**2


def adaptive_composite_trapezium(f, a, b, n0, tol):
    x = np.linspace(a, b, n0)

    while True:
        # note it really should be max|f''| however here we just use f''(\bar{x})
        # where f'' is computed using a centred difference
        # if the intervals are small hopefully this isn't a terrible approximation
        error_summand = (
            np.diff(x) ** 3
            * np.abs(_second_order_central(f, (x[1:] + x[:-1]) / 2))
            / 12
        )
        err = np.sum(error_summand)
        if err < tol:
            break

        # find interval with greatest contribution to error and split interval
        i = np.argmax(error_summand)
        x = np.insert(x, i + 1, 0.5 * (x[i] + x[i + 1]))

    h = np.diff(x)
    return np.sum(0.5 * h * (f(x[:-1]) + f(x[1:])))
