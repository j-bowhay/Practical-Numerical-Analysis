import numpy as np
import scipy


def composite_trapezium(f, a, b, n):
    x = np.linspace(a, b, n)
    y = f(x)
    h = (b - a) / n
    return (h / 2) * (y[0] + 2 * np.sum(y[1:-1]) + y[-1])


def clenshaw_curtis(f, a, b, n):
    #  Chebyshev points
    x = np.cos(np.pi * np.arange(0, n + 1) / n)
    # f evaluated at these points
    fx = f(0.5 * (b - a) * x + 0.5 * (a + b)) / (2 * n)
    g = np.real(np.fft.fft(np.concatenate([fx, fx[-2:0:-1]])))
    # Chebyshev coefficients
    a = np.concatenate([[g[0]], g[1 : n - 1] + g[-1 : n + 1 : -1], [g[-1]]])
    # weight vector
    w = np.zeros_like(a)
    w[::2] = 2 / (1 - np.arange(0, n, 2) ** 2)
    return 0.5 * (b - a) * w @ a


def guass_legendre(f, a, b, n):
    beta = 0.5 / np.sqrt(1 - (2 * np.arange(1, n + 1, dtype=float)) ** (-2))
    T = scipy.sparse.diags([beta, beta], [1, -1]).toarray()
    w, v = np.linalg.eig(T)
    i = np.argsort(w)
    return 2 * v[0, i] ** 2 @ f(w[i])


print(clenshaw_curtis(lambda x: x**2, -1, 1, 10))
print(guass_legendre(lambda x: x**2, 0, 1, 10))