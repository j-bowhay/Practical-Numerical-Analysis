import numpy as np
import matplotlib.pyplot as plt


def generate_chebyshev_nodes(a, b, n):
    """Generates n Chebyshev nodes over the interval [a, b]"""
    i = np.arange(n)

    return (0.5 * (a + b) + 0.5 * (b - a) * np.cos((i * np.pi) / (n - 1)))[::-1]


def monomial_interpolant(f, nodes):
    """Construct a polynomial interpolant of the function f using monomial basis
    functions by sampling from f at the points given by nodes.

    Returns the coefficients of the polynomial and the condition number of the
    Vandermonde matrix.
    """
    f_nodes = f(nodes)
    V = np.vander(nodes, increasing=True)

    # compute coefficients
    a = np.linalg.solve(V, f_nodes)

    return a, np.linalg.cond(V)


def honer_evaluate(a, x):
    """Evaluates the polynomial given by the coefficients a at points x using
    the Horner scheme"""
    b = a[-1]
    for a_n in a[-2::-1]:
        b = a_n + b * x
    return b


def barycentric_interpolate(f_i, x_i, x, chebyshev=False):
    """Constructs and evaluates a Lagrange interpolating polynomial."""
    if chebyshev:
        # For Chebyshev points we have an analytical formula for the weights
        weights = np.ones_like(x_i)
        weights[0] /= 2
        weights[-1] /= 2
        weights *= (-1) ** np.arange(len(x_i))
    else:
        # compute weights w_k=prod(x_k - x_j)
        dist = x_i - x_i[..., np.newaxis]
        np.fill_diagonal(dist, 1)
        weights = 1 / np.prod(dist, axis=1)

    # compute all the x - x_k terms
    c = x - x_i[..., np.newaxis]

    # handle the case where a point coincides with a node
    exact = c == 0
    c[np.nonzero(exact)] = 1

    summand = weights[..., np.newaxis] / c

    # Apply the second barycentric interpolation formula
    p = (f_i @ summand) / np.sum(summand, axis=0)

    # replace any points with nodal values if needed
    p[np.any(exact, axis=0)] = f_i[np.any(exact, axis=1)]
    return p


def linear_spline_interpolate(f_i, x_i, x):
    """Compute and evaluate a linear spline interpolant"""
    if np.any(x < np.amin(x_i)) or np.any(x > np.amax(x_i)):
        raise ValueError("Cannot extrapolate")

    i = np.searchsorted(x_i, x)

    return ((x_i[i] - x) / (x_i[i] - x_i[i - 1])) * f_i[i - 1] + (
        (x - x_i[i - 1]) / (x_i[i] - x_i[i - 1])
    ) * f_i[i]


def main():

    ...

if __name__ == "__main__":
    main()
