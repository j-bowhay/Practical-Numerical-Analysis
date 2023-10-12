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
    # Question 1

    f = lambda x: np.sin(x) + np.sin(x**2)

    interval = [0, 4]
    n = np.logspace(2, 6, num=5, base=2, dtype=int)

    x = np.linspace(*interval, num=1000)
    f_true = f(x)

    monomial_regular_nodes_err = []
    monomial_regular_nodes_condition_number = []
    monomial_chebyshev_nodes_err = []
    monomial_chebyshev_nodes_condition_number = []
    lagrange_regular_nodes_err = []
    lagrange_chebyshev_nodes_err = []

    for n_i in n:
        # Create regularly space nodes
        regular_nodes = np.linspace(*interval, num=n_i)
        chebyshev_nodes = generate_chebyshev_nodes(*interval, n_i)

        # Monomial basis, regular grid
        a, nu = monomial_interpolant(f, regular_nodes)
        err = np.linalg.norm(honer_evaluate(a, x) - f_true, np.inf)

        monomial_regular_nodes_err.append(err)
        monomial_regular_nodes_condition_number.append(nu)

        # Monomial basis, chebyshev grid
        a, nu = monomial_interpolant(f, chebyshev_nodes)
        err = np.linalg.norm(honer_evaluate(a, x) - f_true, np.inf)

        monomial_chebyshev_nodes_err.append(err)
        monomial_chebyshev_nodes_condition_number.append(nu)

        # Lagrange Polynomial, regular grid
        err = np.linalg.norm(
            barycentric_interpolate(f(regular_nodes), regular_nodes, x) - f_true, np.inf
        )

        lagrange_regular_nodes_err.append(err)

        # Lagrange Polynomial, chebyshev grid
        err = np.linalg.norm(
            barycentric_interpolate(
                f(chebyshev_nodes), chebyshev_nodes, x, chebyshev=True
            )
            - f_true,
            np.inf,
        )

        lagrange_chebyshev_nodes_err.append(err)

    # Plot error against number of nodes
    fig, ax = plt.subplots()
    ax.plot(
        n,
        monomial_regular_nodes_err,
        "x-",
        label="Monomial Basis, Regularly Spaced Nodes",
    )
    ax.plot(
        n,
        monomial_chebyshev_nodes_err,
        "x-",
        label="Monomial Basis, Chebyshev Spaced Nodes",
    )
    ax.plot(
        n,
        lagrange_regular_nodes_err,
        "x-",
        label="Lagrange Polynomial, Regular Spaced Nodes",
    )
    ax.plot(
        n,
        lagrange_chebyshev_nodes_err,
        "x-",
        label="Lagrange Polynomial, Chebyshev Spaced Nodes",
    )
    ax.set_yscale("log")
    ax.legend()
    ax.set_xlabel("n")
    ax.set_ylabel("Maximum Error")
    ax.set_title("Q1: Error Plot")
    plt.show()

    # The monomial basis function with both regular and chebyshev points converges as
    # n increases. For the Lagrange polynomial as n increases the Runge phenomenon and
    # error diverges. The Lagrange polynomial with Chebyshev points converges and at
    # a faster rate than the monomial interplant.

    # Plot condition number of the Vandermonde matrix for the monomial interplant
    # against the number of nodes
    fig, ax = plt.subplots()
    ax.plot(
        n,
        monomial_regular_nodes_condition_number,
        "x-",
        label="Monomial Basis, Regularly Spaced Nodes",
    )
    ax.plot(
        n,
        monomial_chebyshev_nodes_condition_number,
        "x-",
        label="Monomial Basis, Chebyshev Spaced Nodes",
    )
    ax.set_yscale("log")
    ax.legend()
    ax.set_xlabel("n")
    ax.set_ylabel("Condition number")
    ax.set_title("Q1: Condition Number Plot")
    plt.show()

    # We see that the condition number gets very large!!

    # Question 2
    n = np.logspace(2, 7, num=6, base=2, dtype=int)

    linear_spline_err = []

    for n_i in n:
        xi = np.linspace(*interval, num=n_i)
        f_i = f(xi)

        err = np.linalg.norm(linear_spline_interpolate(f_i, xi, x) - f_true, np.inf)
        linear_spline_err.append(err)

    fig, ax = plt.subplots()
    ax.plot(n, linear_spline_err, label="Linear Spline")
    ax.set_yscale("log")
    ax.legend()
    ax.set_xlabel("n")
    ax.set_ylabel("Maximum Error")
    ax.set_title("Q2: Error Plot")
    plt.show()

    # As expected the error decreases with more points

    # Question 3

    g = lambda x: np.abs(x - 1 / 3)
    g_true = g(x)

    linear_spline_err_g = []

    for n_i in n:
        xi = np.linspace(*interval, num=n_i)
        g_i = g(xi)

        err = np.linalg.norm(linear_spline_interpolate(g_i, xi, x) - g_true, np.inf)
        linear_spline_err_g.append(err)

    fig, ax = plt.subplots()
    ax.plot(n, linear_spline_err_g, label="Linear Spline")
    ax.set_yscale("log")
    ax.legend()
    ax.set_xlabel("n")
    ax.set_ylabel("Maximum Error")
    ax.set_title("Q3: Error Plot")
    plt.show()

    # Here the error stops decreases very slowly as the interpolating function struggles
    # capture the sharp corner in g. There would need to be a node at x=1/3 to approximate
    # this function better.


if __name__ == "__main__":
    main()
