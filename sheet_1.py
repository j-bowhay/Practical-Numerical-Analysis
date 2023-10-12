import numpy as np
import matplotlib.pyplot as plt


def generate_chebyshev_nodes(a, b, n):
    """Generates n Chebyshev nodes over the interval [a, b]"""
    i = np.arange(n)

    return 0.5 * (a + b) + 0.5 * (b - a) * np.cos((i * np.pi) / (n - 1))


def monomial_interplant(f, nodes):
    """Construct a polynomial interplant of the function f using monomial basis
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


def barycentric_interpolate(f, nodes, x, weights=None):
    if weights is None:
        # compute weights w_k=prod(x_k - x_j)
        dist = nodes - nodes[..., np.newaxis]
        np.fill_diagonal(dist, 1)
        weights = 1 / np.prod(dist, axis=1)

    # compute all the x - x_k terms
    c = x - nodes[..., np.newaxis]
    exact = np.nonzero(c == 0)
    c[exact] = 1

    summand = weights[..., np.newaxis] / c

    p = np.dot(f(nodes), summand) / np.sum(summand, axis=-1)[..., np.newaxis]
    return p


def barycentric_chebyshev_weights(n):
    k = np.arange(n)
    w = (-1) ** k * 2 ** (n - 1) / n
    w[0] /= 2
    w[-1] /= 2
    return w


def main():
    # Question 1

    f = lambda x: np.sin(x) + np.sin(x**2)

    interval = [0, 4]
    n = np.logspace(2, 6, num=5, base=2, dtype=int)

    # Plot comparison
    x = np.linspace(*interval)
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
        a, nu = monomial_interplant(f, regular_nodes)
        err = np.linalg.norm(honer_evaluate(a, x) - f_true, np.inf)

        monomial_regular_nodes_err.append(err)
        monomial_regular_nodes_condition_number.append(nu)

        # Monomial basis, chebyshev grid
        a, nu = monomial_interplant(f, chebyshev_nodes)
        err = np.linalg.norm(honer_evaluate(a, x) - f_true, np.inf)

        monomial_chebyshev_nodes_err.append(err)
        monomial_chebyshev_nodes_condition_number.append(nu)

        # Lagrange Polynomial, regular grid
        err = np.linalg.norm(
            barycentric_interpolate(f, regular_nodes, x) - f_true, np.inf
        )

        lagrange_regular_nodes_err.append(err)

        # Lagrange Polynomial, chebyshev grid
        weights = barycentric_chebyshev_weights(n_i)
        err = np.linalg.norm(
            barycentric_interpolate(f, regular_nodes, x, weights=weights) - f_true,
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
    plt.show()

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
    plt.show()

    # We see that the condition number gets very large!!


if __name__ == "__main__":
    main()
