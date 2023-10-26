import math


def bisect(f, a, b, tol):
    count = 0
    f_c = math.inf
    while abs(f_c) > tol:
        count += 1
        c = 0.5 * (a + b)
        f_c = f(c)
        if f_c == 0:
            break
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c
    return c, count


def regula_falsi(f, a, b, tol):
    count = 0
    f_c = math.inf
    while abs(f_c) > tol:
        f_a = f(a)
        f_b = f(b)
        c = (a * f_b - b * f_a) / (f_b - f_a)
        f_c = f(c)
        count += 1
        if f_c == 0:
            break
        elif f_a * f_c < 0:
            b = c
        else:
            a = c
    return c, count


def illinois(f, a, b, tol):
    count = 0
    right = True
    count = 0
    f_c = math.inf
    while abs(f_c) > tol:
        f_a = f(a)
        f_b = f(b)
        if count > 1:
            if right:
                f_b /= 2
            else:
                f_a /= 2
        c = (a * f_b - b * f_a) / (f_b - f_a)
        f_c = f(c)
        count += 1
        if f_c == 0:
            break
        elif f_a * f(c) < 0:
            b = c
            if right:
                right_count = 0
            else:
                count += 1
            right = False
        else:
            a = c
            if not right:
                count = 0
            else:
                count += 1
            right = True
    return c, count


def newton_raphson(f, f_prime, x0, tol):
    k = 0
    x = x0
    while abs(f(x)) > tol:
        x -= f(x) / f_prime(x)
        k += 1
    return x, k
