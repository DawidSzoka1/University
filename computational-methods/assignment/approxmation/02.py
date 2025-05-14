import math

from utils.main import compute_integral

p_x = 1
a = -1
b = 1


def wielomian_legendre(x, n):
    if n == 0:
        return 1
    if n == 1:
        return x
    if n == 2:
        return 1 / 2 * (3 * x ** 2 - 1)
    return 1 / (n + x) * (2 * n + 1) * x * wielomian_legendre(n - 1, x) - n / (n + 1) * wielomian_legendre(n - 2, x)


def calculate_lambdai(i, a, b):
    return compute_integral(a, b, 100, i, wielomian_legendre, 'lambda')


def func(x):
    return math.exp(x)


def functio_ci(x, i):
    return p_x * wielomian_legendre(x, i) * func(x)


def calculate_ci(i, a, b):
    return 1/calculate_lambdai(i, a, b) * compute_integral(a, b, 100, i, functio_ci)

print(calculate_ci(2, a, b))