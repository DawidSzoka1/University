import math


def compute_ti(xi, xi2):
    return (xi2 + xi) / 2


def compute_h(xi, xi2):
    return (xi2 - xi) / 2


def compute_xi(i, a, b, n):
    return a + i / n * (b - a)


def function(x):
    return (1.4 * x + 0.3)/(2.3 + math.cos(0.4*x**2 + 1))


a = 0.4
b = 1.3
n = 3


def compute_integral(a, b, n, func=function):
    x_values = [a]
    ti_values = []
    for i in range(1, n + 1):
        x_values.append(compute_xi(i, a, b, n))
        ti_values.append(compute_ti(x_values[i - 1], x_values[i]))
    h = compute_h(x_values[-2], x_values[-1])

    y_values = []
    for xi in x_values:
        y_values.append(func(xi))
    y_ti_values = []
    for ti in ti_values:
        y_ti_values.append(func(ti))

    suma = y_values[0] + y_values[-1]
    for xi in range(1, len(y_values) - 1):
        suma += y_values[xi] * 2
    for ti in y_ti_values:
        suma += ti * 4
    return h / 3 * suma


print(f"przyblizona wartosc calki dla n = {n} wynosi {compute_integral(a, b, n)}")
