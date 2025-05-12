import math
def function(x):
    return (1.4 * x + 0.3)/(2.3 + math.cos(0.4*x**2 + 1))


def compute_h(n, a, b):
    return (b - a) / n


def compute_xi(i, a, b, n):
    return a + i / n * (b - a)


def compute_integral(a, b, n, i_value=None, func=None):
    h = compute_h(n, a, b)
    x_values = [a]
    for i in range(1, n):
        x_values.append(compute_xi(i, a, b, n))
    x_values.append(b)
    y_values = []
    for xi in x_values:
        y_values.append(func(xi, i_value))
    sum_first_last = y_values[0] / 2 + y_values[-1] / 2
    sum_full = sum_first_last
    for i in range(1, len(x_values) - 1):
        sum_full += y_values[i]

    return h * sum_full
