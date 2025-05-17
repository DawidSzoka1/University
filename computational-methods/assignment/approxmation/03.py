import numpy as np

x_values = [-1, -0.5, 0, 0.5, 1]
n = 3
m = 5


def calculate_sij(x_values, i, j, m):
    result = 0
    for s in range(m):
        result += x_values[s] ** (i + j)
    return result


def calculate_tk(x_values, y_values, k, m):
    result = 0
    for i in range(m):
        result += (x_values[i] ** k) * y_values[i]
    return result


def function(x):
    return (x ** 3 - 2 * x + 10) ** (1 / 2)


def calculate_wx(look_for_value, x_values, n, m, func=function):
    y_values = [func(x) for x in x_values]
    b_array = np.zeros((n + 1, 1))
    for i in range(n + 1):
        b_array[i][0] = calculate_tk(x_values, y_values, i, m)

    a_array = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        for j in range(n + 1):
            a_array[i, j] = calculate_sij(x_values, i, j, m)
    values, *_ = np.linalg.lstsq(a_array, b_array, rcond=None)
    result = 0
    for i in range(n+1):
        result += values[i][0] * look_for_value ** i
    return result


print(calculate_wx(0.4, x_values, n, m))
