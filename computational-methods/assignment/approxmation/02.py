import numpy as np

x_values = [-1, -0.5, 0, 0.5, 1]
n = 10
m = 4


def calculate_aij(x_values, i, j):
    result = 0
    for x in x_values:
        result += x ** (i + j)
    return result


def calculate_bk(x_values, y_values, k):
    result = 0
    for i in range(len(x_values)):
        result += (x_values[i] ** k) * y_values[i]
    return result


def function(x):
    return (x ** 3 - 2 * x + 10) ** (1 / 2)


def calculate_wx(look_for_value, x_values, n, m, func=function):
    y_values = [func(x) for x in x_values]
    b_array = np.zeros((m + 1, 1))
    for i in range(m + 1):
        b_array[i][0] = calculate_bk(x_values, y_values, i)

    a_array = np.zeros((m + 1, m + 1))
    for i in range(m + 1):
        for j in range(m + 1):
            a_array[i, j] = calculate_aij(x_values, i, j)
    values = np.linalg.solve(a_array, b_array)
    result = 0
    for i in range(m):
        result += values[i][0] * look_for_value ** i
    return result


print(calculate_wx(0.4, x_values, n, m))
