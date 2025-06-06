from approxmation.utils.main import compute_integral
import numpy as np

a = -1
b = 1
p_x = 1
n = 6
x = 0.4


def function(x):
    return (x ** 3 - 2 * x + 10) ** (1 / 2)


def function_bi(x, i):
    return x ** i * function(x) * p_x


def calculate_bi(i, a, b, func):
    return compute_integral(a, b, 100, i, func)


def calculate_aij(i, j, a, b):
    return (1 / (i + j + 1)) * (b ** (i + j + 1) - a ** (i + j + 1))


def calculate_wx(x, n, a, b, func=function_bi):
    a_array = np.zeros((n + 1, n + 1), dtype=float)
    for i in range(n + 1):
        for j in range(n + 1):
            a_array[i, j] = calculate_aij(i, j, a, b)
    b_array = np.zeros((n + 1, 1), dtype=float)
    for i in range(n + 1):
        b_array[i, 0] = calculate_bi(i, a, b, func)
    result = np.linalg.solve(a_array, b_array)
    sum = 0
    for i in range(n + 1):
        sum += result[i] * (x ** i)
    return sum[0]


print(f"W({x}) = ", calculate_wx(x, n, a, b, function_bi))
print(function(0.4) - calculate_wx(x, n, a, b, function_bi))
