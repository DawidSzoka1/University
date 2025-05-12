import math
from sympy import symbols, Eq, solve
from main import compute_integral
import numpy as np

a = 1
b = 3
p_x = 1
n = 2

def function(x):
    return x ** (1 / 2)


def function_bi(x, i):
    return x ** i * function(x) * p_x


def calculate_bi(i):
    return compute_integral(a, b, 100, i, function_bi)


def calculate_aij(i, j):
    return (1 / (i + j + 1)) * (b ** (i + j + 1) - a ** (i + j + 1))


a_array = np.zeros((n+1, n+1), dtype=float)
for i in range(n + 1):
    for j in range(n + 1):
        a_array[i, j] = calculate_aij(i, j)
b_array = np.zeros((n+1, 1), dtype=float)
for i in range(n+1):
    b_array[i, 0] = calculate_bi(i)
result = np.linalg.solve(a_array, b_array)
def w_x(x):
    sum = 0
    for i in range(n):
        sum += result[i] * (x ** i)
    return sum

print(w_x(1))