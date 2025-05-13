import numpy as np

x_values = [1, 2, 3, 4]
y_values = [6, 19, 40, 69]
n = 4
m = 2


def calculate_aij(x_values, i, j):
    result = 0
    for x in x_values:
        result += x ** (i + j)
    return result



b_array = np.zeros((m + 1, 1))
for i in range(m + 1):
    result = []
    for j in range(n):
        result.append(x_values[j] ** i * y_values[j])
    sum = 0
    for x in result:
        sum += x
    b_array[i][0] = sum

a_array = np.zeros((m + 1, m + 1))
for i in range(m + 1):
    for j in range(m + 1):
        a_array[i, j] = calculate_aij(x_values, i, j)
result = np.linalg.solve(a_array, b_array)

def calculate_wx(x, n, values):
    result = 0
    for i in range(n):
        result += values[i] * x ** i
    return result

print(calculate_wx(2.5, 3, result))