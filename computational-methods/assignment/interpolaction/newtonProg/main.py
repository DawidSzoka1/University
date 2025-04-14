import math

def compute_forward_differences(y_values):
    n = len(y_values)
    differences = [y_values.copy()]
    for k in range(1, n):
        prev = differences[-1]
        current = [prev[i+1] - prev[i] for i in range(len(prev) - 1)]
        differences.append(current)
    return differences

def newton_interpolation(x_values, y_values, x):
    h = x_values[1] - x_values[0]
    differences = compute_forward_differences(y_values)

    n = len(x_values)
    result = y_values[0]
    product = 1

    for k in range(1, n):
        product *= (x - x_values[k-1])
        coefficient = differences[k][0] / (math.factorial(k) * (h ** k))
        result += coefficient * product

    return result


x_nodes = [-4, -2, 0, 2, 4]
y_nodes = [802, 78, 2, -50, -318]
point = 3

output = newton_interpolation(x_nodes, y_nodes, point)
print(f"W({point}) = {output}")
