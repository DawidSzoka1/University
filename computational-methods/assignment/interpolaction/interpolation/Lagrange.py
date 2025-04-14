def lagrange_interpolation(x_values, y_values, x):
    n = len(x_values)
    result = 0
    for i in range(n):
        temp = y_values[i]
        for j in range(n):
            if j != i:
                temp *= (x - x_values[j]) / (x_values[i] - x_values[j])
        result += temp

    return result
