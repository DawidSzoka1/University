from Lagrange import lagrange_interpolation


if __name__ == '__main__':
    x_values = [-4, -2, 0, 2, 4]
    y_values = [802, 78, 2, -50, -318]  
    interpolated_value = lagrange_interpolation(x_values, y_values, 3)
    print(interpolated_value)

# powinno wyjsc x = 3, W(x) = -157