from interpolation import newton

if '__main__' == __name__:
    x_points = [-4, -2, 0, 2, 4]
    y_points = [802, 78, 2, -50, -318]
    xi = 3
    wynik = newton(x_points, y_points, xi)
    print(f"Interpolacja Newtona dla x={xi}: {wynik}")