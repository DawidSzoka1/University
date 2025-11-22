import numpy as np


def function(x, y):
    return 3 * x ** 3 - x * y + y ** 2 - 2 * y + 1


def reversed_matrix(matrix):
    return 1 / (matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]) * np.array(
        [[matrix[1, 1], -matrix[0, 1]], [-matrix[1, 0], matrix[0, 0]]])


def functionPochx(x, y):
    return 9 * x ** 2 - y


def functionPochy(x, y):
    return -x + 2 * y - 2


def functionPoch2x(x, y):
    return 18 * x


def functionPoch2y(x, y):
    return 2


def functionPochxy(x, y):
    return -1


h = 0.00001
round_spot = 4


def dfdx(f, x, y):
    return round((f(x + h, y) - f(x, y)) / h, round_spot)


def dfdy(f, x, y):
    return round((f(x, y + h) - f(x, y)) / h, round_spot)


def dfdxdx(f, x, y):
    return round((f(x + 2 * h, y) - 2 * f(x + h, y) + f(x, y)) / (h ** 2), round_spot)


def dfdydy(f, x, y):
    return round((f(x, y + 2 * h) - 2 * f(x, y + h) + f(x, y)) / (h ** 2), round_spot)


def dfdxdy(f, x, y):
    return round((f(x + h, y + h) - f(x + h, y) - f(x, y + h) + f(x, y)) / (h ** 2), round_spot)


if __name__ == '__main__':
    def f_test(x, y):
        return x ** 2 * y ** 2


    x0, y0 = 2, 1

    print("TEST POCHODNYCH DLA f = x^2 y^2 W PUNKCIE (2,1):")
    print("df/dx =", dfdx(f_test, x0, y0), "  (oczekiwane 4)")
    print("df/dy =", dfdy(f_test, x0, y0), "  (oczekiwane 8)")
    print("d2f/dx2 =", dfdxdx(f_test, x0, y0), "  (oczekiwane 2)")
    print("d2f/dy2 =", dfdydy(f_test, x0, y0), "  (oczekiwane 8)")
    print("d2f/dxdy =", dfdxdy(f_test, x0, y0), "  (oczekiwane 8)")
    print()
