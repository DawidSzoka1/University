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


h = 0.000001
round_spot = 10


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


def example_function(x, y):
    return 10 * x ** 2 + 12 * x * y + 10 * y ** 2


def testFunctionPochxy(x, y):
    return 12


def testFunctionPochxx(x, y):
    return 20


def testFUnctionPochyy(x, y):
    return 20


def testFunctionPochx(x, y):
    return 20 * x + 12 * y


def testFunctionPochy(x, y):
    return 12 * x + 20 * y


def gradient(x, y, pochx=functionPochx, pochy=functionPochy):
    return np.array([pochx(x, y), pochy(x, y)])


def gradientF(x, y, fun=function):
    return np.array([dfdx(fun, x, y), dfdy(fun, x, y)])


def hassego(x, y, func=None, pochxx=functionPoch2x, pochyy=functionPoch2y, pochxy=functionPochxy):
    if func:
        return np.array(
            [[pochxx(func, x, y), pochxy(func, x, y)],
             [pochxy(func, x, y), pochyy(func, x, y)]])
    return np.array(
        [[pochxx(x, y), pochxy(x, y)],
         [pochxy(x, y), pochyy(x, y)]])


def calculate_alpha(grad, hess):
    a = grad[0]
    b = grad[1]
    c = hess[0][0]
    d = hess[0][1]
    e = hess[1][0]
    f = hess[1][1]
    return (a ** 2 + b ** 2) / ((a * c + b * e) * a + (a * d + b * f) * b)


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
