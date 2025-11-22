from myFunction import *
import numpy as np

x0 = 2
y0 = 2


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


def reversed_matrix(matrix):
    return np.array([[matrix[1, 1], -matrix[0, 1]], [-matrix[1, 0], matrix[0, 0]]])


def calculate(x0, y0, e, function=function, pochxy=functionPochxy, pochx=functionPochx, pochy=functionPochy,
              poch2x=functionPoch2x, poch2y=functionPoch2y, iterations=100):
    xk, yk = x0, y0
    for i in range(iterations):
        gradient = np.array([pochx(xk, yk), pochy(xk, yk)])
        hessego = np.array(
            [[poch2x(xk, yk), pochxy(xk, yk)], [pochxy(xk, yk), poch2y(xk, yk)]])

        xk2, yk2 = np.array([xk, yk]) - 1 / (hessego[0, 0] * hessego[1, 1] - hessego[0, 1] * hessego[1, 0]) * np.dot(
            reversed_matrix(hessego), gradient)
        if abs(xk2 - xk) <= e and abs(yk2 - yk) <= e:
            return xk2, yk2
        xk, yk = xk2, yk2
    return xk, yk

print(calculate(10, 12, 10, pochxy=testFunctionPochxy, pochx=testFunctionPochx, pochy=testFunctionPochy,
          poch2x=testFunctionPochxx, poch2y=testFUnctionPochyy))
