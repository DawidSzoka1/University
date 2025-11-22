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



def calculate(x0, y0, e, function=function, pochxy=functionPochxy, pochx=functionPochx, pochy=functionPochy,
              poch2x=functionPoch2x, poch2y=functionPoch2y, iterations=100):
    xk, yk = x0, y0
    for i in range(iterations):
        gradient = np.array([pochx(xk, yk), pochy(xk, yk)])
        hessego = np.array(
            [[poch2x(xk, yk), pochxy(xk, yk)], [pochxy(xk, yk), poch2y(xk, yk)]])

        xk2, yk2 = np.array([xk, yk]) - np.dot(reversed_matrix(hessego), gradient)
        print(
            f"iteracja {i + 1}: gradtient: \n{gradient}\nmacierz hessego: \n{hessego}\nx_{i + 1}={xk2}, y_{i + 1}={yk2}")
        gradient_k2 = np.array([pochx(xk2, yk2), pochy(xk2, yk2)])
        if np.linalg.norm(gradient_k2) <= e or (abs(xk2 - xk) <= e and abs(yk2 - yk) <= e):
            print()
            return xk2, yk2, i + 1
        xk, yk = xk2, yk2
    return xk, yk, iterations


if __name__ == "__main__":
    print(calculate(10, 12, 0.01, pochxy=testFunctionPochxy, pochx=testFunctionPochx, pochy=testFunctionPochy,
                    poch2x=testFunctionPochxx, poch2y=testFUnctionPochyy))

    print(calculate(2, 2, 0.01))
