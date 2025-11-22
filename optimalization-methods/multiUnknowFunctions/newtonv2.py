from myFunction import *
import numpy as np

x0 = 2
y0 = 2


def testFunction(x, y):
    return 10 * x ** 2 + 12 * x * y + 10 * y ** 2


def calculate(x0, y0, e, function=function, pochxy=dfdxdy, pochx=dfdx, pochy=dfdy,
              poch2x=dfdxdx, poch2y=dfdydy, iterations=100):
    xk, yk = x0, y0
    for i in range(iterations):
        gradient = np.array([pochx(function, xk, yk), pochy(function, xk, yk)])
        hessego = np.array(
            [[poch2x(function, xk, yk), pochxy(function, xk, yk)],
             [pochxy(function, xk, yk), poch2y(function, xk, yk)]])
        xk2, yk2 = np.array([xk, yk]) - np.dot(reversed_matrix(hessego), gradient)
        print(
            f"iteracja {i + 1}: gradtient: \n{gradient}\nmacierz hessego: \n{hessego}\nx_{i + 1}={xk2}, y_{i + 1}={yk2}")
        gradient_k2 = np.array([pochx(function, xk2, yk2), pochy(function, xk2, yk2)])
        if np.linalg.norm(gradient_k2) <= e or (abs(xk2 - xk) <= e and abs(yk2 - yk) <= e):
            return xk2, yk2, i + 1
        xk, yk = xk2, yk2
    return xk, yk, iterations


if __name__ == "__main__":
    print(calculate(2, 2, 0.01 ))
    print(calculate(10, 12, 0.01, function=testFunction))
