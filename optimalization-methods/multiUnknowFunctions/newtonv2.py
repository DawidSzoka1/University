from myFunction import *
import numpy as np

x0 = 2
y0 = 2


def calculate(x0, y0, e, function=function, pochxy=dfdxdy, pochx=dfdx, pochy=dfdy,
              poch2x=dfdxdx, poch2y=dfdydy, iterations=100):
    xk, yk = x0, y0
    for i in range(iterations):
        grad = gradientF(xk, yk, function)
        hess = hassego(xk, yk, func=function, pochxx=poch2x, pochyy=poch2y, pochxy=pochxy)
        xk2, yk2 = np.array([xk, yk]) - np.dot(reversed_matrix(hess), grad)
        print(
            f"iteracja {i + 1}: gradtient: \n{grad}\nmacierz hessego: \n{hess}\nx_{i + 1}={xk2}, y_{i + 1}={yk2}")
        gradient_k2 = gradientF(xk2, yk2, function)
        if np.linalg.norm(gradient_k2) <= e or (abs(xk2 - xk) <= e and abs(yk2 - yk) <= e):
            return xk2, yk2, i + 1
        xk, yk = xk2, yk2
    return xk, yk, iterations


if __name__ == "__main__":
    print(calculate(10, 12, 0.01, function=example_function))
    print(calculate(2, 2, 0.01))
