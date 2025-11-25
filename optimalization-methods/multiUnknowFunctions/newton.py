from myFunction import *
import numpy as np

x0 = 2
y0 = 2


def calculate(x0, y0, e, function=function, pochxy=functionPochxy, pochx=functionPochx, pochy=functionPochy,
              poch2x=functionPoch2x, poch2y=functionPoch2y, iterations=100):
    xk, yk = x0, y0
    for i in range(iterations):
        grad = gradient(xk, yk, pochx, pochy)
        hasse = hassego(xk, yk, pochxx=poch2x, pochyy=poch2y, pochxy=pochxy)

        xk2, yk2 = np.array([xk, yk]) - np.dot(reversed_matrix(hasse), grad)
        print(
            f"iteracja {i + 1}: gradtient: \n{grad}\nmacierz hessego: \n{hasse}\nx_{i + 1}={xk2}, y_{i + 1}={yk2}")
        grad_k2 = gradient(xk2, yk2, pochx, pochy)
        if np.linalg.norm(grad_k2) <= e or (abs(xk2 - xk) <= e and abs(yk2 - yk) <= e):
            print()
            return xk2, yk2, i + 1
        xk, yk = xk2, yk2
    return xk, yk, iterations


if __name__ == "__main__":
    print(calculate(10, 12, 0.01, pochxy=testFunctionPochxy, pochx=testFunctionPochx, pochy=testFunctionPochy,
                    poch2x=testFunctionPochxx, poch2y=testFUnctionPochyy))

    print(calculate(2, 2, 0.01))
