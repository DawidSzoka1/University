from myFunction import *


def calculate(x0, y0, epsilon, iterations=100, function=function, pochx=functionPochx, pochy=functionPochy,
              poch2x=functionPoch2x, poch2y=functionPoch2y, pochxy=functionPochxy):
    xk, yk = x0, y0
    for i in range(iterations):
        grad = gradient(xk, yk, pochx=pochx, pochy=pochy)
        hass = hassego(xk, yk, pochxx=poch2x, pochyy=poch2y, pochxy=pochxy)
        alpha_k = calculate_alpha(grad, hass)
        xk2, yk2 = np.array([xk, yk]) - alpha_k * grad
        grad_next = gradient(xk2, yk2, pochx=pochx, pochy=pochy)
        if np.linalg.norm(grad_next) <= epsilon or (abs(xk2 - xk) <= epsilon and abs(yk2 - yk) <= epsilon):
            return xk2, yk2, i + 1
        xk, yk = xk2, yk2
    return xk, yk, iterations


if __name__ == "__main__":
    print(calculate(2, 2, 0.01, 100))
    print(calculate(10, 12, 0.01, 100, pochxy=testFunctionPochxy, pochx=testFunctionPochx, pochy=testFunctionPochy,
                    poch2x=testFunctionPochxx, poch2y=testFUnctionPochyy))
