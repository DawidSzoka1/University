from math import sqrt

import requests

from myFunction import *



def calculate(x0, y0, epsilon, iterations=100, function=function, pochx=functionPochx, pochy=functionPochy,
              poch2x=functionPoch2x, poch2y=functionPoch2y, pochxy=functionPochxy, check_e=True):
    xk, yk = x0, y0
    gs_epsilon = epsilon / 100
    for i in range(iterations):
        xk_0, yk_0 = xk, yk
        def func_x(val_x): return pochx(x=val_x, y=yk)
        xk = helper_calculate(-xk, xk, gs_epsilon, func_x, iteration=100000)
        def func_y(val_y): return pochy(x=xk, y=val_y)
        yk = helper_calculate(-yk, yk, gs_epsilon, func_y, iteration=100000)
        print(f"Iteracja {i+1}: x = {xk:.6f}, y = {yk:.6f}")
        if check_e:
            grad = gradient(xk, yk)
            if np.linalg.norm(grad) <= epsilon:
                return xk, yk, i + 1, function(xk, yk)
    return xk, yk, iterations, function(xk, yk)





if __name__ == "__main__":
    test = calculate(2, 2, 0.01, 100, check_e=True)
    # test2 = calculate(10, 10, 0.07, 100, pochx=testFunctionPochx, pochy=testFunctionPochy)
    print(test)