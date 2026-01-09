from myFunction import *



def calculate(x0, y0, epsilon, iterations=100, function=function, pochx=functionPochx, pochy=functionPochy,
              poch2x=functionPoch2x, poch2y=functionPoch2y, pochxy=functionPochxy, check_e=True):
    xk, yk = x0, y0
    for i in range(iterations):
        def func_x(val_x): return pochx(x=val_x, y=yk)
        xk = helper_calculate(-xk - 100, xk + 100,0.00000000000001, func_x, iteration=iterations)
        def func_y(val_y): return pochy(x=xk, y=val_y)
        yk = helper_calculate(-yk - 100, yk + 100,0.00000000000001, func_y, iteration=iterations)
        print(f"x_{i+1} = {xk} and y_{i} = {yk}")
        if not check_e:
            continue
        grad = gradient(xk, yk, pochx=pochx, pochy=pochy)
        if np.linalg.norm(grad) <= epsilon:
            return xk, yk, i + 1
    return xk, yk, iterations





if __name__ == "__main__":
    # test = calculate(2, 2, 0.01, 100, check_e=False)
    test2 = calculate(10, 10, 0.07, 100, pochx=testFunctionPochx, pochy=testFunctionPochy)
    print(test2)