from myFunction import *



def calculate(x0, y0, epsilon, iterations=100, function=function, pochx=dfdx, pochy=dfdy,
              poch2x=dfdxdx, poch2y=dfdydy, pochxy=dfdxdy, check_e=True):
    xk, yk = x0, y0
    for i in range(iterations):
        def func_x(val_x): return pochx(function, x=val_x, y=yk)
        xk = helper_calculate(-xk - 100, xk + 100,0.00000000000001, func_x, iteration=iterations)
        def func_y(val_y): return pochy(function, x=xk, y=val_y)
        yk = helper_calculate(-yk - 100, yk + 100,0.00000000000001, func_y, iteration=iterations)
        print(f"x_{i+1} = {xk} and y_{i} = {yk}")
        if not check_e:
            continue
        grad = gradientF(xk, yk, fun=function)
        if np.linalg.norm(grad) <= epsilon:
            return xk, yk, i + 1
    return xk, yk, iterations





if __name__ == "__main__":
    # test = calculate(2, 2, 0.01, 100, check_e=False)
    test2 = calculate(10, 10, 0.07, 100, function=example_function)
    print(test2)