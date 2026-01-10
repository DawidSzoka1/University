from math import sqrt

from myFunction import *

def trial_stage(x_start, y_start, e, function):
    x, y = x_start, y_start
    f_0 = function(x, y)

    # KROK DLA X
    x_new = x + e
    f_plus = function(x_new, y)
    if f_plus < f_0:
        x = x_new
        f_0 = f_plus
    else:
        x_new = x - e
        f_minus = function(x_new, y)
        if f_minus < f_0:
            x = x_new
            f_0 = f_minus

    # KROK DLA Y
    y_new = y + e
    f_plus = function(x, y_new)
    if f_plus < f_0:
        y = y_new
        f_0 = f_plus
    else:
        y_new = y - e
        f_minus = function(x, y_new)
        if f_minus < f_0:
            y = y_new
            f_0 = f_minus

    return x, y, f_0



def calculate(x0, y0, epsilon, iterations=100, function=function, pochx=functionPochx, pochy=functionPochy,
              poch2x=functionPoch2x, poch2y=functionPoch2y, pochxy=functionPochxy, check_e=True, e_start=0.5, beta=0.5):
    xb, yb = x0, y0
    xb_0, yb_0 = x0, y0
    x_0, y_0 = x0, y0
    e = e_start
    for i in range(1, iterations + 1):
        x, y, f = trial_stage(x_0, y_0, e, function)
        fb = function(xb, yb)
        if fb > f:
            xb_0, yb_0 = xb, yb
            xb, yb = x, y
            x_0, y_0 = 2 * xb - xb_0, 2 * yb - yb_0
            print(f"Etap {i}: Roboczy -> x: {xb:.4f}, y: {yb:.4f}, f: {f:.5f}")
        else:
            if check_e:
                if e <= epsilon:
                    print(f"Osiągnięto dokładność epsilon przy e={e}")
                    return xb, yb, i, function(xb, yb)
            e = e * beta
            print(f"Etap {i}: Zmniejszenie kroku e do: {e}")
            x_0, y_0 = xb, yb

    return xb, yb, iterations, function(xb, yb)


if __name__ == "__main__":
    def example(x, y):
        return 2.5 * (x ** 2 - y) ** 2 + (1 - x) ** 2
    print(calculate(10, 12, 0.01, 100, e_start=0.5, beta=0.5, function=example_function))
    print(calculate(-0.5, 1, 0.01, 100, e_start=0.5, beta=0.5, function=example))
    print(calculate(2, 2, 0.01, 10000, e_start=0.5, beta=0.5))
