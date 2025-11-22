import matplotlib.pyplot as plt
import numpy as np
from myFunction import funkcja, przykladowa

a = 0.6
b = 5.8
e = 0.01

def maksimum(x1_y, xsr_y, x2_y, xsr, x1, x2, a, b):
    if x1_y > xsr_y:
        b = xsr
        xsr = x1
    else:
        if x2_y > xsr_y:
            a = xsr
            xsr = x2
        else:
            a = x1
            b = x2
    return a, b, xsr


def minimum(x1_y, xsr_y, x2_y, xsr, x1, x2, a, b):
    if x1_y < xsr_y:
        b = xsr
        xsr = x1
    else:
        if x2_y < xsr_y:
            a = xsr
            xsr = x2
        else:
            a = x1
            b = x2
    return a, b, xsr


colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']


def calculate(a, b, e, function=funkcja, maks=True, iteration=100, poch_f=None, poch2=None, poch3=None):
    xsr = (a + b) / 2
    L = b - a
    xsr_y = function(xsr)
    print(f"Iteracja {1}, xsr = {xsr}, L = {L}, a = {a}, b = {b}, f(xsr) = {xsr_y}")
    for i in range(iteration):
        L = b -a
        x1 = a + L / 4
        x2 = b - L / 4
        x1_y = function(x1)
        xsr_y = function(xsr)
        x2_y = function(x2)
        if L <= e:
            return xsr, function(xsr), i
        if i < 5:
            plt.axvline(a, color=colors[i], linestyle='--')
            plt.axvline(b, color=colors[i], linestyle='--')
        if maks:
            a, b, xsr = maksimum(x1_y, xsr_y, x2_y, xsr, x1, x2, a, b)
        else:
            a, b, xsr = minimum(x1_y, xsr_y, x2_y, xsr, x1, x2, a, b)
        L = b - a
        print(f"Iteracja {i + 2}, x1 = {x1}, x2 = {x2}, xsr = {xsr}, L = {L}, a = {a}, b = {b}, f(x1) = {x1_y}, f(x2) = {x2_y}, f(xsr) = {xsr_y}")
    return xsr, function(xsr), iteration


if __name__ == "__main__":
    plt.figure(figsize=(6, 6))
    xsr, fx, it = calculate(a, b, 0.00001, maks=False, iteration=1000)
    x = np.linspace(a, b, 1000)
    y = funkcja(x)
    plt.plot(x, y)
    plt.scatter(xsr, funkcja(xsr), color='black', label='minimum')
    plt.legend()
    plt.grid(True)
    plt.show()
    print(f"minimum f(x) jest w punkcie {xsr}")

