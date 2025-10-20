import matplotlib.pyplot as plt
import numpy as np
from myFunction import funkcja

a = 0.6
b = 5.8
e = 0.0000000001


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


def calculate(a, b, e, function=funkcja, maks=True, num_iterations=100):
    xsr = (a + b) / 2
    for i in range(num_iterations):
        L = b - a
        x1 = a + L / 4
        x2 = b - L / 4
        x1_y = function(x1)
        xsr_y = function(xsr)
        x2_y = function(x2)
        print(
            f"Iteracja {i + 1}, x1 = {x1}, x2 = {x2}, xsr = {xsr}, L = {L}, a = {a}, b = {b}, f(x1) = {x1_y}, f(x2) = {x2_y}, f(xsr) = {xsr_y}")
        if i < 5:
            plt.axvline(a, color=colors[i], linestyle='--')
            plt.axvline(b, color=colors[i], linestyle='--')
        if maks:
            a, b, xsr = maksimum(x1_y, xsr_y, x2_y, xsr, x1, x2, a, b)
        else:
            a, b, xsr = minimum(x1_y, xsr_y, x2_y, xsr, x1, x2, a, b)
        if L <= e:
            return xsr
    return xsr


plt.figure(figsize=(6, 6))
xsr = calculate(a, b, e, maks=False, num_iterations=6)
x = np.linspace(a, b, 1000)
y = funkcja(x)
plt.plot(x, y)
plt.scatter(xsr, funkcja(xsr), color='black', label='minimum')
plt.legend()
plt.grid(True)
plt.show()
print(xsr)
