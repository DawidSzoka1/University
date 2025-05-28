def function(x):
    return 5 * x ** 2 +  11 * x - 79

def first_derivative(x):
    return 10 * x + 11

def second_derivative(x):
    return 10


def condition_met(bottom, top, func):
    if func(bottom) * func(top) < 0:
        return 1
    return 0

def find_xn(bottom, top, second_derivative, func):
    if second_derivative(top) * func(top) > 0:
        return top
    return bottom

def stop_met(xn, xn1, e, func):
    if abs(func(xn1)) < e:
        return 1
    if abs(xn1 - xn) < e:
        return 1
    return 0
