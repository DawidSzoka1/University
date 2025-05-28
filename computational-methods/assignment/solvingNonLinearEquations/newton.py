from util import condition_met, function

a = 0
b = 5
e = 0.01


def first_derivative(x):
    return 10 * x + 11

def second_derivative(x):
    return 10

def stop_met(xn, xn1, e, func):
    if abs(func(xn1)) < e:
        return 1
    if abs(xn1 - xn) < e:
        return 1
    return 0

def convergence_met(bottom, top, first_derivative, second_derivative):
    if (first_derivative(bottom) * first_derivative(top) >= 0 and
            (second_derivative(bottom) * second_derivative(top) >= 0)):
        return 1
    return 0

def find_xn(bottom, top, second_derivative, func):
    if second_derivative(bottom) * func(bottom) > 0:
        return bottom
    return top

def find_xn1(xn, func, first_derviative):
    return xn - func(xn) / first_derviative(xn)

def find_result(a, b, e, func=function,
                first_derivative=first_derivative, second_derivative=second_derivative):
    if not condition_met(a, b, func):
        return "warunek konieczny nie jest spełniony"
    start = find_xn(a, b, second_derivative, func)
    xn1 = find_xn1(start, func, first_derivative)
    while not stop_met(start, xn1, e, func):
        start = xn1
        xn1 = find_xn1(xn1, func, first_derivative)
    return xn1


print(find_result(a, b, e, function, first_derivative, second_derivative))