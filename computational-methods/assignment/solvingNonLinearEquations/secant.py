from util import condition_met, stop_met, find_xn, function, second_derivative

a = 0
b = 5
e = 0.01


def find_xn2(xn, motionless, a, b, func):
    if motionless == a:
        return xn - (func(xn) / (func(xn) - func(a))) * (xn - a)
    else:
        return xn - (func(xn) / (func(b) - func(xn))) * (b - xn)


def find_result(a, b, e, func=function, second_derivative=second_derivative):
    if not condition_met(a, b, func):
        return "warunek konieczny nie jest spełniony"
    iteration = 1
    xn = find_xn(a, b, func, second_derivative)
    motionless = xn
    x0 = a if b == motionless else b
    xn1 = find_xn2(x0, motionless, a, b, func)
    while not stop_met(xn, xn1, e, func):
        xn = xn1
        xn1 = find_xn2(xn1, motionless, a, b, func)
        iteration += 1
    return xn1, iteration

print(find_result(a, b, e, function, second_derivative))