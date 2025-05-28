from util import condition_met, function, find_xn, stop_met, first_derivative, second_derivative

a = 0
b = 5
e = 0.01

def convergence_met(bottom, top, first_derivative, second_derivative):
    if (first_derivative(bottom) * first_derivative(top) >= 0 and
            (second_derivative(bottom) * second_derivative(top) >= 0)):
        return 1
    return 0


def find_xn1(xn, func, first_derviative):
    return xn - func(xn) / first_derviative(xn)

def find_result(a, b, e, func=function,
                first_derivative=first_derivative, second_derivative=second_derivative):
    if not condition_met(a, b, func):
        return "warunek konieczny nie jest spełniony"
    iteration = 1
    start = find_xn(a, b, second_derivative, func)
    xn1 = find_xn1(start, func, first_derivative)
    while not stop_met(start, xn1, e, func):
        start = xn1
        xn1 = find_xn1(xn1, func, first_derivative)
        iteration += 1
    return xn1, iteration


print(find_result(a, b, e, function, first_derivative, second_derivative))