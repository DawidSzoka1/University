from util import condition_met, function

a = 0
b = 5
e = 0.05

def x_middl(a, b):
    return 1/2 * (a + b)


def accuracy_meet(value, e, func=function):
    if abs(func(value)) < e:
        return 1
    return 0


def find_result(a, b, e, func=function):
    bottom = a
    x_m = x_middl(a, b)
    top = b
    iteration = 1
    while not accuracy_meet(x_m, e, func):
        if condition_met(bottom, x_m, func):
            top = x_m
        elif condition_met(x_m, top, func):
            bottom = x_m
        else:
            return "warunek konieczny nie jest spełniony"
        x_m = x_middl(bottom, top)
        iteration += 1
    return x_m, iteration


print(find_result(a, b, e, function))