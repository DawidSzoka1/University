a = 0
b = 5
e = 0.05
def function(x):
    return 5 * x ** 2 +  11 * x - 79


def x_middl(a, b):
    return 1/2 * (a + b)

def condition_met(bottom, top, func=function):
    if func(bottom) * func(top) < 0:
        return 1
    return 0


def accuracy_meet(value, e, func=function):
    if abs(func(value)) < e:
        return 1
    return 0


def find_result(a, b, e, func=function):
    bottom = a
    x_m = x_middl(a, b)
    top = b
    for i in range(5):
        if condition_met(bottom, x_m, func):
            top = x_m
        elif condition_met(x_m, top, func):
            bottom = x_m
        else:
            return "warunek konieczny nie jest spełniony"
        if accuracy_meet(x_m, e, func):
            return x_m
        x_m = x_middl(bottom, top)
    return f"Po 5 iteracjach: {x_m}"


print(find_result(a, b, e, function))