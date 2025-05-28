def function(x):
    return 5 * x ** 2 +  11 * x - 79

def condition_met(bottom, top, func):
    if func(bottom) * func(top) < 0:
        return 1
    return 0
