from myFunction import funkcja


def value_k():
    return (5 ** (1 / 2) - 1) / 2


def calculate(a, b, e, function=funkcja, maks=True, iteration=100, poch_f=None, poch2=None, poch3=None):
    k = value_k()
    x1 = b - k * (b - a)
    x2 = a + k * (b - a)
    for i in range(iteration):
        f_x1 = function(x1)
        f_x2 = function(x2)
        if maks:
            if f_x1 > f_x2:
                b = x2
                x2 = x1
                x1 = b - k * (b - a)
            else:
                a = x1
                x1 = x2
                x2 = a + k * (b - a)
        else:
            if f_x1 < f_x2:
                b = x2
                x2 = x1
                x1 = b - k * (b - a)
            else:
                a = x1
                x1 = x2
                x2 = a + k * (b - a)
        if abs(x2 - x1) < e:
            return (a + b) / 2, function((a + b) / 2), i
    return (a + b) / 2, function((a + b) / 2), iteration


if __name__ == "__main__":
    a, b = 0.6, 5.8
    print(calculate(a, b, 0.0000001, maks=False, iteration=100))
