def fun(x):
    return 1/3 * pow(x, 3) + 1/ 2 * pow(x, 2) - 5 * x  + 2

def dif_funa(x):
    return pow(x, 2) + x - 5


def calculate(a, b, e, function=None, maks=True, iteration=100, poch_f=None, poch2=None, poch3=None):
    xsr = 1 / 2 * (a + b)
    for i in range(iteration):
        if poch_f(a) * poch_f(b) >= 0:
            return "WARUNEK NIE ZOSTAl SPELNIONY"
        if poch_f(xsr) == 0:
            return xsr, function(xsr), i
        elif abs(poch_f(xsr)) < e:
            return xsr, function(xsr), i
        if poch_f(xsr) * poch_f(a) < 0:
            b = xsr
        else:
            a = xsr
        print(f"Iteracja {i+1}:  xsr = {xsr}, f'(xsr) = {poch_f(xsr)}")
        print(f"Nowy przedzial: [{a};{b}]")
        xsr = 1 / 2 * (a + b)
    return xsr, function(xsr), i

if __name__ == "__main__":
    print(calculate(1, 2, dif_funa, 0.05))
