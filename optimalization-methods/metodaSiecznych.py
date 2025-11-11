# def poch_fun(x):
#     return pow(x, 2) + x - 5
#
#
# def poch3_fun(x):
#     return 2


def find_montionless(a, b, poch_f, poch3_f):
    poch1_a = poch_f(a)
    poch3_a = poch3_f(a)
    if poch1_a * poch3_a >= 0:
        return b, 'A'
    poch1_b = poch_f(b)
    poch3_b = poch3_f(b)
    if poch1_b * poch3_b >= 0:
        return a, 'B'
    return None

def calculate(a, b, e, function=None, maks=True, iteration=100, poch_f=None, poch2=None, poch3=None):
    x0, method = find_montionless(a, b, poch_f, poch3)
    xn = x0
    for i in range(iteration):
        if method == 'A':
            xn1 = xn - (poch_f(xn) / (poch_f(xn) - poch_f(a))) * (xn - a)
        else:
            xn1 = xn - (poch_f(xn) / (poch_f(b) - poch_f(xn))) * (b - xn)
        print(f"Iteracja {i+1}: x{i+1} = {xn1}, f'(x{i+1}) = {poch_f(xn1)}")
        if abs(poch_f(xn1)) < e:
            return xn1, function(xn1), i

        if abs(xn1 - xn) < e:
            return xn1, function(xn1), i
        xn = xn1
    return xn1, function(xn1), iteration


if __name__ == "__main__":
    from myFunction import funkcja,poch_funkcja, poch2_funkcja, poch3_funkcja, a, b
    print(calculate(a, b, 1e-7, function=funkcja,poch_f=poch_funkcja, poch2=poch2_funkcja, poch3=poch3_funkcja))