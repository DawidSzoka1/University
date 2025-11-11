# def poch2_funkcja(x):
#     return 2 * x + 1
#
#
# def poch1_funkcja(x):
#     return x ** 2 + x - 5
#
#
# def poch3_funkcja(x):
#     return 2


def calculate(a, b, e, function=None, maks=True, iteration=100, poch_f=None, poch2=None, poch3=None):
    x0 = a if poch3(a) * poch_f(a) > 0 else b
    xn = x0
    for i in range(iteration):
        xn1 = xn - poch_f(xn) / poch2(xn)
        print(f"Iteracja {i+1}: x{i+1} = {xn1}, f'(x{i+1}) = {poch_f(xn1)}, f\"(x{i+1}) = {poch2(xn1)}")
        if abs(poch_f(xn1)) < e:
            return xn1, function(xn1), i
        if abs(xn1 - xn) < e:
            return xn1, function(xn1), i
        xn = xn1
    return xn1, function(xn1), iteration


if __name__ == "__main__":
    from myFunction import funkcja,poch_funkcja, poch2_funkcja, poch3_funkcja, a, b
    print(calculate(a, b, 1e-7, function=funkcja,poch_f=poch_funkcja, poch2=poch2_funkcja, poch3=poch3_funkcja))
