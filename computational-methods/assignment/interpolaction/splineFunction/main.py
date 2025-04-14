from sympy import symbols
import numpy as np

x_values = [-4, -2, 0, 2, 4]
y_values = [802, 78, 2, -50, -318]
derivative = [-676, -180]

def exprW(a, x, poch=False):
    if poch:
        result = a[1]
        for i in range(2, len(a)):
            result += i * a[i] * pow(x, i - 1)
        return result
    result = a[0]
    for i in range(1, len(a)):
        result += a[i] * pow(x, i)
    return result


def alphaS(x, xs, alpha, poch=False):
    if poch:
        return 3 * alpha * pow(x - xs, 2)
    return alpha * pow(x - xs, 3)


def splineFunction(x_values, y_values, derivative, looking_for):
    n = len(x_values)
    x_symbol = symbols('x_symbol')
    a = []
    alpha = []
    for i in range(4):
        a.append(symbols('a' + str(i)))
    for i in range(n - 2):
        alpha.append(symbols('alpha' + str(i+1)))
    all_symbols = a + alpha
    expressions = []
    for i, x in enumerate(x_values):
        value = exprW(a, x)
        for j in range(1, i):
            value += alphaS(x, x_values[j], alpha[j-1])

        expressions.append(value)
    expressions.append(exprW(a, x_values[0], True))
    value = exprW(a, x_values[n-1], True)
    for i in range(0, len(alpha)):
        value += alphaS(x_values[n-1], x_values[i+1], alpha[i], True)
    expressions.append(value)
    expressions_values = []
    for expr in expressions:
        val = []
        for symbol in all_symbols:
            val.append(expr.coeff(symbol))
        expressions_values.append(val)

    A = np.array([
        value for value in expressions_values
    ], dtype=np.float64)
    B = np.array([
        [y] for y in y_values + derivative
    ], dtype=np.float64)
    result = np.linalg.solve(A, B)
    subs = dict(zip(all_symbols, result))
    subs = {k: round(v.item(), 3) for k, v in subs.items()}
    Wx = 0
    for i,sym in enumerate(a):
        Wx += sym * pow(x_symbol, i)

    equaiton = Wx.subs(subs)
    for i,sym in enumerate(alpha, 0):
        if looking_for > x_values[i+1]:
            equaiton += (sym * pow(x_symbol - x_values[i+1], 3)).subs(subs)
        else:
            break
    return equaiton.subs({x_symbol: looking_for})



print("S(3) = ",
      splineFunction(x_values, y_values, derivative, 3))
