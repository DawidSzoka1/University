def fun(x, y, k):
    if k == 0:
        return y[0]
    return (fun(x[1:], y[1:], k - 1) - fun(x[:-1], y[:-1], k - 1)) / (x[k] - x[0])


def newton(x, y, xi):
    result = y[0]
    product_term = 1.0
    for k in range(1, len(x)):
        product_term *= (xi - x[k - 1])
        result += fun(x[:k + 1], y[:k + 1], k) * product_term
    return result
