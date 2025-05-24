from approxmation.utils.main import compute_integral

p_x = 1
a = -1
b = 1
n = 6


def wielomian_legendre(x, n):
    if n == 0:
        return 1
    if n == 1:
        return x
    if n == 2:
        return 1 / 2 * (3 * x ** 2 - 1)
    return ((2 * n - 1) * x * wielomian_legendre(x, n - 1) -
            (n - 1) * wielomian_legendre(x, n - 2)) / n


def calculate_lambdai(i, a, b):
    return compute_integral(a, b, 100, i, wielomian_legendre, 'lambda')


def func(x):
    return (x ** 3 - 2 * x + 10) ** (1 / 2)


def function_ci(x, i):
    return p_x * wielomian_legendre(x, i) * func(x)


def calculate_ci(i, a, b):
    return 1 / calculate_lambdai(i, a, b) * compute_integral(a, b, 100, i, function_ci)


def calculate_gx(x, a, b, n):
    result = 0
    for i in range(n + 1):
        result += calculate_ci(i, a, b) * wielomian_legendre(x, i)
    return result


print("g(0.4) = ", calculate_gx(0.4, a, b, n))
print(func(0.4) - calculate_gx(0.4, a, b, n))