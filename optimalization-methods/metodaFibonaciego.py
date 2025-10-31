from myFunction import funkcja, przykladowa

calculated_value = {'0': 1, '1': 1}


def fibonacci(n):
    if calculated_value.get(f"{n}"):
        return calculated_value[f"{n}"]
    if n < 0:
        return
    fn, fnext = 1, 1
    for i in range(n - 1):
        fnext, fn = fn + fnext, fnext
        calculated_value[f'{i + 2}'] = fnext

    return fnext


def find_max_n(a, b, e):
    n = 1
    value = (b - a) / fibonacci(n)
    while value >= 2 * e:
        n += 1
        value = (b - a) / fibonacci(n)
    return n - 1


def calculate_x1(a, b, fn, fn1):
    return b - fn / fn1 * (b - a)


def calculate_x2(a, b, fn, fn1):
    return a + fn / fn1 * (b - a)


def maksimum(x1_y, x2_y, x1, x2, a, b, fib_prev, fib_cur):
    if x1_y > x2_y:
        b = x2
        x2 = x1
        x1 = calculate_x1(a, b, fib_prev, fib_cur)
    else:
        a = x1
        x1 = x2
        x2 = calculate_x2(a, b, fib_prev, fib_cur)
    return a, b, x1, x2


def minimum(x1_y, x2_y, x1, x2, a, b, fib_prev, fib_cur):
    if x1_y < x2_y:
        b = x2
        x2 = x1
        x1 = calculate_x1(a, b, fib_prev, fib_cur)
    else:
        a = x1
        x1 = x2
        x2 = calculate_x2(a, b, fib_prev, fib_cur)
    return a, b, x1, x2


def calculate(a, b, e, function=funkcja, maks=False):
    n = find_max_n(a, b, e)
    print(f"n = {n}")
    x1 = calculate_x1(a, b, calculated_value.get(f'{n - 1}'), calculated_value.get(f'{n}'))
    x2 = calculate_x2(a, b, calculated_value.get(f'{n - 1}'), calculated_value.get(f'{n}'))
    iteration = 1
    while abs(x2 - x1) >= e and n > 1:
        x1_y = function(x1)
        x2_y = function(x2)
        print(f"Iteracja {iteration}: f(x1) = {x1_y}, f(x2) = {x2_y}, x1 = {x1}, x2= {x2}, a = {a}, b = {b}, n = {n}")
        n -= 1
        iteration += 1
        fib_prev = calculated_value.get(f'{n - 1}')
        fib_cur = calculated_value.get(f'{n}')
        if not fib_prev:
            fib_prev = fibonacci(n - 1)
            calculated_value[f'{n - 1}'] = fib_prev
        if not fib_cur:
            fib_cur = fibonacci(n)
            calculated_value[f'{n}'] = fib_cur
        if maks:
            a, b, x1, x2 = maksimum(x1_y, x2_y, x1, x2, a, b, fib_prev, fib_cur)
        else:
            a, b, x1, x2 = minimum(x1_y, x2_y, x1, x2, a, b, fib_prev, fib_cur)
    print(f"Iteracja {iteration}: f(x1) = {x1_y}, f(x2) = {x2_y}, x1 = {x1}, x2= {x2}, a = {a}, b = {b}, n = {n}")
    return (a + b) / 2


a = 0.6
b = 5.8
e = 0.01


moj_przyklad = calculate(a, b, e, maks=False)
print(f"Minimum funkcji mojej jest w punkcie: {moj_przyklad}")

test = calculate(60, 150, 3, przykladowa, maks=False)

print(f"Minimum funkcji przykladowe jest w punkcie: {test}")