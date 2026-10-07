import math
import random

def func(x):
    return 10 + x**2 - 10 * math.cos(20 * math.pi * x)


def find_m(population):
    result = 2
    m = 1
    while result < population:
        if result >= population:
            break
        result = result * 2
        m += 1
    return m


def get_all_binary(m, n):
    result = []
    non_bin = []
    count = 0
    while count <= n:
        num = random.randint(0, 2**m)
        if num not in non_bin:
            non_bin.append(num)
            result.append(bin(num))
            count += 1
        if count > 2**m:
            break
    return result


def get_decimals(binar):
    result = [int(x, 2) for x in binar]
    return result


def get_scale(decimals, a, b, m,d ):
    result = [round(a + ((b - a) * i) / (2 ** m - 1), d) for i in decimals]
    return result


def calc(a, b, d, n, f):
    population = (b - a) * (10 ** d) + 1
    m = find_m(population)
    all_binary = get_all_binary(m, n)
    to_decimal = get_decimals(all_binary)
    scale = get_scale(to_decimal, a, b, m,d )
    min_x = b
    for x in scale:
        if f(x) < f(min_x):
            min_x = x
    return min_x


print(calc(-1, 1, 4, 100, func))
print(calc(-1, 1, 3, 10, func))
