import math
import random


def func_nd(*args):
    n = len(args)

    suma = sum(x ** 2 - 10 * math.cos(20 * math.pi * x) for x in args)

    return 10 * n + suma


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
    max_val = (2 ** m) - 1
    samples_count = min(n, max_val + 1)

    random_numbers = random.sample(range(max_val + 1), samples_count)
    return [bin(num) for num in random_numbers]


def get_decimals(binar):
    result = [int(x, 2) for x in binar]
    return result


def get_scale(decimals, a, b, m,d ):
    result = [round(a + ((b - a) * i) / (2 ** m - 1), d) for i in decimals]
    return result


def calc(data, n, f):
    length_a = [(x.get('b') - x.get('a')) * (10 ** x.get('d')) + 1 for x in data]
    ms = [find_m(leng) for leng in length_a]
    binarys = [get_all_binary(m, n) for m in ms]

    to_decimal = [get_decimals(all_binary) for all_binary in binarys]
    scale = []
    for dec_list, var_data, m_val in zip(to_decimal, data, ms):
        scale.append(get_scale(dec_list, var_data['a'], var_data['b'], m_val, var_data['d']))

    points = list(zip(*scale))
    print(points)
    min_point = points[0]
    fun_min = f(*min_point)
    for pt in points:
        candidate = f(*pt)
        if candidate < fun_min:
            min_point = pt
            fun_min = candidate
    return min_point


zmienne_3d = [
    {'a': -1, 'b': 1, 'd': 6}, # x1
    {'a': -1, 'b': 1, 'd': 6}, # x2
    {'a': -1, 'b': 1, 'd': 6}  # x3
]
najlepszy_punkt = calc(zmienne_3d, 1000000, func_nd)
print("Najlepszy punkt:", najlepszy_punkt)
print("Wartość funkcji:", func_nd(*najlepszy_punkt))
