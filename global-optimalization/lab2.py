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
    samples_count = min(n, 2 ** m)

    random_numbers = set()
    while len(random_numbers) < samples_count:
        random_numbers.add(random.getrandbits(m))
    return [format(num, f'0{m}b') for num in random_numbers]


def split_chromosome(chromosome, ms):
    result = []
    start = 0
    for m in ms:
        result.append(chromosome[start:start + m])
        start += m
    return result


def get_decimals(binar):
    result = [int(x, 2) for x in binar]
    return result


def get_scale(decimals, data, ms):
    result = [round(var_data['a'] + ((var_data['b'] - var_data['a']) * i) / (2 ** m_val - 1), var_data['d'])
              for i, var_data, m_val in zip(decimals, data, ms)]
    return tuple(result)


def calc(data, n, f):
    length_a = [(x.get('b') - x.get('a')) * (10 ** x.get('d')) + 1 for x in data]
    ms = [find_m(leng) for leng in length_a]
    chromosomes = get_all_binary(sum(ms), n)

    points = []
    print(chromosomes[0])
    print(split_chromosome(chromosomes[0], ms))

    for chromosome in chromosomes:
        parts = split_chromosome(chromosome, ms)
        to_decimal = get_decimals(parts)
        points.append(get_scale(to_decimal, data, ms))

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
