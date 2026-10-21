import math
import random


def func(x):
    return 10 + x ** 2 - 10 * math.cos(20 * math.pi * x)


def tournament_without_replacement(population, k, f):
    pop_shuffled = population[:]
    random.shuffle(pop_shuffled)

    groups = [[] for _ in range(k)]

    for i, ind in enumerate(pop_shuffled):
        groups[i % k].append(ind)

    winners = []
    for group in groups:
        best = min(group, key=lambda x: f(x))
        winners.append(best)

    return winners


def tournament_with_replacement(population, k, f, tour_size=2):
    winners = []

    for _ in range(k):
        fighters = random.sample(population, tour_size)

        best = min(fighters, key=lambda x: f(x))
        winners.append(best)

    return winners


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

    random_numbers = set()
    while len(random_numbers) < samples_count:
        random_numbers.add(random.getrandbits(m))

    return [bin(num) for num in random_numbers]


def get_decimals(binar):
    result = [int(x, 2) for x in binar]
    return result


def get_scale(decimals, a, b, m, d):
    result = [round(a + ((b - a) * i) / (2 ** m - 1), d) for i in decimals]
    return result


def calc(a, b, d, n, k, f, tournament=tournament_without_replacement):
    population = (b - a) * (10 ** d) + 1
    m = find_m(population)
    all_binary = get_all_binary(m, n)
    to_decimal = get_decimals(all_binary)
    population = get_scale(to_decimal, a, b, m, d)

    new_population=tournament(population, k, f)
    while len(new_population) > 1:
        k -= 1
        new_population = tournament(new_population, k, f)

    return new_population


wynik = calc(-1, 1, 4, 100, 10, func)
print("Wyselekcjonowana populacja:", wynik)
print("Wartosc funkcji: ", func(wynik[0]))