import time
import random
import threading
from multiprocessing import Manager, Process
import matplotlib.pyplot as plt


# ----------------------------
# Parametry
# ----------------------------
THINK = (0.01, 0.03)
EAT = (0.01, 0.03)

TESTS = 5
PHILOSOPHER_COUNTS = [5, 10, 15, 20, 30]


# ----------------------------
# Statystyki
# ----------------------------

def create_stats(n):
    return {
        "waiting": [0.0] * n,
        "eating": [0.0] * n,
        "thinking": [0.0] * n,
    }


# ----------------------------
# Implementacje
# ----------------------------

def sequential(n):
    stats = create_stats(n)
    start = time.time()

    for i in range(n):
        for _ in range(3):

            t0 = time.time()
            time.sleep(random.uniform(*THINK))
            stats["thinking"][i] += time.time() - t0

            # brak czekania w wersji sekwencyjnej

            t0 = time.time()
            time.sleep(random.uniform(*EAT))
            stats["eating"][i] += time.time() - t0

    return time.time() - start, stats


def threading_test(n):
    locks = [threading.Lock() for _ in range(n)]
    stats = create_stats(n)
    threads = []

    start = time.time()

    def philosopher(i):
        left = locks[i]
        right = locks[(i + 1) % n]
        first = left if i % 2 == 0 else right
        second = right if i % 2 == 0 else left

        for _ in range(3):
            t0 = time.time()
            time.sleep(random.uniform(*THINK))
            stats["thinking"][i] += time.time() - t0

            t0 = time.time()
            with first:
                with second:
                    stats["waiting"][i] += time.time() - t0

                    t0e = time.time()
                    time.sleep(random.uniform(*EAT))
                    stats["eating"][i] += time.time() - t0e

    for i in range(n):
        t = threading.Thread(target=philosopher, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    return time.time() - start, stats


def multiprocessing_test(n):

    manager = Manager()
    locks = [manager.Lock() for _ in range(n)]

    waiting = manager.list([0.0] * n)
    eating = manager.list([0.0] * n)
    thinking = manager.list([0.0] * n)
    flag = manager.Value('b', True)

    def philosopher(i, locks, flag, waiting, eating, thinking):
        left = locks[i]
        right = locks[(i + 1) % n]
        first = left if i % 2 == 0 else right
        second = right if i % 2 == 0 else left

        # THINK
        t0 = time.time()
        time.sleep(random.uniform(*THINK))
        thinking[i] += time.time() - t0

        # WAIT + EAT
        t0 = time.time()
        with first:
            with second:
                waiting[i] += time.time() - t0

                t0e = time.time()
                time.sleep(random.uniform(*EAT))
                eating[i] += time.time() - t0e

        flag.value = False

    procs = []
    start = time.time()

    for i in range(n):
        p = Process(target=philosopher,
                    args=(i, locks, flag, waiting, eating, thinking))
        procs.append(p)
        p.start()

    for p in procs:
        p.join()

    stats = {
        "waiting": list(waiting),
        "eating": list(eating),
        "thinking": list(thinking),
    }

    return time.time() - start, stats


# ----------------------------
# Benchmark
# ----------------------------

seq_times = []
thread_times = []
mp_times = []

seq_stats_all = []
thread_stats_all = []
mp_stats_all = []


for n in PHILOSOPHER_COUNTS:
    seq_sum = 0
    thread_sum = 0
    mp_sum = 0

    seq_local_stats = create_stats(n)
    thread_local_stats = create_stats(n)
    mp_local_stats = create_stats(n)

    for _ in range(TESTS):

        t, stats = sequential(n)
        seq_sum += t
        for k in stats:
            for i in range(n):
                seq_local_stats[k][i] += stats[k][i]

        t, stats = threading_test(n)
        thread_sum += t
        for k in stats:
            for i in range(n):
                thread_local_stats[k][i] += stats[k][i]

        t, stats = multiprocessing_test(n)
        mp_sum += t
        for k in stats:
            for i in range(n):
                mp_local_stats[k][i] += stats[k][i]

    # średnie
    for k in seq_local_stats:
        seq_local_stats[k] = [x / TESTS for x in seq_local_stats[k]]
        thread_local_stats[k] = [x / TESTS for x in thread_local_stats[k]]
        mp_local_stats[k] = [x / TESTS for x in mp_local_stats[k]]

    seq_times.append(seq_sum / TESTS)
    thread_times.append(thread_sum / TESTS)
    mp_times.append(mp_sum / TESTS)

    seq_stats_all.append(seq_local_stats)
    thread_stats_all.append(thread_local_stats)
    mp_stats_all.append(mp_local_stats)

    print(f"--- N = {n} ---")
    print("Sequential:      ", seq_sum / TESTS)
    print("Threading:       ", thread_sum / TESTS)
    print("Multiprocessing: ", mp_sum / TESTS)
    print()


# ----------------------------
# Wykres czasu
# ----------------------------

plt.figure(figsize=(10, 6))
plt.plot(PHILOSOPHER_COUNTS, seq_times, label="Sekwencyjna")
plt.plot(PHILOSOPHER_COUNTS, thread_times, label="Threading")
plt.plot(PHILOSOPHER_COUNTS, mp_times, label="Multiprocessing")

plt.xlabel("Liczba filozofów")
plt.ylabel("Średni czas (s)")
plt.title("Porównanie wydajności — 3 implementacje")
plt.legend()
plt.grid(True)
plt.savefig("testfive-philosophers.png")
plt.show()


# ----------------------------
# Wykres czasu czekania
# ----------------------------

def avg_wait(stats):
    return [sum(s["waiting"]) / len(s["waiting"]) for s in stats]


plt.figure(figsize=(10, 6))
plt.plot(PHILOSOPHER_COUNTS, avg_wait(seq_stats_all), marker="o", label="SEQ — czekanie")
plt.plot(PHILOSOPHER_COUNTS, avg_wait(thread_stats_all), marker="o", label="THREAD — czekanie")
plt.plot(PHILOSOPHER_COUNTS, avg_wait(mp_stats_all), marker="o", label="MP — czekanie")
plt.xlabel("Liczba filozofów")
plt.ylabel("Średni czas oczekiwania (s)")
plt.title("Średni czas czekania na jedzenie")
plt.legend()
plt.grid(True)
plt.savefig("waiting-times.png")
plt.show()


# ----------------------------
# Wykres czasu jedzenia
# ----------------------------

def avg_eat(stats):
    return [sum(s["eating"]) / len(s["eating"]) for s in stats]

plt.figure(figsize=(10, 6))
plt.plot(PHILOSOPHER_COUNTS, avg_eat(seq_stats_all), marker="o", label="SEQ — jedzenie")
plt.plot(PHILOSOPHER_COUNTS, avg_eat(thread_stats_all), marker="o", label="THREAD — jedzenie")
plt.plot(PHILOSOPHER_COUNTS, avg_eat(mp_stats_all), marker="o", label="MP — jedzenie")
plt.xlabel("Liczba filozofów")
plt.ylabel("Średni czas jedzenia (s)")
plt.title("Średni czas przebywania w sekcji krytycznej")
plt.legend()
plt.grid(True)
plt.savefig("eating-times.png")
plt.show()
