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
# Implementacje
# ----------------------------

def sequential(n):
    states = [0] * n
    start = time.time()
    for i in range(n):
        for _ in range(3):
            time.sleep(random.uniform(*THINK))
            time.sleep(random.uniform(*EAT))
    return time.time() - start


def threading_test(n):
    locks = [threading.Lock() for _ in range(n)]
    states = [0] * n
    threads = []
    start = time.time()

    def philosopher(i):
        left = locks[i]
        right = locks[(i+1) % n]
        first = left if i % 2 == 0 else right
        second = right if i % 2 == 0 else left
        for _ in range(3):
            time.sleep(random.uniform(*THINK))
            with first:
                with second:
                    time.sleep(random.uniform(*EAT))

    for i in range(n):
        t = threading.Thread(target=philosopher, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    return time.time() - start


def multiprocessing_test(n):

    def philosopher(i, locks, flag):
        left = locks[i]
        right = locks[(i+1) % n]
        first = left if i % 2 == 0 else right
        second = right if i % 2 == 0 else left
        while flag.value:
            time.sleep(random.uniform(*THINK))
            with first:
                with second:
                    time.sleep(random.uniform(*EAT))
            flag.value = False  # 1 iteracja

    manager = Manager()
    locks = [manager.Lock() for _ in range(n)]
    flag = manager.Value('b', True)
    procs = []

    start = time.time()

    for i in range(n):
        p = Process(target=philosopher, args=(i, locks, flag))
        procs.append(p)
        p.start()

    for p in procs:
        p.join()

    return time.time() - start


# ----------------------------
# Benchmark
# ----------------------------

seq_times = []
thread_times = []
mp_times = []

for n in PHILOSOPHER_COUNTS:
    seq_sum = 0
    thread_sum = 0
    mp_sum = 0

    for _ in range(TESTS):
        seq_sum += sequential(n)
        thread_sum += threading_test(n)
        mp_sum += multiprocessing_test(n)

    seq_times.append(seq_sum / TESTS)
    thread_times.append(thread_sum / TESTS)
    mp_times.append(mp_sum / TESTS)

    print(f"--- N = {n} ---")
    print("Sequential:      ", seq_sum / TESTS)
    print("Threading:       ", thread_sum / TESTS)
    print("Multiprocessing: ", mp_sum / TESTS)
    print()


# ----------------------------
# Wykres porównawczy
# ----------------------------

plt.figure(figsize=(10,6))
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

seq_thread = []
seq_mp = []
mp_thread = []

for i in range(len(seq_times)):
    seq_thread.append(seq_times[i] / thread_times[i])
    seq_mp.append(seq_times[i] / mp_times[i])
    mp_thread.append(mp_times[i] / thread_times[i])

plt.figure(figsize=(10,6))
plt.plot(PHILOSOPHER_COUNTS, seq_thread, marker="o", label="Sekwencyjna / Wątki  (x razy wolniej)")
plt.plot(PHILOSOPHER_COUNTS, seq_mp, marker="o", label="Sekwencyjna / Multiprocessing  (x razy wolniej)")
plt.plot(PHILOSOPHER_COUNTS, mp_thread, marker="o", label="Multiprocessing / Wątki")
plt.xlabel("Liczba filozofów")
plt.ylabel("Przyspieszenie (x)")
plt.title("Porównanie przyspieszeń między implementacjami")
plt.legend()
plt.grid(True, alpha=0.4)
plt.savefig("five-philosophers-speedup.png")
plt.show()