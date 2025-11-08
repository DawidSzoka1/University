from multiprocessing import Queue, Process
import random
import time
import math
import matplotlib.pyplot as plt


def monte_carlo_sequential(n):
    points_in = 0
    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x ** 2 + y ** 2 <= 1:
            points_in += 1
    return 4 * points_in / n


def monte_carlo(n, x_range, y_range, queue):
    local_in = 0
    for _ in range(n):
        x = random.uniform(*x_range)
        y = random.uniform(*y_range)
        if x ** 2 + y ** 2 <= 1:
            local_in += 1
    queue.put(local_in)


def monte_carlo_parallel(n_total, sectors):
    queue = Queue()
    processes = []

    if sectors == 1:
        ranges = [((-1, 1), (-1, 1))]
    elif sectors == 2:
        ranges = [
            ((-1, 0), (-1, 1)),
            ((0, 1), (-1, 1))
        ]
    elif sectors == 4:
        ranges = []
        for i in range(2):
            for j in range(2):
                x_range = (-1 + i, -1 + (i + 1))
                y_range = (-1 + j, -1 + (j + 1))
                ranges.append((x_range, y_range))
    elif sectors == 8:
        ranges = []
        for i in range(4):
            for j in range(2):
                x_range = (-1 + i * 0.5, -1 + (i + 1) * 0.5)
                y_range = (-1 + j, -1 + (j + 1))
                ranges.append((x_range, y_range))
    elif sectors == 16:
        # 4x4
        ranges = []
        for i in range(4):
            for j in range(4):
                x_range = (-1 + i * 0.5, -1 + (i + 1) * 0.5)
                y_range = (-1 + j * 0.5, -1 + (j + 1) * 0.5)
                ranges.append((x_range, y_range))
    elif  sectors == 32:
        ranges = []
        for i in range(8):
            for j in range(4):
                x_range = (-1 + i * 0.25, -1 + (i + 1) * 0.25)
                y_range = (-1 + j * 0.5, -1 + (j + 1) * 0.5)
                ranges.append((x_range, y_range))
        ranges = set(ranges)
    else:
        raise ValueError("Obsługiwane sektory: 1, 2, 4, 8, 16")
    n_per_sector = n_total // sectors

    # start procesów
    for x_range, y_range in ranges:
        p = Process(target=monte_carlo, args=(n_per_sector, x_range, y_range, queue))
        p.start()
        processes.append(p)

    # odbiór wyników
    total_in = 0
    for _ in range(sectors):
        total_in += queue.get()

    for p in processes:
        p.join()

    return 4 * total_in / n_total


# --- Eksperyment ---
if __name__ == "__main__":
    random.seed(42)

    sectors_list = [1, 2, 4, 16, 32]
    n_points_list = [int(1e6 * i) for i in range(1, 11)]

    results_time = {s: [] for s in sectors_list}
    results_pi = {s: [] for s in sectors_list}

    for n_total in n_points_list:
        print(f"\nLiczba punktów: {n_total:,}")
        # --- wersja sekwencyjna ---
        start = time.time()
        pi_seq = monte_carlo_sequential(n_total)
        seq_time = time.time() - start
        results_time[1].append(seq_time)
        results_pi[1].append(pi_seq)
        print(f"Sekwencyjna → π={pi_seq:.6f}, t={seq_time:.3f}s")

        # --- wersje równoległe ---
        for sectors in sectors_list:
            if sectors == 1:
                continue
            start = time.time()
            pi_par = monte_carlo_parallel(n_total, sectors)
            par_time = time.time() - start
            results_time[sectors].append(par_time)
            results_pi[sectors].append(pi_par)
            print(f"{sectors} sektory → π={pi_par:.6f}, t={par_time:.3f}s")

    # --- Rysowanie wykresów ---
    plt.figure(figsize=(10, 5))
    for s in sectors_list:
        plt.plot(n_points_list, results_time[s], label=f"{s} sektory", marker='o')
    plt.xlabel("Liczba punktów")
    plt.ylabel("Czas [s]")
    plt.title("Czas obliczeń Monte Carlo dla różnych sektorów")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    for s in sectors_list:
        errors = [abs(pi - math.pi) for pi in results_pi[s]]
        plt.plot(n_points_list, errors, label=f"{s} sektory", marker='o')
    plt.xlabel("Liczba punktów")
    plt.ylabel("|π_est - π|")
    plt.title("Błąd przybliżenia liczby π dla różnych sektorów")
    plt.legend()
    plt.grid(True)
    plt.show()
