from multiprocessing import Queue, Process
import random
import time
import math
import matplotlib.pyplot as plt
import numpy as np
import csv
import os


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
        ranges = [((-1, 0), (-1, 1)), ((0, 1), (-1, 1))]
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
        ranges = []
        for i in range(4):
            for j in range(4):
                x_range = (-1 + i * 0.5, -1 + (i + 1) * 0.5)
                y_range = (-1 + j * 0.5, -1 + (j + 1) * 0.5)
                ranges.append((x_range, y_range))
    elif sectors == 32:
        ranges = []
        for i in range(8):
            for j in range(4):
                x_range = (-1 + i * 0.25, -1 + (i + 1) * 0.25)
                y_range = (-1 + j * 0.5, -1 + (j + 1) * 0.5)
                ranges.append((x_range, y_range))
        ranges = set(ranges)
    else:
        raise ValueError("Obsługiwane rdzenie: 1, 2, 4, 8, 16, 32")

    n_per_sector = n_total // sectors

    for x_range, y_range in ranges:
        p = Process(target=monte_carlo, args=(n_per_sector, x_range, y_range, queue))
        p.start()
        processes.append(p)

    total_in = 0
    for _ in range(sectors):
        total_in += queue.get()

    for p in processes:
        p.join()

    return 4 * total_in / n_total


# --- Eksperyment ---
if __name__ == "__main__":
    random.seed(42)

    cores_list = [1, 2, 4, 8, 16, 32]
    n_points_list = [int(1e6 * i) for i in range(1, 11)]
    n_repeats = 5

    results_time = {s: [] for s in cores_list}
    results_pi = {s: [] for s in cores_list}
    results_std_time = {s: [] for s in cores_list}
    results_error_pi = {s: [] for s in cores_list}

    # CSV output file
    OUTPUT_FILE = "montecarlo_results.csv"
    SOURCE_TAG = "local"

    write_header = not os.path.exists(OUTPUT_FILE)
    with open(OUTPUT_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "source", "points", "cores", "mean_pi", "std_pi", "mean_time", "std_time"
            ])

        for n_total in n_points_list:
            print(f"\nLiczba punktów: {n_total:,}")

            for cores in cores_list:
                times = []
                pis = []
                for _ in range(n_repeats):
                    start = time.time()
                    if cores == 1:
                        pi_est = monte_carlo_sequential(n_total)
                    else:
                        pi_est = monte_carlo_parallel(n_total, cores)
                    times.append(time.time() - start)
                    pis.append(pi_est)

                mean_t = np.mean(times)
                std_t = np.std(times)
                mean_pi = np.mean(pis, dtype=np.float64)
                error_pi = abs((math.pi - mean_pi) / math.pi) * 100

                results_time[cores].append(mean_t)
                results_pi[cores].append(mean_pi)
                results_std_time[cores].append(std_t)
                results_error_pi[cores].append(error_pi)

                writer.writerow([SOURCE_TAG, n_total, cores, mean_pi, error_pi, mean_t, std_t])

                print(f"{cores} cores → π={mean_pi:.6f} ± {error_pi:.6f}, "
                      f"t={mean_t:.3f}s ± {std_t:.3f}s")

    # --- Wykresy lokalne (opcjonalnie) ---
    plt.figure(figsize=(10, 5))
    for s in cores_list:
        plt.errorbar(n_points_list, results_time[s],
                     yerr=results_std_time[s],
                     label=f"{s} rdzenie", marker='o', capsize=3)
    plt.xlabel("Liczba punktów")
    plt.ylabel("Czas [s]")
    plt.title(f"Średni czas Monte Carlo ({SOURCE_TAG})")
    plt.legend()
    plt.grid(True)
    plt.savefig("czas_lokalny.png", dpi=300)
    plt.show()

    plt.figure(figsize=(10, 5))
    seq_times = np.array(results_time[1])
    for s in cores_list:
        plt.plot(n_points_list, results_error_pi[s], label=f"{s} rdzenie", marker='o')
    plt.xlabel("Liczba punktów")
    plt.ylabel("Błąd procentowy")
    plt.title(f"Błąd ({SOURCE_TAG})")
    plt.legend()
    plt.grid(True)
    plt.savefig("blad_lokalny.png", dpi=300)
    plt.show()

    plt.figure(figsize=(10, 5))
    seq_times = np.array(results_time[1])
    for s in cores_list:
        speedup = seq_times / np.array(results_time[s])
        plt.plot(n_points_list, speedup, label=f"{s} rdzenie", marker='o')
    plt.xlabel("Liczba punktów")
    plt.ylabel("Przyspieszenie (Speedup)")
    plt.title(f"Speedup ({SOURCE_TAG})")
    plt.legend()
    plt.grid(True)
    plt.savefig("speedup_lokalny.png", dpi=300)
    plt.show()

    plt.figure(figsize=(10, 5))
    for s in cores_list:
        speedup = seq_times / np.array(results_time[s])
        efficiency = speedup / s
        plt.plot(n_points_list, efficiency, label=f"{s} rdzenie", marker='o')
    plt.xlabel("Liczba punktów")
    plt.ylabel("Efektywność równoległa")
    plt.title(f"Efektywność równoległa ({SOURCE_TAG})")
    plt.legend()
    plt.grid(True)
    plt.savefig("efektywnosc_lokalna.png", dpi=300)
    plt.show()

    print(f"\n✅ Wyniki zapisano do pliku: {OUTPUT_FILE}")
