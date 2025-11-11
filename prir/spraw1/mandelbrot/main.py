import numpy as np
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool, cpu_count
from functools import partial
import pandas as pd
import psutil
import threading

X_MIN, X_MAX = -2.0, 1.0
Y_MIN, Y_MAX = -1.0, 1.0


def mandelbrot_pixel(c, max_iter=100):
    z = 0 + 0j
    for i in range(max_iter):
        z = z * z + c
        if (z.real * z.real + z.imag * z.imag) > 4.0:
            return i
    return max_iter


def mandelbrot_array(width, height, max_iter=100):
    xs = np.linspace(X_MIN, X_MAX, width)
    ys = np.linspace(Y_MIN, Y_MAX, height)
    img = np.empty((height, width), dtype=np.int32)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            img[j, i] = mandelbrot_pixel(complex(x, y), max_iter=max_iter)
    return img


def monitor_cpu(interval, stop_event, samples):
    """Monitoruje CPU podczas obliczeń i zapisuje wartości w samples."""
    while not stop_event.is_set():
        samples.append(psutil.cpu_percent(percpu=True))
        time.sleep(interval)


def measure_with_cpu_monitor(func, *args, **kwargs):
    """Uruchamia dowolną funkcję z monitoringiem CPU i zwraca: wynik, czas, cpu_mean, cpu_max"""
    samples = []
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor_cpu, args=(0.1, stop_event, samples))
    monitor_thread.start()

    start = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start

    stop_event.set()
    monitor_thread.join()

    if samples:
        cpu_mean = np.mean(samples)
        per_core_mean = np.mean(samples, axis=0)
        cpu_max = np.max(per_core_mean)
    else:
        cpu_mean, cpu_max = 0, 0

    return result, elapsed, cpu_mean, cpu_max


def _compute_row_range(row_range, width, height):
    row_start, row_end = row_range
    xs = np.linspace(X_MIN, X_MAX, width)
    ys = np.linspace(Y_MIN, Y_MAX, height)[row_start:row_end]
    out = np.empty((row_end - row_start, width), dtype=np.int32)
    for jj, y in enumerate(ys):
        for i, x in enumerate(xs):
            out[jj, i] = mandelbrot_pixel(complex(x, y))
    return row_start, out


def mandelbrot_parallel(width, height, n_workers, save_image=False):
    rows = np.linspace(0, height, n_workers + 1, dtype=int)
    ranges = [(int(rows[i]), int(rows[i + 1])) for i in range(len(rows) - 1)]

    with Pool(processes=n_workers) as pool:
        func = partial(_compute_row_range, width=width, height=height)
        results = pool.map(func, ranges)

    if save_image:
        img = np.zeros((height, width), dtype=np.int32)
        for row_start, block in results:
            img[row_start:row_start + block.shape[0], :] = block
        plt.imsave(f'mandelbrot_{width}x{height}.png', img, cmap='hot')

    return results


def run_experiments(resolutions, workers):
    data_records = []

    for (width, height) in resolutions:
        pixels = width * height
        print(f"\nRozdzielczość: {width}x{height} ({pixels / 1e6:.2f} MPix)")

        # # --- Sekwencyjnie z pomiarem CPU ---
        _, seq_time, cpu_mean_seq, cpu_max_seq = measure_with_cpu_monitor(mandelbrot_array, width, height)
        data_records.append((width, height, pixels, 1, seq_time, 1.0, 1.0, cpu_mean_seq, cpu_max_seq))
        print(f"  Sekwencyjnie: {seq_time:.3f}s | CPUavg={cpu_mean_seq:.1f}% | CPUmax={cpu_max_seq:.1f}%")

        # --- Równolegle ---
        for n in workers:
            if n > cpu_count():
                print(f"  Pomijam {n} workerów (więcej niż rdzeni {cpu_count()})")
                continue
            save_img = (n == max(workers))
            _, par_time, cpu_mean, cpu_max = measure_with_cpu_monitor(mandelbrot_parallel, width, height, n, save_img)
            speedup = seq_time / par_time
            eff = speedup / n
            data_records.append((width, height, pixels, n, par_time, speedup, eff, cpu_mean, cpu_max))
            print(
                f"  Równolegle ({n}): {par_time:.3f}s | Speedup={speedup:.2f} | Efektywność={eff:.2f} | "
                f"CPUavg={cpu_mean:.1f}% | CPUmax={cpu_max:.1f}%"
            )

    df = pd.DataFrame(
        data_records,
        columns=[
            'width', 'height', 'pixels', 'workers',
            'time', 'speedup', 'efficiency',
            'cpu_mean', 'cpu_max'
        ]
    )
    df.to_csv('results.csv', index=False)
    return df


def plot_all(df):
    worker_sets = sorted(df['workers'].unique())

    # --- Czas vs piksele ---
    plt.figure(figsize=(10, 6))
    for n in worker_sets:
        subset = df[df['workers'] == n].sort_values('pixels')
        plt.plot(subset['pixels'] / 1e6, subset['time'], 'o-', label=f'{n} rdzeni')
    plt.xlabel('Liczba pikseli [MPix]')
    plt.ylabel('Czas obliczeń [s]')
    plt.title('Czas obliczeń vs liczba pikseli')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('czas_vs_piksele_local.png', dpi=200)

    # --- Średnie obciążenie CPU ---
    plt.figure(figsize=(10, 6))
    for n in worker_sets:
        subset = df[df['workers'] == n].sort_values('pixels')
        plt.plot(subset['pixels'] / 1e6, subset['cpu_mean'], 'o-', label=f'{n} rdzeni')
    plt.xlabel('Liczba pikseli [MPix]')
    plt.ylabel('Średnie obciążenie CPU [%]')
    plt.title('Średnie obciążenie CPU vs liczba pikseli')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('cpu_usage_vs_pixels_local.png', dpi=200)

    plt.figure(figsize=(10, 6))
    for n in worker_sets:
        subset = df[df['workers'] == n].sort_values('pixels')
        plt.plot(subset['pixels'] / 1e6, subset['speedup'], 'o-', label=f'{n} rdzeni')
    plt.xlabel('Liczba pikseli [MPix]')
    plt.ylabel('Speedup')
    plt.title('Speedup vs liczba pikseli')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('speedup_vs_rdzenie_local.png', dpi=300)


def main():
    resolutions = [
        (320, 240), (400, 300), (480, 320), (800, 600),
        (1024, 768), (1280, 720), (1366, 768), (1920, 1080),
        (2048, 1024), (2048, 1152), (2560, 1440), (2560, 2048)
    ]
    workers = [2, 4]
    df = run_experiments(resolutions, workers)
    plot_all(df)
    print("\n✅ Zapisano wszystkie wyniki, obrazy, wykresy i obciążenie CPU (również sekwencyjnie).")


if __name__ == '__main__':
    main()
