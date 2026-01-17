import subprocess
import os
import csv
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image

# ==========================================
# KONFIGURACJA
# ==========================================
EXECUTABLE = "./image_proc"
TEST_IMG_DIR = "test_images"
CSV_FILE = "wyniki_benchmarku.csv"
ALGO_ID = "2"  # 1=Sepia, 2=Gaussian, 3=Sobel (Testujemy Gaussa jako reprezentatywny)

# Rozdzielczości do testowania (Szerokość, Wysokość)
RESOLUTIONS = [
    (640, 480),
    (1280, 720),    # HD
    (1920, 1080),   # Full HD
    (2560, 1440),   # 2K
    (3840, 2160),   # 4K
    (5120, 2880)    # 5K (duże obciążenie)
]

# ==========================================
# 1. GENEROWANIE OBRAZÓW TESTOWYCH
# ==========================================
def generate_test_images():
    if not os.path.exists(TEST_IMG_DIR):
        os.makedirs(TEST_IMG_DIR)

    print(f"--- Generowanie {len(RESOLUTIONS)} obrazów testowych ---")
    image_paths = []
    for w, h in RESOLUTIONS:
        filename = os.path.join(TEST_IMG_DIR, f"img_{w}x{h}.png")
        # Generujemy losowy szum, aby plik miał odpowiedni rozmiar
        if not os.path.exists(filename):
            arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            img = Image.fromarray(arr)
            img.save(filename)
        image_paths.append((w, h, filename))
    return image_paths

# ==========================================
# 2. URUCHAMIANIE BENCHMARKU (C++)
# ==========================================
def run_benchmark(image_paths):
    results_res = []   # Dane: Resolution vs Time
    results_block = [] # Dane: Threads vs Time (tylko dla największego obrazu)

    print(f"--- Uruchamianie {EXECUTABLE} ---")

    # Otwieramy plik CSV do zapisu
    with open(CSV_FILE, 'w', newline='') as csvfile:
        fieldnames = ['width', 'height', 'pixels', 'cpu_ms', 'gpu_ms', 'threads_per_block', 'block_ms']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for idx, (w, h, filepath) in enumerate(image_paths):
            print(f"Przetwarzanie: {w}x{h}...")

            # Wywołanie programu C++: ./image_proc 2 <plik> <algo>
            # Tryb 2 to Twój tryb benchmarku
            cmd = [EXECUTABLE, "2", filepath, ALGO_ID]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"Błąd przy pliku {filepath}: {result.stderr}")
                continue

            # Parsowanie wyjścia C++
            # Oczekujemy linii: RES_DATA w h cpu gpu
            # Oraz wielu linii: BLOCK_DATA threads ms
            lines = result.stdout.split('\n')

            cpu_time = 0
            gpu_time = 0 # Standardowe 16x16

            current_blocks = {} # Przechowuje wyniki bloków dla tego obrazu

            for line in lines:
                parts = line.split()
                if not parts: continue

                if parts[0] == "RES_DATA":
                    # Format: RES_DATA <w> <h> <cpu> <gpu>
                    cpu_time = float(parts[3])
                    gpu_time = float(parts[4])
                    results_res.append({
                        'width': int(parts[1]),
                        'height': int(parts[2]),
                        'pixels': int(parts[1]) * int(parts[2]),
                        'cpu_ms': cpu_time,
                        'gpu_ms': gpu_time
                    })

                    # Zapis podstawowy do CSV (bez danych o blokach szczegółowych)
                    writer.writerow({
                        'width': w, 'height': h, 'pixels': w*h,
                        'cpu_ms': cpu_time, 'gpu_ms': gpu_time,
                        'threads_per_block': 256, 'block_ms': gpu_time
                    })

                elif parts[0] == "BLOCK_DATA":
                    # Format: BLOCK_DATA <threads> <ms>
                    t = int(parts[1])
                    ms = float(parts[2])
                    current_blocks[t] = ms

            # Pobieramy dane o wątkach TYLKO dla największego obrazu (najlepsze skalowanie)
            if idx == len(image_paths) - 1:
                for t, ms in current_blocks.items():
                    results_block.append({'threads': t, 'ms': ms})

    return results_res, results_block

# ==========================================
# 3. RYSOWANIE WYKRESÓW
# ==========================================
def plot_results(res_data, block_data):
    df_res = pd.DataFrame(res_data)
    df_block = pd.DataFrame(block_data)

    plt.style.use('ggplot') # Ładny styl wykresów

    # --- WYKRES 1: CPU vs GPU (Zależność od rozmiaru) ---
    plt.figure(figsize=(10, 6))

    # Sortujemy po liczbie pikseli
    df_res = df_res.sort_values('pixels')

    # Etykiety osi X (rozdzielczości)
    labels = [f"{r['width']}x{r['height']}" for i, r in df_res.iterrows()]
    x_pos = np.arange(len(labels))

    plt.plot(x_pos, df_res['cpu_ms'], marker='o', label='CPU', linewidth=2, color='red')
    plt.plot(x_pos, df_res['gpu_ms'], marker='s', label='GPU (CUDA)', linewidth=2, color='blue')

    plt.xticks(x_pos, labels, rotation=45)
    plt.title(f'Porównanie czasu wykonania CPU vs GPU\n(Algorytm ID: {ALGO_ID})')
    plt.xlabel('Rozdzielczość obrazu')
    plt.ylabel('Czas wykonania (ms)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('wykres_cpu_vs_gpu.png')
    print("Zapisano: wykres_cpu_vs_gpu.png")

    # --- WYKRES 2: Przyspieszenie (Speedup) ---
    plt.figure(figsize=(10, 6))
    speedup = df_res['cpu_ms'] / df_res['gpu_ms']

    bars = plt.bar(x_pos, speedup, color='green', alpha=0.7)
    plt.xticks(x_pos, labels, rotation=45)
    plt.title('Przyspieszenie GPU względem CPU (Speedup)')
    plt.xlabel('Rozdzielczość obrazu')
    plt.ylabel('Krotność przyspieszenia (x razy szybciej)')

    # Dodanie wartości nad słupkami
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}x', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('wykres_speedup.png')
    print("Zapisano: wykres_speedup.png")

    # --- WYKRES 3: Czas vs Liczba wątków w bloku ---
    #
    if not df_block.empty:
        plt.figure(figsize=(8, 6))
        df_block = df_block.sort_values('threads')

        plt.plot(df_block['threads'].astype(str), df_block['ms'], marker='D', color='purple', linestyle='--')

        plt.title(f'Wydajność w zależności od liczby wątków w bloku\n(Dla największego obrazu: {labels[-1]})')
        plt.xlabel('Liczba wątków w bloku (Block Size)')
        plt.ylabel('Czas wykonania (ms)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('wykres_watki.png')
        print("Zapisano: wykres_watki.png")

if __name__ == "__main__":
    # 1. Generuj obrazy
    imgs = generate_test_images()

    # 2. Uruchom C++ i zbierz dane
    res_data, block_data = run_benchmark(imgs)

    # 3. Narysuj wykresy
    plot_results(res_data, block_data)

    print("\nGotowe! Sprawdź pliki .png oraz wyniki_benchmarku.csv")