import subprocess # Import modułu do uruchamiania zewnętrznych procesów
import os # Import modułu do operacji na systemie plików
import csv # Import modułu do obsługi plików CSV
import time # Import modułu czasu (opcjonalnie, tutaj głównie używany wewnątrz C++)
import matplotlib.pyplot as plt # Import biblioteki do tworzenia wykresów
import pandas as pd # Import biblioteki pandas do analizy danych
import numpy as np # Import biblioteki numpy do obliczeń numerycznych i generowania macierzy
from PIL import Image # Import biblioteki Pillow do tworzenia i zapisu obrazów

# ==========================================
# KONFIGURACJA
# ==========================================
EXECUTABLE = "./image_proc" # Ścieżka do skompilowanego programu C++
TEST_IMG_DIR = "test_images" # Nazwa katalogu na obrazy testowe
CSV_FILE = "wyniki_benchmarku.csv" # Nazwa pliku wyjściowego z danymi
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
    if not os.path.exists(TEST_IMG_DIR): # Sprawdzenie czy katalog istnieje
        os.makedirs(TEST_IMG_DIR) # Utworzenie katalogu jeśli brak

    print(f"--- Generowanie {len(RESOLUTIONS)} obrazów testowych ---") # Komunikat
    image_paths = [] # Lista do przechowywania ścieżek wygenerowanych obrazów
    for w, h in RESOLUTIONS: # Pętla po zdefiniowanych rozdzielczościach
        filename = os.path.join(TEST_IMG_DIR, f"img_{w}x{h}.png") # Tworzenie ścieżki pliku
        # Generujemy losowy szum, aby plik miał odpowiedni rozmiar
        if not os.path.exists(filename): # Jeśli plik jeszcze nie istnieje
            arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8) # Generowanie losowej macierzy pikseli RGB
            img = Image.fromarray(arr) # Konwersja macierzy na obraz
            img.save(filename) # Zapis obrazu do pliku
        image_paths.append((w, h, filename)) # Dodanie danych do listy
    return image_paths # Zwrócenie listy utworzonych obrazów

# ==========================================
# 2. URUCHAMIANIE BENCHMARKU (C++)
# ==========================================
def run_benchmark(image_paths):
    results_res = []   # Lista na wyniki: Rozdzielczość vs Czas
    results_block = [] # Lista na wyniki: Wątki vs Czas (tylko dla największego obrazu)

    print(f"--- Uruchamianie {EXECUTABLE} ---") # Komunikat startowy

    # Otwieramy plik CSV do zapisu
    with open(CSV_FILE, 'w', newline='') as csvfile: # Otwarcie pliku w trybie zapisu
        fieldnames = ['width', 'height', 'pixels', 'cpu_ms', 'gpu_ms', 'threads_per_block', 'block_ms'] # Nagłówki kolumn
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames) # Utworzenie obiektu piszącego CSV
        writer.writeheader() # Zapisanie nagłówków

        for idx, (w, h, filepath) in enumerate(image_paths): # Pętla po plikach testowych
            print(f"Przetwarzanie: {w}x{h}...") # Informacja o postępie

            # Wywołanie programu C++: ./image_proc 2 <plik> <algo>
            # Tryb 2 to Twój tryb benchmarku
            cmd = [EXECUTABLE, "2", filepath, ALGO_ID] # Budowanie listy argumentów
            result = subprocess.run(cmd, capture_output=True, text=True) # Uruchomienie procesu

            if result.returncode != 0: # Sprawdzenie czy wystąpił błąd
                print(f"Błąd przy pliku {filepath}: {result.stderr}") # Wypisanie błędu
                continue # Przejście do następnego pliku

            # Parsowanie wyjścia C++
            # Oczekujemy linii: RES_DATA w h cpu gpu
            # Oraz wielu linii: BLOCK_DATA threads ms
            lines = result.stdout.split('\n') # Podział wyjścia na linie

            cpu_time = 0 # Zmienna tymczasowa na czas CPU
            gpu_time = 0 # Standardowe 16x16 - czas GPU

            current_blocks = {} # Słownik przechowujący wyniki bloków dla tego obrazu

            for line in lines: # Iteracja po liniach outputu
                parts = line.split() # Podział linii na części
                if not parts: continue # Pominięcie pustych linii

                if parts[0] == "RES_DATA": # Znaleziono dane ogólne
                    # Format: RES_DATA <w> <h> <cpu> <gpu>
                    cpu_time = float(parts[3]) # Parsowanie czasu CPU
                    gpu_time = float(parts[4]) # Parsowanie czasu GPU
                    results_res.append({ # Dodanie do listy wyników
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

                elif parts[0] == "BLOCK_DATA": # Znaleziono dane bloków
                    # Format: BLOCK_DATA <threads> <ms>
                    t = int(parts[1]) # Liczba wątków
                    ms = float(parts[2]) # Czas
                    current_blocks[t] = ms # Zapis do słownika

            # Pobieramy dane o wątkach TYLKO dla największego obrazu (najlepsze skalowanie)
            if idx == len(image_paths) - 1: # Sprawdzenie czy to ostatni element
                for t, ms in current_blocks.items(): # Pętla po zebranych danych bloków
                    results_block.append({'threads': t, 'ms': ms}) # Dodanie do listy wyników bloków

    return results_res, results_block # Zwrócenie zebranych danych

# ==========================================
# 3. RYSOWANIE WYKRESÓW
# ==========================================
def plot_results(res_data, block_data):
    df_res = pd.DataFrame(res_data) # Konwersja listy słowników na DataFrame
    df_block = pd.DataFrame(block_data) # Konwersja bloków na DataFrame

    plt.style.use('ggplot') # Ładny styl wykresów (ggplot)

    # --- WYKRES 1: CPU vs GPU (Zależność od rozmiaru) ---
    plt.figure(figsize=(10, 6)) # Nowa figura wykresu

    # Sortujemy po liczbie pikseli
    df_res = df_res.sort_values('pixels') # Sortowanie danych

    # Etykiety osi X (rozdzielczości)
    labels = [f"{r['width']}x{r['height']}" for i, r in df_res.iterrows()] # Lista etykiet
    x_pos = np.arange(len(labels)) # Pozycje na osi X

    plt.plot(x_pos, df_res['cpu_ms'], marker='o', label='CPU', linewidth=2, color='red') # Linia CPU
    plt.plot(x_pos, df_res['gpu_ms'], marker='s', label='GPU (CUDA)', linewidth=2, color='blue') # Linia GPU

    plt.xticks(x_pos, labels, rotation=45) # Ustawienie etykiet osi X
    plt.title(f'Porównanie czasu wykonania CPU vs GPU\n(Algorytm ID: {ALGO_ID})') # Tytuł
    plt.xlabel('Rozdzielczość obrazu') # Opis osi X
    plt.ylabel('Czas wykonania (ms)') # Opis osi Y
    plt.legend() # Legenda
    plt.grid(True) # Siatka
    plt.tight_layout() # Dopasowanie marginesów
    plt.savefig('wykres_cpu_vs_gpu.png') # Zapis do pliku
    print("Zapisano: wykres_cpu_vs_gpu.png")

    # --- WYKRES 2: Przyspieszenie (Speedup) ---
    plt.figure(figsize=(10, 6)) # Nowa figura
    speedup = df_res['cpu_ms'] / df_res['gpu_ms'] # Obliczenie przyspieszenia

    bars = plt.bar(x_pos, speedup, color='green', alpha=0.7) # Wykres słupkowy
    plt.xticks(x_pos, labels, rotation=45) # Etykiety osi X
    plt.title('Przyspieszenie GPU względem CPU (Speedup)') # Tytuł
    plt.xlabel('Rozdzielczość obrazu') # Opis X
    plt.ylabel('Krotność przyspieszenia (x razy szybciej)') # Opis Y

    # Dodanie wartości nad słupkami
    for bar in bars: # Pętla po słupkach
        height = bar.get_height() # Pobranie wysokości
        plt.text(bar.get_x() + bar.get_width()/2., height, # Wstawienie tekstu
                 f'{height:.1f}x', ha='center', va='bottom')

    plt.tight_layout() # Dopasowanie
    plt.savefig('wykres_speedup.png') # Zapis
    print("Zapisano: wykres_speedup.png")

    # --- WYKRES 3: Czas vs Liczba wątków w bloku ---
    #
    if not df_block.empty: # Jeśli są dane
        plt.figure(figsize=(8, 6)) # Nowa figura
        df_block = df_block.sort_values('threads') # Sortowanie po liczbie wątków

        plt.plot(df_block['threads'].astype(str), df_block['ms'], marker='D', color='purple', linestyle='--') # Rysowanie

        plt.title(f'Wydajność w zależności od liczby wątków w bloku\n(Dla największego obrazu: {labels[-1]})') # Tytuł
        plt.xlabel('Liczba wątków w bloku (Block Size)') # Opis X
        plt.ylabel('Czas wykonania (ms)') # Opis Y
        plt.grid(True) # Siatka
        plt.tight_layout() # Dopasowanie
        plt.savefig('wykres_watki.png') # Zapis
        print("Zapisano: wykres_watki.png")

if __name__ == "__main__":
    # 1. Generuj obrazy
    imgs = generate_test_images() # Generowanie

    # 2. Uruchom C++ i zbierz dane
    res_data, block_data = run_benchmark(imgs) # Benchmark

    # 3. Narysuj wykresy
    plot_results(res_data, block_data) # Wizualizacja

    print("\nGotowe! Sprawdź pliki .png oraz wyniki_benchmarku.csv") # Koniec