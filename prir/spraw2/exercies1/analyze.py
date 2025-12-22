import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import math

# --- KONFIGURACJA ---
OUTPUT_DIR = 'output_images_julia'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 1. Wczytanie danych
try:
    df = pd.read_csv('results.csv')
    df['pixels_millions'] = df['pixels'] / 1_000_000
except FileNotFoundError:
    print("Błąd: Brak pliku results.csv!")
    exit()

# --- CZĘŚĆ A: KOLORY (BIAŁE TŁO - STYL "INK ON PAPER") ---
print("Generowanie obrazów (Styl: Białe tło)...")

def create_white_bg_palette():
    palette = []
    for i in range(256):
        # t od 0 do 1
        t = i / 255.0

        # Jeśli i jest bliskie 0 (tło, szybka ucieczka punktu), chcemy BIEL.
        # Im wyższe i (głębiej we fraktalu), tym ciemniejszy kolor.

        if i < 5:
            # Absolutne tło = Białe
            palette.extend([255, 255, 255])
            continue

        # Wzór: Zaczynamy od bieli, przechodzimy w niebieski/fiolet, kończymy na czerni
        # Odwracamy logikę: 1.0 to biały, 0.0 to czarny

        # Funkcja cosinusowa przesunięta tak, by startowała od jasnego
        # R: Szybko spada do zera (robi się ciemno-niebieski)
        r = int(255 * (0.5 + 0.5 * math.cos(2 * math.pi * t * 1.0)))
        # G: Też spada, dając odcienie fioletu
        g = int(255 * (0.5 + 0.5 * math.cos(2 * math.pi * t * 1.5)))
        # B: Zostaje wysokie najdłużej (dlatego dominanta niebieska)
        b = int(255 * (0.5 + 0.5 * math.sin(2 * math.pi * t + 1.0))) + 50

        # Rozjaśnianie dla niskich indeksów, żeby przejście z bieli było płynne
        if i < 50:
            factor = (50 - i) / 50.0
            r = int(r + (255 - r) * factor)
            g = int(g + (255 - g) * factor)
            b = int(b + (255 - b) * factor)

        palette.extend([min(255, r), min(255, g), min(255, b)])
    return palette

fractal_palette = create_white_bg_palette()

# Szukamy plików .bin
for filename in os.listdir('.'):
    if filename.endswith(".bin") and filename.startswith("img_"):
        try:
            parts = filename.replace('.bin', '').split('_')
            res_part = parts[-1]
            w, h = map(int, res_part.split('x'))

            with open(filename, 'rb') as f:
                raw_data = np.frombuffer(f.read(), dtype=np.uint8)

            if raw_data.size != w * h: continue

            img_array = raw_data.reshape((h, w))
            img = Image.fromarray(img_array, mode='L')
            img = img.convert("P")

            # Aplikacja palety z białym tłem
            img.putpalette(fractal_palette)

            output_name = f"{OUTPUT_DIR}/{filename.replace('.bin', '.png')}"
            img.save(output_name)
            print(f"Wygenerowano obraz: {output_name}")

        except Exception:
            pass

# --- CZĘŚĆ B: WYKRESY ---
print("Rysowanie wykresów...")

plt.style.use('default')
plt.rcParams.update({
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.5,
    'grid.linestyle': '--',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white'
})

cpu_data = df[df['device'] == 'CPU']
gpu_data_julia = df[(df['device'] == 'GPU') & (df['type'] == 'Julia')]

# 1. Wykres Liniowy CPU
if not cpu_data.empty:
    plt.figure(figsize=(10, 6))
    plt.plot(cpu_data['pixels_millions'], cpu_data['timeMs'],
             marker='o', color='#e74c3c', label='CPU Time')
    plt.title('Czas CPU (im niżej tym lepiej)')
    plt.xlabel('Mpx')
    plt.ylabel('ms')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/wykres_cpu_linear.png')
    plt.close()

# 2. Wykres Liniowy GPU
if not gpu_data_julia.empty:
    plt.figure(figsize=(10, 6))
    plt.plot(gpu_data_julia['pixels_millions'], gpu_data_julia['timeMs'],
             marker='s', color='#2980b9', label='GPU Julia')
    plt.title('Czas GPU (im niżej tym lepiej)')
    plt.xlabel('Mpx')
    plt.ylabel('ms')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/wykres_julia_gpu_linear.png')
    plt.close()

# 3. Wykres Porównawczy (Log)
common = pd.merge(cpu_data, gpu_data_julia, on=['width', 'height'], suffixes=('_cpu', '_gpu'))

if not common.empty:
    plt.figure(figsize=(12, 6))
    x = np.arange(len(common))
    width = 0.35
    plt.bar(x - width/2, common['timeMs_cpu'], width, label='CPU', color='#e74c3c', edgecolor='black')
    plt.bar(x + width/2, common['timeMs_gpu'], width, label='GPU', color='#2980b9', edgecolor='black')
    plt.yscale('log')
    plt.xticks(x, [f"{w}x{h}" for w, h in zip(common['width'], common['height'])])
    plt.ylabel('Czas (ms) - LOG')
    plt.title('Porównanie czasu (Logarytmicznie)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/wykres_porownanie_julia.png')
    plt.close()

    # --- NOWOŚĆ: 4. WYKRES PRZYSPIESZENIA (SPEEDUP FACTOR) ---
    print("Generowanie wykresu przyspieszenia...")

    # Obliczamy ile razy GPU jest szybsze (CPU time / GPU time)
    common['speedup'] = common['timeMs_cpu'] / common['timeMs_gpu']

    plt.figure(figsize=(12, 7))
    bars = plt.bar(x, common['speedup'], color='#27ae60', edgecolor='black', width=0.6)

    plt.xticks(x, [f"{w}x{h}" for w, h in zip(common['width'], common['height'])], rotation=0)
    plt.ylabel('Krotność przyspieszenia (x razy)')
    plt.title('Ile razy GPU jest szybsze od CPU?', fontsize=14, fontweight='bold')

    # Dodajemy liczby nad słupkami (np. "150x")
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}x',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/wykres_speedup.png')
    plt.close()

print(f"Gotowe! Sprawdź plik '{OUTPUT_DIR}/wykres_speedup.png' - tam zobaczysz ile razy GPU wygrywa.")