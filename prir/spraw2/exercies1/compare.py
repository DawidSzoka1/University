import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- KONFIGURACJA ---
FILE_LOCAL = 'results.csv'
FILE_COLAB = 'results_colab.csv'
FRAKTAL = 'Julia' # Analizujemy Julię
FILE_BLOCKS = 'benchmark_blocks.csv'

# Sprawdzenie plików
if not os.path.exists(FILE_LOCAL) or not os.path.exists(FILE_COLAB):
    print("BŁĄD: Brakuje plików .csv!")
    exit()

# 1. Wczytanie
print("Wczytywanie danych i filtrowanie...")
df_loc = pd.read_csv(FILE_LOCAL)
df_col = pd.read_csv(FILE_COLAB)
df_blocks = pd.read_csv(FILE_BLOCKS)

# Filtrowanie tylko wybranego fraktala
df_loc = df_loc[df_loc['type'] == FRAKTAL]
df_col = df_col[df_col['type'] == FRAKTAL]

# 2. Rozdzielenie danych na CPU i GPU

# LOKALNE
loc_cpu = df_loc[df_loc['device'] == 'CPU'][['width', 'height', 'timeMs']].copy()
loc_cpu.rename(columns={'timeMs': 'time_local_cpu'}, inplace=True)

loc_gpu = df_loc[df_loc['device'].str.contains('GPU')][['width', 'height', 'timeMs']].copy()
loc_gpu.rename(columns={'timeMs': 'time_local_gpu'}, inplace=True)

# COLAB
col_cpu = df_col[df_col['device'] == 'CPU'][['width', 'height', 'timeMs']].copy()
col_cpu.rename(columns={'timeMs': 'time_colab_cpu'}, inplace=True)

col_gpu = df_col[df_col['device'].str.contains('GPU')][['width', 'height', 'timeMs']].copy()
col_gpu.rename(columns={'timeMs': 'time_colab_gpu'}, inplace=True)

# 3. Łączenie w jedną tabelę (Merging)
merged = pd.merge(loc_gpu, col_gpu, on=['width', 'height'], how='outer')
merged = pd.merge(merged, loc_cpu, on=['width', 'height'], how='left')
merged = pd.merge(merged, col_cpu, on=['width', 'height'], how='left')

# Sortowanie po pikselach
merged['pixels'] = merged['width'] * merged['height']
merged = merged.sort_values('pixels')

# 4. Obliczenia
merged['speedup_local'] = merged['time_local_cpu'] / merged['time_local_gpu']
merged['speedup_colab'] = merged['time_colab_cpu'] / merged['time_colab_gpu']

# --- RYSOWANIE ---
print("Generowanie wykresów...")
plt.style.use('default')
plt.rcParams.update({'font.size': 10, 'figure.facecolor': 'white', 'axes.grid': True, 'grid.alpha': 0.3})

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18))
width = 0.35
x = np.arange(len(merged))
labels = [f"{int(r['width'])}x{int(r['height'])}" for i, r in merged.iterrows()]

# --- WYKRES 1: CPU vs CPU ---
df_cpu = merged.dropna(subset=['time_local_cpu', 'time_colab_cpu'])
if not df_cpu.empty:
    x_cpu = np.arange(len(df_cpu))
    l_cpu = [f"{int(r['width'])}x{int(r['height'])}" for i, r in df_cpu.iterrows()]

    rects1 = ax1.bar(x_cpu - width/2, df_cpu['time_local_cpu'], width, label='Twój CPU', color='#e67e22', edgecolor='black')
    rects2 = ax1.bar(x_cpu + width/2, df_cpu['time_colab_cpu'], width, label='Colab CPU', color='#95a5a6', edgecolor='black')

    ax1.set_title('1. Porównanie CPU (Lokalny vs Colab)', fontweight='bold')
    ax1.set_ylabel('Czas (ms)')
    ax1.set_xticks(x_cpu)
    ax1.set_xticklabels(l_cpu)
    ax1.legend()

    # Etykiety dla CPU
    for rect in rects1 + rects2:
        h = rect.get_height()
        ax1.annotate(f'{int(h)}ms', (rect.get_x() + rect.get_width()/2, h), xytext=(0,3), textcoords="offset points", ha='center', fontsize=8)

else:
    ax1.text(0.5, 0.5, "Brak danych CPU", ha='center')

# --- WYKRES 2: GPU vs GPU (Najważniejszy) ---
ax2.bar(x - width/2, merged['time_local_gpu'], width, label='Twój RTX', color='#2ecc71', edgecolor='black')
ax2.bar(x + width/2, merged['time_colab_gpu'], width, label='Colab GPU', color='#34495e', edgecolor='black')

ax2.set_title('2. Porównanie GPU (Twój RTX vs Colab)', fontweight='bold')
ax2.set_ylabel('Czas (ms) - Skala Log')
ax2.set_yscale('log')
ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=45)
ax2.legend()

# Etykiety nad słupkami GPU (POPRAWIONE: dodano 'ms')
def format_ms(val):
    if pd.isna(val): return ""
    if val < 10: return f"{val:.2f}ms" # Dla bardzo małych czasów pokaż przecinek
    return f"{int(val)}ms"

for i, row in merged.iterrows():
    # Twój
    if pd.notna(row['time_local_gpu']):
        ax2.annotate(format_ms(row['time_local_gpu']), (i - width/2, row['time_local_gpu']), ha='center', xytext=(0,5), textcoords="offset points", fontsize=8, fontweight='bold')
    # Colab
    if pd.notna(row['time_colab_gpu']):
        ax2.annotate(format_ms(row['time_colab_gpu']), (i + width/2, row['time_colab_gpu']), ha='center', xytext=(0,5), textcoords="offset points", fontsize=8)

# --- WYKRES 3: SPEEDUP (POPRAWIONE: dodano etykiety dla Colaba) ---
if not df_cpu.empty:
    # Rysowanie słupków
    ax3.bar(x_cpu - width/2, df_cpu['speedup_local'], width, label='Przyspieszenie u Ciebie', color='#27ae60', edgecolor='black')
    ax3.bar(x_cpu + width/2, df_cpu['speedup_colab'], width, label='Przyspieszenie w Colabie', color='#7f8c8d', edgecolor='black')

    ax3.set_title('3. Ile razy GPU jest szybsze od CPU? (Krotność)', fontweight='bold')
    ax3.set_ylabel('Krotność (x razy)')
    ax3.set_xticks(x_cpu)
    ax3.set_xticklabels(l_cpu)
    ax3.legend()
    ax3.axhline(1, color='black', linestyle='--')

    # Etykiety (POPRAWIONE: pętla po obu zestawach)
    for i, row in enumerate(df_cpu.itertuples()):
        # Lokalne (Zielone)
        ax3.annotate(f"{row.speedup_local:.0f}x", (i - width/2, row.speedup_local), ha='center', va='bottom', xytext=(0,3), textcoords="offset points", fontsize=9, fontweight='bold', color='darkgreen')
        # Colab (Szare) - TEGO BRAKOWAŁO
        ax3.annotate(f"{row.speedup_colab:.0f}x", (i + width/2, row.speedup_colab), ha='center', va='bottom', xytext=(0,3), textcoords="offset points", fontsize=9, fontweight='bold', color='dimgray')

plt.tight_layout()
plt.savefig('porownanie_totalne.png', dpi=300)
print("Zrobione! Sprawdź plik porownanie_totalne.png - teraz powinno być czytelnie.")
plt.show()


ax3 = plt.subplot(2, 1, 2)
ax3.plot(df_blocks['total_threads'], df_blocks['gpu_time_ms'], 'b-D', markersize=8, linewidth=2, label='Czas renderowania 8K')
ax3.set_title('3. Analiza wydajności bloku (Liczba wątków vs Czas w 8K)', fontweight='bold')
ax3.set_xlabel('Całkowita liczba wątków w bloku (side x side)')
ax3.set_ylabel('Czas (ms)')
ax3.set_xticks(df_blocks['total_threads'])
ax3.grid(True, alpha=0.3)
for i, txt in enumerate(df_blocks['gpu_time_ms']):
    ax3.annotate(f"{txt:.2f}ms", (df_blocks['total_threads'][i], df_blocks['gpu_time_ms'][i]),
                 textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

ax3.legend()

plt.tight_layout()
plt.savefig('analiza_lokalna_sprzetu.png', dpi=300)