import pandas as pd             # Import biblioteki pandas do analizy danych
import matplotlib.pyplot as plt # Import biblioteki matplotlib do rysowania wykresów
import numpy as np              # Import biblioteki numpy do obliczeń
import os                       # Import biblioteki os do obsługi plików

# --- KONFIGURACJA ---
FILE_LOCAL = 'results.csv'      # Nazwa lokalnego pliku z wynikami
FILE_COLAB = 'results_colab.csv' # Nazwa pliku z wynikami z Google Colab
FRAKTAL = 'Julia' # Analizujemy Julię # Wybór typu fraktala do analizy
FILE_BLOCKS = 'benchmark_blocks.csv' # Nazwa pliku z wynikami testu bloków

# Sprawdzenie plików
if not os.path.exists(FILE_LOCAL) or not os.path.exists(FILE_COLAB): # Sprawdzenie czy oba pliki CSV istnieją
    print("BŁĄD: Brakuje plików .csv!") # Komunikat błędu
    exit()                      # Wyjście z programu

# 1. Wczytanie
print("Wczytywanie danych i filtrowanie...") # Informacja dla użytkownika
df_loc = pd.read_csv(FILE_LOCAL) # Wczytanie lokalnych wyników
df_col = pd.read_csv(FILE_COLAB) # Wczytanie wyników z Colab
df_blocks = pd.read_csv(FILE_BLOCKS) # Wczytanie wyników benchmarku bloków

# Filtrowanie tylko wybranego fraktala
df_loc = df_loc[df_loc['type'] == FRAKTAL] # Pozostawienie wierszy tylko dla wybranego fraktala (lokalne)
df_col = df_col[df_col['type'] == FRAKTAL] # Pozostawienie wierszy tylko dla wybranego fraktala (Colab)

# 2. Rozdzielenie danych na CPU i GPU

# LOKALNE
loc_cpu = df_loc[df_loc['device'] == 'CPU'][['width', 'height', 'timeMs']].copy() # Wyodrębnienie danych CPU z lokalnego pliku
loc_cpu.rename(columns={'timeMs': 'time_local_cpu'}, inplace=True) # Zmiana nazwy kolumny czasu na unikalną

loc_gpu = df_loc[df_loc['device'].str.contains('GPU')][['width', 'height', 'timeMs']].copy() # Wyodrębnienie danych GPU (lokalne)
loc_gpu.rename(columns={'timeMs': 'time_local_gpu'}, inplace=True) # Zmiana nazwy kolumny czasu

# COLAB
col_cpu = df_col[df_col['device'] == 'CPU'][['width', 'height', 'timeMs']].copy() # Wyodrębnienie danych CPU z Colab
col_cpu.rename(columns={'timeMs': 'time_colab_cpu'}, inplace=True) # Zmiana nazwy kolumny czasu

col_gpu = df_col[df_col['device'].str.contains('GPU')][['width', 'height', 'timeMs']].copy() # Wyodrębnienie danych GPU (Colab)
col_gpu.rename(columns={'timeMs': 'time_colab_gpu'}, inplace=True) # Zmiana nazwy kolumny czasu

# 3. Łączenie w jedną tabelę (Merging)
merged = pd.merge(loc_gpu, col_gpu, on=['width', 'height'], how='outer') # Łączenie danych GPU (lokalne i Colab) po rozdzielczości
merged = pd.merge(merged, loc_cpu, on=['width', 'height'], how='left')   # Dołączenie danych CPU (lokalne)
merged = pd.merge(merged, col_cpu, on=['width', 'height'], how='left')   # Dołączenie danych CPU (Colab)

# Sortowanie po pikselach
merged['pixels'] = merged['width'] * merged['height'] # Obliczenie liczby pikseli dla sortowania
merged = merged.sort_values('pixels') # Posortowanie ramki danych rosnąco według liczby pikseli

# 4. Obliczenia
merged['speedup_local'] = merged['time_local_cpu'] / merged['time_local_gpu'] # Obliczenie przyspieszenia lokalnego (CPU/GPU)
merged['speedup_colab'] = merged['time_colab_cpu'] / merged['time_colab_gpu'] # Obliczenie przyspieszenia w Colab (CPU/GPU)

# --- RYSOWANIE ---
print("Generowanie wykresów...") # Informacja
plt.style.use('default')         # Styl wykresów
plt.rcParams.update({'font.size': 10, 'figure.facecolor': 'white', 'axes.grid': True, 'grid.alpha': 0.3}) # Ustawienia wyglądu

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18)) # Utworzenie figury z 3 subplotami (wierszami)
width = 0.35                     # Szerokość słupków
x = np.arange(len(merged))       # Indeksy osi X
labels = [f"{int(r['width'])}x{int(r['height'])}" for i, r in merged.iterrows()] # Etykiety osi X (rozdzielczości)

# --- WYKRES 1: CPU vs CPU ---
df_cpu = merged.dropna(subset=['time_local_cpu', 'time_colab_cpu']) # Usunięcie wierszy bez danych CPU
if not df_cpu.empty:             # Jeśli są dane CPU
    x_cpu = np.arange(len(df_cpu)) # Indeksy dla CPU
    l_cpu = [f"{int(r['width'])}x{int(r['height'])}" for i, r in df_cpu.iterrows()] # Etykiety dla CPU

    rects1 = ax1.bar(x_cpu - width/2, df_cpu['time_local_cpu'], width, label='Twój CPU', color='#e67e22', edgecolor='black') # Słupki Twój CPU
    rects2 = ax1.bar(x_cpu + width/2, df_cpu['time_colab_cpu'], width, label='Colab CPU', color='#95a5a6', edgecolor='black') # Słupki Colab CPU

    ax1.set_title('1. Porównanie CPU (Lokalny vs Colab)', fontweight='bold') # Tytuł pierwszego wykresu
    ax1.set_ylabel('Czas (ms)')  # Oś Y
    ax1.set_xticks(x_cpu)        # Ustawienie punktów na osi X
    ax1.set_xticklabels(l_cpu)   # Ustawienie etykiet na osi X
    ax1.legend()                 # Legenda

    # Etykiety dla CPU
    for rect in rects1 + rects2: # Pętla po wszystkich słupkach CPU
        h = rect.get_height()    # Pobranie wysokości
        ax1.annotate(f'{int(h)}ms', (rect.get_x() + rect.get_width()/2, h), xytext=(0,3), textcoords="offset points", ha='center', fontsize=8) # Dodanie opisu wartości

else:
    ax1.text(0.5, 0.5, "Brak danych CPU", ha='center') # Informacja o braku danych

# --- WYKRES 2: GPU vs GPU (Najważniejszy) ---
ax2.bar(x - width/2, merged['time_local_gpu'], width, label='Twój RTX', color='#2ecc71', edgecolor='black') # Słupki Twój RTX
ax2.bar(x + width/2, merged['time_colab_gpu'], width, label='Colab GPU', color='#34495e', edgecolor='black') # Słupki Colab GPU

ax2.set_title('2. Porównanie GPU (Twój RTX vs Colab)', fontweight='bold') # Tytuł drugiego wykresu
ax2.set_ylabel('Czas (ms) - Skala Log') # Oś Y
ax2.set_yscale('log')            # Skala logarytmiczna dla czytelności dużych różnic
ax2.set_xticks(x)                # Oś X
ax2.set_xticklabels(labels, rotation=45) # Etykiety obrócone o 45 stopni
ax2.legend()                     # Legenda

# Etykiety nad słupkami GPU (POPRAWIONE: dodano 'ms')
def format_ms(val):              # Funkcja pomocnicza do formatowania czasu
    if pd.isna(val): return ""   # Jeśli brak danych, zwróć pusty ciąg
    if val < 10: return f"{val:.2f}ms" # Dla małych wartości (poniżej 10ms) pokaż 2 miejsca po przecinku
    return f"{int(val)}ms"       # Dla dużych wartości pokaż liczbę całkowitą

for i, row in merged.iterrows(): # Pętla po wierszach danych
    # Twój
    if pd.notna(row['time_local_gpu']): # Jeśli jest wynik lokalnego GPU
        ax2.annotate(format_ms(row['time_local_gpu']), (i - width/2, row['time_local_gpu']), ha='center', xytext=(0,5), textcoords="offset points", fontsize=8, fontweight='bold') # Dodaj opis
    # Colab
    if pd.notna(row['time_colab_gpu']): # Jeśli jest wynik GPU Colab
        ax2.annotate(format_ms(row['time_colab_gpu']), (i + width/2, row['time_colab_gpu']), ha='center', xytext=(0,5), textcoords="offset points", fontsize=8) # Dodaj opis

# --- WYKRES 3: SPEEDUP (POPRAWIONE: dodano etykiety dla Colaba) ---
if not df_cpu.empty:             # Jeśli są dane do obliczenia przyspieszenia
    # Rysowanie słupków
    ax3.bar(x_cpu - width/2, df_cpu['speedup_local'], width, label='Przyspieszenie u Ciebie', color='#27ae60', edgecolor='black') # Przyspieszenie lokalne
    ax3.bar(x_cpu + width/2, df_cpu['speedup_colab'], width, label='Przyspieszenie w Colabie', color='#7f8c8d', edgecolor='black') # Przyspieszenie Colab

    ax3.set_title('3. Ile razy GPU jest szybsze od CPU? (Krotność)', fontweight='bold') # Tytuł trzeciego wykresu
    ax3.set_ylabel('Krotność (x razy)') # Oś Y
    ax3.set_xticks(x_cpu)        # Oś X
    ax3.set_xticklabels(l_cpu)   # Etykiety osi X
    ax3.legend()                 # Legenda
    ax3.axhline(1, color='black', linestyle='--') # Linia pozioma na poziomie 1 (brak przyspieszenia)

    # Etykiety (POPRAWIONE: pętla po obu zestawach)
    for i, row in enumerate(df_cpu.itertuples()): # Pętla po wynikach
        # Lokalne (Zielone)
        ax3.annotate(f"{row.speedup_local:.0f}x", (i - width/2, row.speedup_local), ha='center', va='bottom', xytext=(0,3), textcoords="offset points", fontsize=9, fontweight='bold', color='darkgreen') # Opis krotności lokalnej
        # Colab (Szare) - TEGO BRAKOWAŁO
        ax3.annotate(f"{row.speedup_colab:.0f}x", (i + width/2, row.speedup_colab), ha='center', va='bottom', xytext=(0,3), textcoords="offset points", fontsize=9, fontweight='bold', color='dimgray') # Opis krotności Colab

plt.tight_layout()               # Dopasowanie układu
plt.savefig('porownanie_totalne.png', dpi=300) # Zapis głównego wykresu do pliku
print("Zrobione! Sprawdź plik porownanie_totalne.png - teraz powinno być czytelnie.") # Komunikat
plt.show()                       # Wyświetlenie wykresu (jeśli środowisko graficzne jest dostępne)


ax3 = plt.subplot(2, 1, 2)       # Nadpisanie zmiennej ax3 nowym subplotem (uwaga: to tworzy nowy układ wykresów niezależny od 'fig')
ax3.plot(df_blocks['total_threads'], df_blocks['gpu_time_ms'], 'b-D', markersize=8, linewidth=2, label='Czas renderowania 8K') # Wykres liniowy z punktami
ax3.set_title('3. Analiza wydajności bloku (Liczba wątków vs Czas w 8K)', fontweight='bold') # Tytuł
ax3.set_xlabel('Całkowita liczba wątków w bloku (side x side)') # Opis osi X
ax3.set_ylabel('Czas (ms)')      # Opis osi Y
ax3.set_xticks(df_blocks['total_threads']) # Ustawienie znaczników osi X
ax3.grid(True, alpha=0.3)        # Siatka
for i, txt in enumerate(df_blocks['gpu_time_ms']): # Pętla do dodania etykiet wartości
    ax3.annotate(f"{txt:.2f}ms", (df_blocks['total_threads'][i], df_blocks['gpu_time_ms'][i]), # Pozycjonowanie tekstu
                 textcoords="offset points", xytext=(0,10), ha='center', fontsize=9) # Przesunięcie i styl tekstu

ax3.legend()                     # Legenda

plt.tight_layout()               # Dopasowanie układu
plt.savefig('analiza_lokalna_sprzetu.png', dpi=300) # Zapis drugiego wykresu do pliku