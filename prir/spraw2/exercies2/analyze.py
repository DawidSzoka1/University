import pandas as pd             # Import biblioteki pandas do analizy danych
import matplotlib.pyplot as plt # Import biblioteki matplotlib do wykresów 2D
import numpy as np              # Import biblioteki numpy do obliczeń numerycznych
import plotly.graph_objects as go # Import biblioteki plotly do interaktywnych wykresów 3D
import os                       # Import biblioteki os do operacji na plikach

# Sprawdzenie plików
required_files = ['benchmark_resolution.csv', 'benchmark_blocks.csv', 'scene_data.csv'] # Lista wymaganych plików
for f in required_files:        # Pętla sprawdzająca każdy plik
    if not os.path.exists(f):   # Jeśli plik nie istnieje
        print(f"BŁĄD: Nie znaleziono pliku '{f}'. Uruchom program C++!") # Wypisz błąd
        exit()                  # Zakończ działanie programu

# Wczytanie danych
df_res = pd.read_csv('benchmark_resolution.csv') # Wczytanie wyników benchmarku rozdzielczości
df_blocks = pd.read_csv('benchmark_blocks.csv')  # Wczytanie wyników benchmarku bloków
df_scene = pd.read_csv('scene_data.csv')         # Wczytanie danych o geometrii sceny

# ==========================================
# 1. WYKRESY WYDAJNOŚCI (Matplotlib)
# ==========================================
# Zmieniamy układ na 3x2, aby zmieścić dodatkowe analizy wątków
fig, axs = plt.subplots(3, 2, figsize=(16, 15))  # Utworzenie siatki wykresów 3 wiersze x 2 kolumny
fig.suptitle('Zaawansowana Analiza Wydajności Ray Tracingu - RTX 5080', fontsize=20) # Główny tytuł całej planszy

# --- A. Czas vs Rozdzielczość (Log Scale) ---
valid_cpu = df_res[df_res['cpu_time_ms'] > 0]    # Filtrowanie danych: bierzemy tylko te, gdzie CPU faktycznie liczyło
axs[0, 0].plot(valid_cpu['pixels'], valid_cpu['cpu_time_ms'], 'r-o', label='CPU', linewidth=2) # Rysowanie linii dla CPU (czerwona)
axs[0, 0].plot(df_res['pixels'], df_res['gpu_time_ms'], 'b-s', label='GPU', linewidth=2) # Rysowanie linii dla GPU (niebieska)
axs[0, 0].set_title('Czas wykonania vs Rozdzielczość') # Tytuł pierwszego wykresu
axs[0, 0].set_yscale('log')                      # Ustawienie skali logarytmicznej na osi Y (dla dużych różnic)
axs[0, 0].grid(True, which="both", linestyle='--', alpha=0.5) # Włączenie siatki pomocniczej
axs[0, 0].legend()                               # Wyświetlenie legendy

# --- B. Przyspieszenie (Speedup) ---
speedup = valid_cpu['cpu_time_ms'].values / df_res.loc[valid_cpu.index, 'gpu_time_ms'].values # Obliczenie przyspieszenia (Czas CPU / Czas GPU)
axs[0, 1].bar(valid_cpu['label'], speedup, color='forestgreen') # Rysowanie wykresu słupkowego przyspieszenia
axs[0, 1].set_title('Przyspieszenie GPU (Speedup Factor)') # Tytuł wykresu
axs[0, 1].set_ylabel('x-faster')                 # Opis osi Y (ile razy szybciej)
axs[0, 1].tick_params(axis='x', rotation=45)     # Obrót etykiet osi X o 45 stopni

# --- C. Czas vs Liczba Wątków w Bloku (Gęste dane) ---
# To pokazuje "sweet spot" dla Twojej karty
axs[1, 0].plot(df_blocks['total_threads'], df_blocks['gpu_time_ms'], 'o-', color='darkorange', linewidth=2.5) # Wykres liniowy zależności czasu od liczby wątków
# Zaznaczamy Warp (32 wątki) i standardowy blok (256 wątków) dla odniesienia
axs[1, 0].set_title('Wpływ rozmiaru bloku na czas (4K)') # Tytuł wykresu
axs[1, 0].set_xlabel('Całkowita liczba wątków w bloku') # Opis osi X
axs[1, 0].set_ylabel('Czas [ms]')                       # Opis osi Y
axs[1, 0].grid(True, alpha=0.3)                         # Włączenie delikatnej siatki
axs[1, 0].legend()                                      # Legenda (pusta, bo brak label w plot, ale wywołanie nie szkodzi)


# --- E. Porównanie różnych wielkości bloków (Wykres słupkowy) ---
axs[2, 0].bar(df_blocks['label'], df_blocks['gpu_time_ms'], color='teal') # Wykres słupkowy czasów dla różnych bloków
axs[2, 0].set_title('Porównanie konkretnych konfiguracji bloków') # Tytuł
axs[2, 0].set_ylabel('Czas [ms]')                        # Opis osi Y
for i, val in enumerate(df_blocks['gpu_time_ms']):       # Pętla po wartościach, żeby dodać etykiety tekstowe
    axs[2, 0].text(i, val, f"{val:.1f}", ha='center', va='bottom') # Wypisanie wartości nad słupkiem

# --- F. Tabela wyników ---
axs[2, 1].axis('off')                                    # Wyłączenie osi dla ostatniego pola (będzie tam tabela)
table_data = df_blocks[['label', 'total_threads', 'gpu_time_ms']].copy() # Przygotowanie danych do tabeli
table = axs[2, 1].table(cellText=table_data.values, colLabels=["Blok", "Wątki", "Czas GPU (ms)"], loc='center') # Rysowanie tabeli
table.scale(1, 1.5)                                      # Przeskalowanie tabeli (większe komórki)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])                # Automatyczne dopasowanie marginesów
plt.savefig('analiza_watkow_gpu.png', dpi=300)           # Zapis wykresów do pliku PNG
plt.show()                                               # Wyświetlenie wykresów

# ==========================================
# 2. INTERAKTYWNA SCENA 3D (BARDZO GĘSTA)
# ==========================================
print("Generowanie interaktywnej sceny 3D z gęstej geometrii...") # Informacja dla użytkownika

fig_3d = go.Figure(data=[go.Scatter3d(           # Tworzenie obiektu wykresu 3D plotly
    x=df_scene['x'],                             # Dane X
    y=df_scene['y'],                             # Dane Y
    z=df_scene['z'],                             # Dane Z
    mode='markers',                              # Tryb punktowy
    marker=dict(                                 # Konfiguracja markerów (punktów)
        size=3, # Małe punkciki, bo jest ich dużo
        color=df_scene[['r', 'g', 'b']].values,  # Kolorowanie punktów wartościami RGB z pliku
        opacity=0.9                              # Przezroczystość
    )
)])

fig_3d.update_layout(                            # Aktualizacja wyglądu wykresu
    title=f"Wizualizacja Inicjałów DS - {len(df_scene)} punktów (Wysoka Jakość)", # Tytuł z liczbą punktów
    scene=dict(                                  # Ustawienia sceny 3D
        xaxis=dict(visible=False),               # Ukrycie osi X
        yaxis=dict(visible=False),               # Ukrycie osi Y
        zaxis=dict(visible=False),               # Ukrycie osi Z
        bgcolor='black',                         # Czarne tło sceny
        aspectmode='data'                        # Zachowanie proporcji danych
    ),
    paper_bgcolor='black',                       # Czarne tło papieru (obszaru wykresu)
    font=dict(color='white')                     # Biała czcionka
)

fig_3d.show()                                    # Wyświetlenie interaktywnego wykresu w przeglądarce