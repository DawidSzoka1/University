import pandas as pd                             # Import biblioteki pandas do analizy i wczytywania danych
import matplotlib.pyplot as plt                 # Import biblioteki matplotlib do tworzenia wykresów
import os                                       # Import modułu os do operacji na plikach (np. sprawdzanie istnienia)

# Wczytanie danych (upewnij się, że pliki są w tym samym folderze)
LOCAL_FILE = 'results.csv'                      # Zdefiniowanie nazwy pliku z wynikami lokalnymi
COLAB_FILE = 'results_colab.csv'                # Zdefiniowanie nazwy pliku z wynikami z Google Colab

def plot_benchmarks():                          # Definicja głównej funkcji rysującej wykresy
    if not os.path.exists(LOCAL_FILE) or not os.path.exists(COLAB_FILE): # Sprawdzenie czy oba pliki CSV istnieją
        print("BŁĄD: Brakuje plików results.csv lub results_colab.csv!") # Wypisanie komunikatu błędu
        return                                  # Zakończenie funkcji w przypadku braku plików

    local = pd.read_csv(LOCAL_FILE)             # Wczytanie danych lokalnych do ramki danych pandas
    colab = pd.read_csv(COLAB_FILE)             # Wczytanie danych z Colab do ramki danych pandas

    # Obliczenie przyspieszeń
    local['Speedup'] = local['CPU'] / local['GPU_16'] # Obliczenie przyspieszenia lokalnego (Czas CPU / Czas GPU 16x16)
    colab['Speedup'] = colab['CPU'] / colab['GPU_16'] # Obliczenie przyspieszenia Colab (Czas CPU / Czas GPU 16x16)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))  # Utworzenie figury z 4 wykresami (siatka 2x2) i określenie rozmiaru
    fig.suptitle('Analiza Porównawcza: RTX 5080 vs Google Colab', fontsize=20, fontweight='bold') # Dodanie głównego tytułu

    # --- 1. CZASY WYKONANIA (LOG SCALE) ---
    ax1 = axes[0, 0]                            # Wybór pierwszego wykresu (lewy górny róg)
    ax1.plot(local['N'], local['CPU'], 'r--o', label='Lokalne CPU (ryzen 9 9950X3D)', alpha=0.8) # Rysowanie linii dla lokalnego CPU
    ax1.plot(colab['N'], colab['CPU'], 'm--s', label='Colab CPU', alpha=0.8)       # Rysowanie linii dla CPU Colab
    ax1.plot(local['N'], local['GPU_16'], 'g-D', label='Lokalne GPU (RTX 5080)', linewidth=2) # Rysowanie linii dla lokalnego GPU
    ax1.plot(colab['N'], colab['GPU_16'], 'b-v', label='Colab GPU', linewidth=2)   # Rysowanie linii dla GPU Colab

    ax1.set_yscale('log')                       # Ustawienie skali logarytmicznej na osi Y (dla dużych różnic wartości)
    ax1.set_title('Czasy wykonania (Skala Logarytmiczna)') # Ustawienie tytułu wykresu
    ax1.set_ylabel('Czas jednej iteracji [s]')   # Opis osi Y
    ax1.set_xlabel('Rozmiar siatki N')          # Opis osi X
    ax1.legend()                                # Wyświetlenie legendy
    ax1.grid(True, which="both", ls="-", alpha=0.2) # Włączenie siatki pomocniczej

    # --- 2. SPEEDUP (SKALA LOGARYTMICZNA - FIX NA "ZERO") ---
    ax2 = axes[0, 1]                            # Wybór drugiego wykresu (prawy górny róg)
    ax2.plot(local['N'], local['Speedup'], 'g-o', label='Speedup: RTX 5080', linewidth=3) # Rysowanie linii przyspieszenia RTX
    ax2.plot(colab['N'], colab['Speedup'], 'b-v', label='Speedup: Colab', alpha=0.5)      # Rysowanie linii przyspieszenia Colab

    ax2.set_yscale('log') # KLUCZOWE: skala logarytmiczna pozwala zobaczyć 500 obok 34 milionów # Ustawienie skali logarytmicznej
    ax2.set_title('Przyspieszenie (Speedup) - Skala Logarytmiczna') # Tytuł wykresu
    ax2.set_ylabel('Krotność przyspieszenia (x-faster)') # Opis osi Y
    ax2.set_xlabel('Rozmiar siatki N')          # Opis osi X
    ax2.legend()                                # Legenda
    ax2.grid(True, which="both", ls="--", alpha=0.5) # Siatka

    # --- 3. PORÓWNANIE CPU: TWÓJ PC VS COLAB ---
    ax3 = axes[1, 0]                            # Wybór trzeciego wykresu (lewy dolny róg)
    cpu_ratio = local['CPU'] / colab['CPU']     # Obliczenie stosunku czasu twojego CPU do CPU Colab
    ax3.bar(local['N'].astype(str), cpu_ratio, color='orange', alpha=0.6, label='Ratio (Moje CPU / Colab CPU)') # Rysowanie wykresu słupkowego
    ax3.axhline(1, color='black', linestyle='-', linewidth=1) # Rysowanie poziomej linii odniesienia na poziomie 1

    ax3.set_title('Wydajność CPU: Moje PC vs Google Colab') # Tytuł wykresu
    ax3.set_ylabel('Ratio (<1 = Moje PC szybsze)') # Opis osi Y
    ax3.set_xlabel('Rozmiar siatki N')          # Opis osi X
    ax3.legend()                                # Legenda
    ax3.grid(axis='y', ls='--', alpha=0.7)      # Siatka pozioma

    # --- 4. WRAŻLIWOŚĆ NA ROZMIAR BLOKU (RTX 5080) ---
    ax4 = axes[1, 1]                            # Wybór czwartego wykresu (prawy dolny róg)
    ax4.plot(local['N'], local['GPU_8'], 'o--', label='Blok 8x8', alpha=0.7)  # Linia dla bloku 8x8
    ax4.plot(local['N'], local['GPU_16'], 's-', label='Blok 16x16', linewidth=2) # Linia dla bloku 16x16
    ax4.plot(local['N'], local['GPU_32'], 'D--', label='Blok 32x32', alpha=0.7) # Linia dla bloku 32x32

    ax4.set_yscale('log')                       # Skala logarytmiczna
    ax4.set_title('Optymalizacja RTX 5080: Rozmiar bloku (Log Scale)') # Tytuł wykresu
    ax4.set_ylabel('Czas [s]')                  # Opis osi Y
    ax4.set_xlabel('Rozmiar siatki N')          # Opis osi X
    ax4.legend()                                # Legenda
    ax4.grid(True, which="both", ls="--", alpha=0.5) # Siatka

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])   # Automatyczne dopasowanie marginesów, zostawiając miejsce na tytuł główny
    plt.savefig('full_report_rtx5080.png')      # Zapisanie całego wykresu do pliku PNG
    print("Raport zapisany jako: full_report_rtx5080.png") # Wypisanie potwierdzenia w konsoli
    plt.show()                                  # Wyświetlenie okna z wykresem

if __name__ == "__main__":                      # Sprawdzenie czy skrypt jest uruchamiany bezpośrednio
    plot_benchmarks()                           # Wywołanie funkcji rysującej