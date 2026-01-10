import pandas as pd
import matplotlib.pyplot as plt
import os

# Wczytanie danych (upewnij się, że pliki są w tym samym folderze)
LOCAL_FILE = 'results.csv'
COLAB_FILE = 'results_colab.csv'

def plot_benchmarks():
    if not os.path.exists(LOCAL_FILE) or not os.path.exists(COLAB_FILE):
        print("BŁĄD: Brakuje plików results.csv lub results_colab.csv!")
        return

    local = pd.read_csv(LOCAL_FILE)
    colab = pd.read_csv(COLAB_FILE)

    # Obliczenie przyspieszeń
    local['Speedup'] = local['CPU'] / local['GPU_16']
    colab['Speedup'] = colab['CPU'] / colab['GPU_16']

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Analiza Porównawcza: RTX 5080 vs Google Colab', fontsize=20, fontweight='bold')

    # --- 1. CZASY WYKONANIA (LOG SCALE) ---
    ax1 = axes[0, 0]
    ax1.plot(local['N'], local['CPU'], 'r--o', label='Lokalne CPU (ryzen 9 9950X3D)', alpha=0.8)
    ax1.plot(colab['N'], colab['CPU'], 'm--s', label='Colab CPU', alpha=0.8)
    ax1.plot(local['N'], local['GPU_16'], 'g-D', label='Lokalne GPU (RTX 5080)', linewidth=2)
    ax1.plot(colab['N'], colab['GPU_16'], 'b-v', label='Colab GPU', linewidth=2)

    ax1.set_yscale('log')
    ax1.set_title('Czasy wykonania (Skala Logarytmiczna)')
    ax1.set_ylabel('Czas jednej iteracji [s]')
    ax1.set_xlabel('Rozmiar siatki N')
    ax1.legend()
    ax1.grid(True, which="both", ls="-", alpha=0.2)

    # --- 2. SPEEDUP (SKALA LOGARYTMICZNA - FIX NA "ZERO") ---
    ax2 = axes[0, 1]
    ax2.plot(local['N'], local['Speedup'], 'g-o', label='Speedup: RTX 5080', linewidth=3)
    ax2.plot(colab['N'], colab['Speedup'], 'b-v', label='Speedup: Colab', alpha=0.5)

    ax2.set_yscale('log') # KLUCZOWE: skala logarytmiczna pozwala zobaczyć 500 obok 34 milionów
    ax2.set_title('Przyspieszenie (Speedup) - Skala Logarytmiczna')
    ax2.set_ylabel('Krotność przyspieszenia (x-faster)')
    ax2.set_xlabel('Rozmiar siatki N')
    ax2.legend()
    ax2.grid(True, which="both", ls="--", alpha=0.5)

    # --- 3. PORÓWNANIE CPU: TWÓJ PC VS COLAB ---
    ax3 = axes[1, 0]
    cpu_ratio = local['CPU'] / colab['CPU']
    ax3.bar(local['N'].astype(str), cpu_ratio, color='orange', alpha=0.6, label='Ratio (Moje CPU / Colab CPU)')
    ax3.axhline(1, color='black', linestyle='-', linewidth=1)

    ax3.set_title('Wydajność CPU: Moje PC vs Google Colab')
    ax3.set_ylabel('Ratio (<1 = Moje PC szybsze)')
    ax3.set_xlabel('Rozmiar siatki N')
    ax3.legend()
    ax3.grid(axis='y', ls='--', alpha=0.7)

    # --- 4. WRAŻLIWOŚĆ NA ROZMIAR BLOKU (RTX 5080) ---
    ax4 = axes[1, 1]
    ax4.plot(local['N'], local['GPU_8'], 'o--', label='Blok 8x8', alpha=0.7)
    ax4.plot(local['N'], local['GPU_16'], 's-', label='Blok 16x16', linewidth=2)
    ax4.plot(local['N'], local['GPU_32'], 'D--', label='Blok 32x32', alpha=0.7)

    ax4.set_yscale('log')
    ax4.set_title('Optymalizacja RTX 5080: Rozmiar bloku (Log Scale)')
    ax4.set_ylabel('Czas [s]')
    ax4.set_xlabel('Rozmiar siatki N')
    ax4.legend()
    ax4.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('full_report_rtx5080.png')
    print("Raport zapisany jako: full_report_rtx5080.png")
    plt.show()

if __name__ == "__main__":
    plot_benchmarks()