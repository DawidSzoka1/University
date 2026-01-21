import pandas as pd             # Import biblioteki pandas do analizy danych (jako pd)
import matplotlib.pyplot as plt # Import biblioteki matplotlib do wykresów (jako plt)
import numpy as np              # Import biblioteki numpy do obliczeń numerycznych (jako np)
import os                       # Import modułu os do operacji na systemie plików
from PIL import Image           # Import klasy Image z biblioteki PIL (Pillow) do obsługi obrazów
import math                     # Import modułu math do funkcji matematycznych (sin, cos, pi)

# --- KONFIGURACJA ---
OUTPUT_DIR = 'output_images_julia' # Definicja nazwy katalogu wyjściowego
if not os.path.exists(OUTPUT_DIR): # Sprawdzenie, czy katalog nie istnieje
    os.makedirs(OUTPUT_DIR)        # Utworzenie katalogu, jeśli go nie ma

# 1. Wczytanie danych
try:
    df = pd.read_csv('results.csv') # Próba wczytania pliku 'results.csv' do ramki danych pandas
    df['pixels_millions'] = df['pixels'] / 1_000_000 # Dodanie kolumny z liczbą pikseli w milionach
except FileNotFoundError:           # Obsługa wyjątku, gdy plik nie istnieje
    print("Błąd: Brak pliku results.csv!") # Wypisanie komunikatu o błędzie
    exit()                          # Zakończenie działania skryptu

# --- CZĘŚĆ A: KOLORY (BIAŁE TŁO - STYL "INK ON PAPER") ---
print("Generowanie obrazów (Styl: Białe tło)...") # Informacja dla użytkownika

def create_white_bg_palette(): # Definicja funkcji tworzącej paletę kolorów
    palette = []               # Inicjalizacja pustej listy na paletę
    for i in range(256):       # Pętla po 256 poziomach szarości (indeksy iteracji)
        # t od 0 do 1
        t = i / 255.0          # Normalizacja indeksu do zakresu 0.0 - 1.0

        # Jeśli i jest bliskie 0 (tło, szybka ucieczka punktu), chcemy BIEL.
        # Im wyższe i (głębiej we fraktalu), tym ciemniejszy kolor.

        if i < 5:              # Jeśli indeks jest bardzo mały (tło)
            # Absolutne tło = Białe
            palette.extend([255, 255, 255]) # Dodaj biały kolor (RGB) do palety
            continue           # Przejdź do następnej iteracji pętli

        # Wzór: Zaczynamy od bieli, przechodzimy w niebieski/fiolet, kończymy na czerni
        # Odwracamy logikę: 1.0 to biały, 0.0 to czarny

        # Funkcja cosinusowa przesunięta tak, by startowała od jasnego
        # R: Szybko spada do zera (robi się ciemno-niebieski)
        r = int(255 * (0.5 + 0.5 * math.cos(2 * math.pi * t * 1.0))) # Obliczenie składowej czerwonej
        # G: Też spada, dając odcienie fioletu
        g = int(255 * (0.5 + 0.5 * math.cos(2 * math.pi * t * 1.5))) # Obliczenie składowej zielonej
        # B: Zostaje wysokie najdłużej (dlatego dominanta niebieska)
        b = int(255 * (0.5 + 0.5 * math.sin(2 * math.pi * t + 1.0))) + 50 # Obliczenie składowej niebieskiej

        # Rozjaśnianie dla niskich indeksów, żeby przejście z bieli było płynne
        if i < 50:             # Dla pierwszych 50 kolorów (przejście z tła)
            factor = (50 - i) / 50.0 # Obliczenie współczynnika mieszania
            r = int(r + (255 - r) * factor) # Rozjaśnienie R w stronę bieli
            g = int(g + (255 - g) * factor) # Rozjaśnienie G w stronę bieli
            b = int(b + (255 - b) * factor) # Rozjaśnienie B w stronę bieli

        palette.extend([min(255, r), min(255, g), min(255, b)]) # Dodanie obliczonego koloru RGB do palety (z ograniczeniem do 255)
    return palette             # Zwrócenie gotowej listy palety

fractal_palette = create_white_bg_palette() # Utworzenie palety i przypisanie do zmiennej

# Szukamy plików .bin
for filename in os.listdir('.'): # Pętla po plikach w bieżącym katalogu
    if filename.endswith(".bin") and filename.startswith("img_"): # Filtrowanie plików: muszą być .bin i zaczynać się od img_
        try:
            parts = filename.replace('.bin', '').split('_') # Usunięcie rozszerzenia i podział nazwy po znaku '_'
            res_part = parts[-1]        # Pobranie ostatniej części nazwy (zawierającej rozdzielczość)
            w, h = map(int, res_part.split('x')) # Rozdzielenie szerokości i wysokości, konwersja na int

            with open(filename, 'rb') as f: # Otwarcie pliku binarnego do odczytu
                raw_data = np.frombuffer(f.read(), dtype=np.uint8) # Wczytanie danych do tablicy numpy typu uint8

            if raw_data.size != w * h: continue # Jeśli rozmiar danych nie pasuje do rozdzielczości, pomiń plik

            img_array = raw_data.reshape((h, w)) # Przekształcenie płaskiej tablicy w macierz 2D (wysokość x szerokość)
            img = Image.fromarray(img_array, mode='L') # Utworzenie obrazu PIL w trybie skali szarości ('L')
            img = img.convert("P")      # Konwersja obrazu do trybu paletowego ('P')

            # Aplikacja palety z białym tłem
            img.putpalette(fractal_palette) # Przypisanie przygotowanej wcześniej palety kolorów do obrazu

            output_name = f"{OUTPUT_DIR}/{filename.replace('.bin', '.png')}" # Utworzenie nazwy pliku wyjściowego (PNG w katalogu output)
            img.save(output_name)       # Zapisanie obrazu na dysk
            print(f"Wygenerowano obraz: {output_name}") # Potwierdzenie wygenerowania obrazu

        except Exception:               # Przechwycenie ewentualnych błędów
            pass                        # Ignorowanie błędów i przejście dalej

# --- CZĘŚĆ B: WYKRESY ---
print("Rysowanie wykresów...")          # Informacja o rozpoczęciu rysowania

plt.style.use('default')                # Ustawienie domyślnego stylu wykresów matplotlib
plt.rcParams.update({                   # Aktualizacja parametrów konfiguracyjnych wykresów
    'font.size': 11,                    # Rozmiar czcionki
    'axes.grid': True,                  # Włączenie siatki
    'grid.alpha': 0.5,                  # Przezroczystość siatki
    'grid.linestyle': '--',             # Styl linii siatki (przerywana)
    'figure.facecolor': 'white',        # Kolor tła figury (biały)
    'axes.facecolor': 'white'           # Kolor tła osi (biały)
})

cpu_data = df[df['device'] == 'CPU']    # Filtrowanie danych: tylko wiersze gdzie device to CPU
gpu_data_julia = df[(df['device'] == 'GPU') & (df['type'] == 'Julia')] # Filtrowanie: GPU i typ fraktala Julia

# 1. Wykres Liniowy CPU
if not cpu_data.empty:                  # Jeśli są dane dla CPU
    plt.figure(figsize=(10, 6))         # Utworzenie nowej figury o wymiarach 10x6 cali
    plt.plot(cpu_data['pixels_millions'], cpu_data['timeMs'], # Rysowanie wykresu liniowego (X: Mpx, Y: czas)
             marker='o', color='#e74c3c', label='CPU Time')   # Stylizacja: kółka, czerwony kolor, etykieta
    plt.title('Czas CPU (im niżej tym lepiej)') # Tytuł wykresu
    plt.xlabel('Mpx')                   # Etykieta osi X
    plt.ylabel('ms')                    # Etykieta osi Y
    plt.legend()                        # Wyświetlenie legendy
    plt.tight_layout()                  # Automatyczne dopasowanie marginesów
    plt.savefig(f'{OUTPUT_DIR}/wykres_cpu_linear.png') # Zapis wykresu do pliku
    plt.close()                         # Zamknięcie figury (zwolnienie pamięci)

# 2. Wykres Liniowy GPU
if not gpu_data_julia.empty:            # Jeśli są dane dla GPU (Julia)
    plt.figure(figsize=(10, 6))         # Utworzenie nowej figury
    plt.plot(gpu_data_julia['pixels_millions'], gpu_data_julia['timeMs'], # Wykres liniowy danych GPU
             marker='s', color='#2980b9', label='GPU Julia')  # Stylizacja: kwadraty, niebieski kolor
    plt.title('Czas GPU (im niżej tym lepiej)') # Tytuł
    plt.xlabel('Mpx')                   # Oś X
    plt.ylabel('ms')                    # Oś Y
    plt.legend()                        # Legenda
    plt.tight_layout()                  # Marginesy
    plt.savefig(f'{OUTPUT_DIR}/wykres_julia_gpu_linear.png') # Zapis
    plt.close()                         # Zamknięcie

# 3. Wykres Porównawczy (Log)
common = pd.merge(cpu_data, gpu_data_julia, on=['width', 'height'], suffixes=('_cpu', '_gpu')) # Łączenie tabel CPU i GPU po rozdzielczości

if not common.empty:                    # Jeśli istnieją wspólne dane do porównania
    plt.figure(figsize=(12, 6))         # Nowa figura
    x = np.arange(len(common))          # Tablica indeksów dla osi X
    width = 0.35                        # Szerokość słupka
    plt.bar(x - width/2, common['timeMs_cpu'], width, label='CPU', color='#e74c3c', edgecolor='black') # Słupki CPU (przesunięte w lewo)
    plt.bar(x + width/2, common['timeMs_gpu'], width, label='GPU', color='#2980b9', edgecolor='black') # Słupki GPU (przesunięte w prawo)
    plt.yscale('log')                   # Ustawienie skali logarytmicznej na osi Y
    plt.xticks(x, [f"{w}x{h}" for w, h in zip(common['width'], common['height'])]) # Podpisy osi X (rozdzielczości)
    plt.ylabel('Czas (ms) - LOG')       # Opis osi Y
    plt.title('Porównanie czasu (Logarytmicznie)') # Tytuł
    plt.legend()                        # Legenda
    plt.tight_layout()                  # Marginesy
    plt.savefig(f'{OUTPUT_DIR}/wykres_porownanie_julia.png') # Zapis
    plt.close()                         # Zamknięcie

    # --- NOWOŚĆ: 4. WYKRES PRZYSPIESZENIA (SPEEDUP FACTOR) ---
    print("Generowanie wykresu przyspieszenia...") # Informacja

    # Obliczamy ile razy GPU jest szybsze (CPU time / GPU time)
    common['speedup'] = common['timeMs_cpu'] / common['timeMs_gpu'] # Dodanie kolumny z krotnością przyspieszenia

    plt.figure(figsize=(12, 7))         # Nowa figura
    bars = plt.bar(x, common['speedup'], color='#27ae60', edgecolor='black', width=0.6) # Wykres słupkowy przyspieszenia (zielony)

    plt.xticks(x, [f"{w}x{h}" for w, h in zip(common['width'], common['height'])], rotation=0) # Oś X z rozdzielczościami
    plt.ylabel('Krotność przyspieszenia (x razy)') # Oś Y
    plt.title('Ile razy GPU jest szybsze od CPU?', fontsize=14, fontweight='bold') # Tytuł

    # Dodajemy liczby nad słupkami (np. "150x")
    for bar in bars:                    # Pętla po każdym słupku
        height = bar.get_height()       # Pobranie wysokości słupka (wartości)
        plt.text(bar.get_x() + bar.get_width()/2., height, # Wstawienie tekstu nad słupkiem
                 f'{int(height)}x',     # Formatowanie tekstu (liczba całkowita + "x")
                 ha='center', va='bottom', fontsize=12, fontweight='bold') # Pozycjonowanie i styl tekstu

    plt.grid(axis='y', linestyle='--', alpha=0.7) # Dodanie poziomej siatki
    plt.tight_layout()                  # Dopasowanie układu
    plt.savefig(f'{OUTPUT_DIR}/wykres_speedup.png') # Zapis
    plt.close()                         # Zamknięcie

print(f"Gotowe! Sprawdź plik '{OUTPUT_DIR}/wykres_speedup.png' - tam zobaczysz ile razy GPU wygrywa.") # Komunikat końcowy