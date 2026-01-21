import customtkinter as ctk # Import biblioteki CustomTkinter do tworzenia nowoczesnego GUI
from tkinter import filedialog # Import modułu do wyświetlania okien wyboru plików
from PIL import Image # Import klasy Image z biblioteki Pillow do obsługi obrazów
import subprocess # Import modułu do uruchamiania zewnętrznych procesów (naszego programu C++)
import os # Import modułu do operacji na systemie plików
import glob # Import modułu do wyszukiwania plików za pomocą wzorców (np. *.jpg)
import matplotlib.pyplot as plt # Import biblioteki matplotlib do rysowania wykresów
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg # Import backendu do osadzania wykresów w Tkinter

ctk.set_appearance_mode("Dark") # Ustawienie ciemnego motywu aplikacji
ctk.set_default_color_theme("blue") # Ustawienie domyślnego koloru akcentu na niebieski

class ImageProcessorApp(ctk.CTk): # Główna klasa aplikacji dziedzicząca po oknie ctk.CTk
    def __init__(self):
        super().__init__() # Wywołanie konstruktora klasy bazowej

        self.title("CUDA Benchmark & Processing") # Ustawienie tytułu okna
        self.geometry("1200x800") # Ustawienie początkowych wymiarów okna

        # Kontener na zakładki
        self.tabview = ctk.CTkTabview(self) # Utworzenie widgetu zakładek
        self.tabview.pack(fill="both", expand=True, padx=20, pady=20) # Umieszczenie zakładek w oknie z marginesami

        self.tab_single = self.tabview.add("Pojedyncze Zdjęcie") # Dodanie zakładki "Pojedyncze Zdjęcie"
        self.tab_batch = self.tabview.add("Testowanie Folderu") # Dodanie zakładki "Testowanie Folderu"

        self.setup_single_tab() # Konfiguracja zawartości pierwszej zakładki
        self.setup_batch_tab() # Konfiguracja zawartości drugiej zakładki

    # ==========================================
    # ZAKŁADKA 1: POJEDYNCZE ZDJĘCIE
    # ==========================================
    def setup_single_tab(self):
        frame = self.tab_single # Przypisanie ramki zakładki do zmiennej
        frame.columnconfigure(0, weight=1) # Konfiguracja wagi kolumny 0 (panel boczny)
        frame.columnconfigure(1, weight=3) # Konfiguracja wagi kolumny 1 (podgląd) - szersza

        # Panel boczny
        panel = ctk.CTkFrame(frame, width=250) # Utworzenie ramki panelu bocznego
        panel.grid(row=0, column=0, sticky="nsew", padx=10, pady=10) # Umieszczenie panelu w siatce

        self.single_path = None # Inicjalizacja zmiennej przechowującej ścieżkę pliku

        ctk.CTkButton(panel, text="Wybierz Plik", command=self.select_single_file).pack(pady=10) # Przycisk wyboru pliku

        self.algo_var = ctk.IntVar(value=1) # Zmienna całkowita przechowująca wybór algorytmu (domyślnie 1)
        ctk.CTkLabel(panel, text="Algorytm:", font=ctk.CTkFont(weight="bold")).pack(pady=(20,5)) # Etykieta sekcji
        ctk.CTkRadioButton(panel, text="Sepia", variable=self.algo_var, value=1).pack(anchor="w", padx=20, pady=5) # Opcja Sepia
        ctk.CTkRadioButton(panel, text="Gaussian Blur", variable=self.algo_var, value=2).pack(anchor="w", padx=20, pady=5) # Opcja Gaussian
        ctk.CTkRadioButton(panel, text="Sobel Edge", variable=self.algo_var, value=3).pack(anchor="w", padx=20, pady=5) # Opcja Sobel

        ctk.CTkButton(panel, text="Przetwarzaj", fg_color="green", command=self.process_single).pack(pady=30) # Przycisk uruchomienia
        self.status_single = ctk.CTkLabel(panel, text="Oczekiwanie...", text_color="gray", wraplength=200) # Etykieta statusu
        self.status_single.pack(side="bottom", pady=10) # Umieszczenie statusu na dole panelu

        # Podgląd
        preview_frame = ctk.CTkFrame(frame) # Ramka na podgląd obrazów
        preview_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10) # Umieszczenie ramki w siatce

        self.lbl_orig = ctk.CTkLabel(preview_frame, text="[Brak Oryginału]") # Etykieta obrazu oryginalnego
        self.lbl_orig.pack(side="left", expand=True, fill="both", padx=5) # Pozycjonowanie po lewej
        self.lbl_res = ctk.CTkLabel(preview_frame, text="[Brak Wyniku]") # Etykieta obrazu wynikowego
        self.lbl_res.pack(side="right", expand=True, fill="both", padx=5) # Pozycjonowanie po prawej

    def select_single_file(self):
        filetypes = [ # Definicja obsługiwanych typów plików
            ("Obrazy", "*.jpg *.jpeg *.png *.bmp *.JPG *.JPEG *.PNG *.BMP"),
            ("Wszystkie pliki", "*.*")
        ]
        path = filedialog.askopenfilename(filetypes=filetypes) # Otwarcie okna dialogowego wyboru pliku
        if path: # Jeśli wybrano plik
            self.single_path = path # Zapisanie ścieżki
            self.show_img(path, self.lbl_orig) # Wyświetlenie oryginału
            self.status_single.configure(text=f"Wybrano: {os.path.basename(path)}") # Aktualizacja statusu

    def process_single(self):
        if not self.single_path: # Jeśli nie wybrano pliku
            self.status_single.configure(text="BŁĄD: Wybierz plik!", text_color="red") # Komunikat błędu
            return # Przerwanie funkcji

        algo = self.algo_var.get() # Pobranie wybranego algorytmu
        executable = "./image_proc" # Ścieżka do pliku wykonywalnego C++

        if not os.path.exists(executable): # Sprawdzenie czy plik exe istnieje
            self.status_single.configure(text="Brak pliku image_proc!", text_color="red") # Błąd
            return

        # Uruchamianie (Mode 1: Zapisz wynik.png)
        cmd = [executable, "1", self.single_path, str(algo)] # Budowanie komendy: ./image_proc 1 plik algo

        try:
            res = subprocess.run(cmd, capture_output=True, text=True) # Uruchomienie procesu i przechwycenie wyjścia
            if res.returncode == 0: # Jeśli proces zakończył się sukcesem
                if os.path.exists("wynik.png"): # Sprawdzenie czy powstał plik wynikowy
                    self.show_img("wynik.png", self.lbl_res) # Wyświetlenie wyniku
                    output_line = res.stdout.strip() # Pobranie stdout i usunięcie białych znaków
                    if "SUCCESS" in output_line: # Sprawdzenie czy program zgłosił sukces
                        time_ms = output_line.split()[1] # Wyciągnięcie czasu z komunikatu
                        self.status_single.configure(text=f"Sukces! Czas: {time_ms} ms", text_color="green") # Wyświetlenie czasu
                    else:
                        self.status_single.configure(text="Zakończono (brak danych o czasie)", text_color="yellow")
                else:
                    self.status_single.configure(text="Błąd: Brak pliku wynik.png", text_color="red")
            else:
                print(f"C++ Error: {res.stderr}") # Wypisanie błędu C++ do konsoli
                self.status_single.configure(text="Błąd wykonania C++", text_color="red")
        except Exception as e: # Przechwycenie wyjątków Pythona
            print(e)
            self.status_single.configure(text=f"Wyjątek Pythona: {e}", text_color="red")

    def show_img(self, path, label): # Funkcja pomocnicza do wyświetlania obrazu w etykiecie
        try:
            img = Image.open(path) # Otwarcie obrazu przez PIL
            img.thumbnail((500, 500)) # Skalowanie obrazu (zachowując proporcje) do max 500x500
            cimg = ctk.CTkImage(img, size=img.size) # Konwersja na obraz CustomTkinter
            label.configure(image=cimg, text="") # Ustawienie obrazu w etykiecie i usunięcie tekstu
        except Exception as e:
            print(f"Błąd wyświetlania: {e}") # Logowanie błędu

    # ==========================================
    # ZAKŁADKA 2: TESTOWANIE FOLDERU (BENCHMARK + ZAPIS)
    # ==========================================
    def setup_batch_tab(self):
        frame = self.tab_batch # Przypisanie ramki

        # Sterowanie na górze
        ctrl_frame = ctk.CTkFrame(frame) # Ramka sterowania
        ctrl_frame.pack(fill="x", padx=10, pady=10) # Umieszczenie ramki

        ctk.CTkButton(ctrl_frame, text="Wybierz Folder", command=self.select_folder).pack(side="left", padx=10) # Przycisk wyboru folderu
        self.folder_lbl = ctk.CTkLabel(ctrl_frame, text="[Folder nie wybrany]") # Etykieta folderu
        self.folder_lbl.pack(side="left", padx=10) # Umieszczenie etykiety

        # Checkbox do zapisu
        self.save_processed_var = ctk.BooleanVar(value=True) # Zmienna boolean dla checkboxa
        ctk.CTkCheckBox(ctrl_frame, text="Zapisuj zdjęcia", variable=self.save_processed_var).pack(side="left", padx=20) # Checkbox

        ctk.CTkButton(ctrl_frame, text="URUCHOM TESTY", fg_color="red", command=self.run_benchmark).pack(side="right", padx=10) # Przycisk uruchomienia

        # Wyświetlanie logów postępu
        self.progress_lbl = ctk.CTkLabel(ctrl_frame, text="") # Etykieta postępu
        self.progress_lbl.pack(side="right", padx=20) # Umieszczenie etykiety

        # Wykresy
        self.plot_frame = ctk.CTkFrame(frame) # Ramka na wykresy
        self.plot_frame.pack(fill="both", expand=True, padx=10, pady=10) # Umieszczenie ramki

    def select_folder(self):
        path = filedialog.askdirectory() # Wybór katalogu
        if path:
            self.batch_folder = path # Zapisanie ścieżki
            self.folder_lbl.configure(text=path) # Aktualizacja etykiety

    def run_benchmark(self):
        if not hasattr(self, 'batch_folder'): # Sprawdzenie czy folder został wybrany
            self.progress_lbl.configure(text="Najpierw wybierz folder!", text_color="red")
            return

        algo = self.algo_var.get() # Pobranie algorytmu
        executable = "./image_proc" # Plik wykonywalny

        if not os.path.exists(executable): # Weryfikacja pliku exe
            self.progress_lbl.configure(text="Brak pliku image_proc!", text_color="red")
            return

        # Znajdowanie plików
        files = glob.glob(os.path.join(self.batch_folder, "*")) # Pobranie wszystkich plików z folderu
        valid_ext = {'.jpg', '.jpeg', '.png', '.bmp'} # Zbiór dozwolonych rozszerzeń
        image_files = [f for f in files if os.path.splitext(f)[1].lower() in valid_ext] # Filtrowanie listy plików

        if not image_files: # Jeśli brak obrazów
            self.progress_lbl.configure(text="Folder jest pusty lub brak obrazów!", text_color="red")
            return

        # Przygotowanie folderu wyjściowego (jeśli zaznaczono opcję zapisu)
        should_save = self.save_processed_var.get() # Odczyt checkboxa
        output_dir = os.path.join(self.batch_folder, "processed_output") # Ścieżka folderu wyjściowego
        if should_save:
            os.makedirs(output_dir, exist_ok=True) # Utworzenie folderu, jeśli nie istnieje

        self.progress_lbl.configure(text="Rozpoczynanie testów...", text_color="white") # Aktualizacja statusu
        self.update() # Odświeżenie GUI

        # Dane do wykresów
        data_res = [] # Lista na wyniki: (piksele, czas_cpu, czas_gpu)
        data_blocks = {} # Słownik na wyniki bloków: piksele -> {wątki: czas}

        # Iteracja po plikach
        for idx, fpath in enumerate(image_files): # Pętla po obrazach z licznikiem
            filename = os.path.basename(fpath) # Nazwa pliku
            self.progress_lbl.configure(text=f"Przetwarzanie {idx+1}/{len(image_files)}: {filename}") # Aktualizacja postępu
            self.update() # Odświeżenie GUI

            # 1. URUCHOMIENIE BENCHMARKU (Mode 2)
            cmd_bench = [executable, "2", fpath, str(algo)] # Komenda benchmarku
            try:
                res = subprocess.run(cmd_bench, capture_output=True, text=True) # Uruchomienie

                # Parsowanie wyników benchmarku
                cpu_t, gpu_t, pixels = 0, 0, 0
                current_blocks = []

                for line in res.stdout.splitlines(): # Iteracja po liniach wyjścia
                    parts = line.split() # Podział linii na słowa
                    if not parts: continue # Pominięcie pustych linii

                    if parts[0] == "RES_DATA": # Linia z wynikami ogólnymi
                        try:
                            w, h = int(parts[1]), int(parts[2]) # Odczyt wymiarów
                            pixels = w * h # Obliczenie pikseli
                            cpu_t = float(parts[3]) # Odczyt czasu CPU
                            gpu_t = float(parts[4]) # Odczyt czasu GPU
                            data_res.append((pixels, cpu_t, gpu_t)) # Dodanie do listy wyników
                        except ValueError: pass # Ignorowanie błędów parsowania

                    if parts[0] == "BLOCK_DATA": # Linia z wynikami bloków
                        try:
                            th = int(parts[1]) # Liczba wątków
                            ms = float(parts[2]) # Czas
                            current_blocks.append((th, ms)) # Dodanie do listy bloków
                        except ValueError: pass

                if pixels > 0 and current_blocks: # Jeśli dane są poprawne
                    data_blocks[pixels] = current_blocks # Zapisanie bloków dla danej rozdzielczości

                # 2. ZAPIS PLIKU (Jeśli opcja włączona)
                # Uruchamiamy Mode 1 (który generuje "wynik.png"), a potem przenosimy plik
                if should_save:
                    cmd_save = [executable, "1", fpath, str(algo)] # Komenda przetwarzania (tryb 1)
                    subprocess.run(cmd_save, capture_output=True) # Ignorujemy output, interesuje nas plik

                    if os.path.exists("wynik.png"): # Sprawdzenie pliku
                        # Nowa nazwa: processed_nazwapliku.png
                        new_name = f"processed_{os.path.splitext(filename)[0]}.png"
                        dest_path = os.path.join(output_dir, new_name) # Ścieżka docelowa

                        # Przenoszenie (move/rename)
                        if os.path.exists(dest_path): os.remove(dest_path) # Usunięcie starego pliku, jeśli istnieje
                        os.rename("wynik.png", dest_path) # Przeniesienie pliku

            except Exception as e:
                print(f"Błąd przy pliku {fpath}: {e}") # Logowanie błędu

        # Rysowanie wykresów
        if not data_res:
            self.progress_lbl.configure(text="Błąd: Nie zebrano żadnych danych!", text_color="red")
        else:
            self.plot_results(data_res, data_blocks) # Wywołanie funkcji rysującej
            msg = "Zakończono pomyślnie."
            if should_save: msg += f" Zapisano w: processed_output"
            self.progress_lbl.configure(text=msg, text_color="green")

    def plot_results(self, data_res, data_blocks):
        data_res.sort(key=lambda x: x[0]) # Sortowanie wyników po liczbie pikseli (rosnąco)

        pixels = [x[0] for x in data_res] # Lista pikseli (oś X)
        cpu_times = [x[1] for x in data_res] # Czasy CPU
        gpu_times = [x[2] for x in data_res] # Czasy GPU

        for widget in self.plot_frame.winfo_children(): # Usunięcie starych wykresów z ramki
            widget.destroy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6)) # Utworzenie figury z 2 wykresami

        # Wykres 1
        ax1.plot(pixels, cpu_times, 'r-o', label='CPU', markersize=6) # Seria CPU (czerwona)
        ax1.plot(pixels, gpu_times, 'b-s', label='GPU', markersize=6) # Seria GPU (niebieska)
        ax1.set_title("Wydajność vs Rozdzielczość") # Tytuł
        ax1.set_xlabel("Liczba pikseli") # Oś X
        ax1.set_ylabel("Czas (ms)") # Oś Y
        ax1.set_xscale('log') # Skala logarytmiczna X
        ax1.set_yscale('log') # Skala logarytmiczna Y
        ax1.legend() # Legenda
        ax1.grid(True, which="both", ls="--") # Siatka

        # Wykres 2
        if data_blocks: # Jeśli są dane o blokach
            max_pix = max(data_blocks.keys()) # Wybór danych dla największego obrazu
            blocks = data_blocks[max_pix] # Pobranie danych
            blocks.sort(key=lambda x: x[0]) # Sortowanie po liczbie wątków

            b_th = [str(x[0]) for x in blocks] # Etykiety osi X (liczba wątków)
            b_ms = [x[1] for x in blocks] # Wartości osi Y (czas)

            bars = ax2.bar(b_th, b_ms, color='orange') # Wykres słupkowy
            ax2.set_title(f"Czas vs Wątki w bloku (dla największego obrazu)") # Tytuł
            ax2.set_xlabel("Wątki w bloku") # Oś X
            ax2.set_ylabel("Czas GPU (ms)") # Oś Y

            for bar in bars: # Dodanie etykiet nad słupkami
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                         f'{height:.1f}', ha='center', va='bottom')

        plt.tight_layout() # Dopasowanie układu
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame) # Utworzenie płótna dla Tkinter
        canvas.draw() # Rysowanie
        canvas.get_tk_widget().pack(fill="both", expand=True) # Umieszczenie w oknie

if __name__ == "__main__":
    app = ImageProcessorApp() # Utworzenie instancji aplikacji
    app.mainloop() # Uruchomienie pętli głównej