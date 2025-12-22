import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
import subprocess
import os
import glob
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class ImageProcessorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("CUDA Benchmark & Processing")
        self.geometry("1200x800")

        # Kontener na zakładki
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=20, pady=20)

        self.tab_single = self.tabview.add("Pojedyncze Zdjęcie")
        self.tab_batch = self.tabview.add("Testowanie Folderu")

        self.setup_single_tab()
        self.setup_batch_tab()

    # ==========================================
    # ZAKŁADKA 1: POJEDYNCZE ZDJĘCIE
    # ==========================================
    def setup_single_tab(self):
        frame = self.tab_single
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=3)

        # Panel boczny
        panel = ctk.CTkFrame(frame, width=250)
        panel.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.single_path = None

        ctk.CTkButton(panel, text="Wybierz Plik", command=self.select_single_file).pack(pady=10)

        self.algo_var = ctk.IntVar(value=1)
        ctk.CTkLabel(panel, text="Algorytm:", font=ctk.CTkFont(weight="bold")).pack(pady=(20,5))
        ctk.CTkRadioButton(panel, text="Sepia", variable=self.algo_var, value=1).pack(anchor="w", padx=20, pady=5)
        ctk.CTkRadioButton(panel, text="Gaussian Blur", variable=self.algo_var, value=2).pack(anchor="w", padx=20, pady=5)
        ctk.CTkRadioButton(panel, text="Sobel Edge", variable=self.algo_var, value=3).pack(anchor="w", padx=20, pady=5)

        ctk.CTkButton(panel, text="Przetwarzaj", fg_color="green", command=self.process_single).pack(pady=30)
        self.status_single = ctk.CTkLabel(panel, text="Oczekiwanie...", text_color="gray", wraplength=200)
        self.status_single.pack(side="bottom", pady=10)

        # Podgląd
        preview_frame = ctk.CTkFrame(frame)
        preview_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        self.lbl_orig = ctk.CTkLabel(preview_frame, text="[Brak Oryginału]")
        self.lbl_orig.pack(side="left", expand=True, fill="both", padx=5)
        self.lbl_res = ctk.CTkLabel(preview_frame, text="[Brak Wyniku]")
        self.lbl_res.pack(side="right", expand=True, fill="both", padx=5)

    def select_single_file(self):
        filetypes = [
            ("Obrazy", "*.jpg *.jpeg *.png *.bmp *.JPG *.JPEG *.PNG *.BMP"),
            ("Wszystkie pliki", "*.*")
        ]
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            self.single_path = path
            self.show_img(path, self.lbl_orig)
            self.status_single.configure(text=f"Wybrano: {os.path.basename(path)}")

    def process_single(self):
        if not self.single_path:
            self.status_single.configure(text="BŁĄD: Wybierz plik!", text_color="red")
            return

        algo = self.algo_var.get()
        executable = "./image_proc"

        if not os.path.exists(executable):
            self.status_single.configure(text="Brak pliku image_proc!", text_color="red")
            return

        # Uruchamianie (Mode 1: Zapisz wynik.png)
        cmd = [executable, "1", self.single_path, str(algo)]

        try:
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0:
                if os.path.exists("wynik.png"):
                    self.show_img("wynik.png", self.lbl_res)
                    output_line = res.stdout.strip()
                    if "SUCCESS" in output_line:
                        time_ms = output_line.split()[1]
                        self.status_single.configure(text=f"Sukces! Czas: {time_ms} ms", text_color="green")
                    else:
                        self.status_single.configure(text="Zakończono (brak danych o czasie)", text_color="yellow")
                else:
                    self.status_single.configure(text="Błąd: Brak pliku wynik.png", text_color="red")
            else:
                print(f"C++ Error: {res.stderr}")
                self.status_single.configure(text="Błąd wykonania C++", text_color="red")
        except Exception as e:
            print(e)
            self.status_single.configure(text=f"Wyjątek Pythona: {e}", text_color="red")

    def show_img(self, path, label):
        try:
            img = Image.open(path)
            img.thumbnail((500, 500))
            cimg = ctk.CTkImage(img, size=img.size)
            label.configure(image=cimg, text="")
        except Exception as e:
            print(f"Błąd wyświetlania: {e}")

    # ==========================================
    # ZAKŁADKA 2: TESTOWANIE FOLDERU (BENCHMARK + ZAPIS)
    # ==========================================
    def setup_batch_tab(self):
        frame = self.tab_batch

        # Sterowanie na górze
        ctrl_frame = ctk.CTkFrame(frame)
        ctrl_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkButton(ctrl_frame, text="Wybierz Folder", command=self.select_folder).pack(side="left", padx=10)
        self.folder_lbl = ctk.CTkLabel(ctrl_frame, text="[Folder nie wybrany]")
        self.folder_lbl.pack(side="left", padx=10)

        # Checkbox do zapisu
        self.save_processed_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(ctrl_frame, text="Zapisuj zdjęcia", variable=self.save_processed_var).pack(side="left", padx=20)

        ctk.CTkButton(ctrl_frame, text="URUCHOM TESTY", fg_color="red", command=self.run_benchmark).pack(side="right", padx=10)

        # Wyświetlanie logów postępu
        self.progress_lbl = ctk.CTkLabel(ctrl_frame, text="")
        self.progress_lbl.pack(side="right", padx=20)

        # Wykresy
        self.plot_frame = ctk.CTkFrame(frame)
        self.plot_frame.pack(fill="both", expand=True, padx=10, pady=10)

    def select_folder(self):
        path = filedialog.askdirectory()
        if path:
            self.batch_folder = path
            self.folder_lbl.configure(text=path)

    def run_benchmark(self):
        if not hasattr(self, 'batch_folder'):
            self.progress_lbl.configure(text="Najpierw wybierz folder!", text_color="red")
            return

        algo = self.algo_var.get()
        executable = "./image_proc"

        if not os.path.exists(executable):
            self.progress_lbl.configure(text="Brak pliku image_proc!", text_color="red")
            return

        # Znajdowanie plików
        files = glob.glob(os.path.join(self.batch_folder, "*"))
        valid_ext = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [f for f in files if os.path.splitext(f)[1].lower() in valid_ext]

        if not image_files:
            self.progress_lbl.configure(text="Folder jest pusty lub brak obrazów!", text_color="red")
            return

        # Przygotowanie folderu wyjściowego (jeśli zaznaczono opcję zapisu)
        should_save = self.save_processed_var.get()
        output_dir = os.path.join(self.batch_folder, "processed_output")
        if should_save:
            os.makedirs(output_dir, exist_ok=True)

        self.progress_lbl.configure(text="Rozpoczynanie testów...", text_color="white")
        self.update()

        # Dane do wykresów
        data_res = [] # (pixels, cpu, gpu)
        data_blocks = {} # pixels -> {threads: time}

        # Iteracja po plikach
        for idx, fpath in enumerate(image_files):
            filename = os.path.basename(fpath)
            self.progress_lbl.configure(text=f"Przetwarzanie {idx+1}/{len(image_files)}: {filename}")
            self.update()

            # 1. URUCHOMIENIE BENCHMARKU (Mode 2)
            cmd_bench = [executable, "2", fpath, str(algo)]
            try:
                res = subprocess.run(cmd_bench, capture_output=True, text=True)

                # Parsowanie wyników benchmarku
                cpu_t, gpu_t, pixels = 0, 0, 0
                current_blocks = []

                for line in res.stdout.splitlines():
                    parts = line.split()
                    if not parts: continue

                    if parts[0] == "RES_DATA":
                        try:
                            w, h = int(parts[1]), int(parts[2])
                            pixels = w * h
                            cpu_t = float(parts[3])
                            gpu_t = float(parts[4])
                            data_res.append((pixels, cpu_t, gpu_t))
                        except ValueError: pass

                    if parts[0] == "BLOCK_DATA":
                        try:
                            th = int(parts[1])
                            ms = float(parts[2])
                            current_blocks.append((th, ms))
                        except ValueError: pass

                if pixels > 0 and current_blocks:
                    data_blocks[pixels] = current_blocks

                # 2. ZAPIS PLIKU (Jeśli opcja włączona)
                # Uruchamiamy Mode 1 (który generuje "wynik.png"), a potem przenosimy plik
                if should_save:
                    cmd_save = [executable, "1", fpath, str(algo)]
                    subprocess.run(cmd_save, capture_output=True) # Ignorujemy output, interesuje nas plik

                    if os.path.exists("wynik.png"):
                        # Nowa nazwa: processed_nazwapliku.png
                        new_name = f"processed_{os.path.splitext(filename)[0]}.png"
                        dest_path = os.path.join(output_dir, new_name)

                        # Przenoszenie (move/rename)
                        if os.path.exists(dest_path): os.remove(dest_path)
                        os.rename("wynik.png", dest_path)

            except Exception as e:
                print(f"Błąd przy pliku {fpath}: {e}")

        # Rysowanie wykresów
        if not data_res:
            self.progress_lbl.configure(text="Błąd: Nie zebrano żadnych danych!", text_color="red")
        else:
            self.plot_results(data_res, data_blocks)
            msg = "Zakończono pomyślnie."
            if should_save: msg += f" Zapisano w: processed_output"
            self.progress_lbl.configure(text=msg, text_color="green")

    def plot_results(self, data_res, data_blocks):
        data_res.sort(key=lambda x: x[0])

        pixels = [x[0] for x in data_res]
        cpu_times = [x[1] for x in data_res]
        gpu_times = [x[2] for x in data_res]

        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Wykres 1
        ax1.plot(pixels, cpu_times, 'r-o', label='CPU', markersize=6)
        ax1.plot(pixels, gpu_times, 'b-s', label='GPU', markersize=6)
        ax1.set_title("Wydajność vs Rozdzielczość")
        ax1.set_xlabel("Liczba pikseli")
        ax1.set_ylabel("Czas (ms)")
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, which="both", ls="--")

        # Wykres 2
        if data_blocks:
            max_pix = max(data_blocks.keys())
            blocks = data_blocks[max_pix]
            blocks.sort(key=lambda x: x[0])

            b_th = [str(x[0]) for x in blocks]
            b_ms = [x[1] for x in blocks]

            bars = ax2.bar(b_th, b_ms, color='orange')
            ax2.set_title(f"Czas vs Wątki w bloku (dla największego obrazu)")
            ax2.set_xlabel("Wątki w bloku")
            ax2.set_ylabel("Czas GPU (ms)")

            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                         f'{height:.1f}', ha='center', va='bottom')

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

if __name__ == "__main__":
    app = ImageProcessorApp()
    app.mainloop()