import subprocess
import os
import shutil

def compile_latex_to_pdf(tex_file_path, output_dir=None, compiler='pdflatex'):
    """
    Kompiluje plik .tex do .pdf przy użyciu systemowego kompilatora.
    """
    if not os.path.exists(tex_file_path):
        print(f"Błąd: Plik {tex_file_path} nie istnieje.")
        return False

    # Pobierz nazwę pliku i katalog roboczy
    tex_file_path = os.path.abspath(tex_file_path)
    work_dir = os.path.dirname(tex_file_path)
    file_name = os.path.basename(tex_file_path)
    job_name = os.path.splitext(file_name)[0]

    if output_dir:
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = work_dir

    print(f"Rozpoczynanie kompilacji pliku: {file_name} za pomocą {compiler}...")

    # Komenda kompilacji
    # -interaction=nonstopmode: nie zatrzymuj się na błędach
    # -output-directory: gdzie mają trafić pliki wynikowe
    cmd = [
        compiler,
        "-interaction=nonstopmode",
        f"-output-directory={output_dir}",
        tex_file_path
    ]

    try:
        # LaTeX często wymaga 2 przebiegów (dla spisu treści/linków)
        for i in range(2):
            print(f"Przebieg {i+1}...")
            result = subprocess.run(
                cmd,
                cwd=work_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            if result.returncode != 0:
                print(f"Błąd podczas przebiegu {i+1}!")
                # Wyświetl końcówkę logu w razie błędu
                print(result.stdout[-500:])
                return False

        print(f"Sukces! PDF został wygenerowany w: {output_dir}/{job_name}.pdf")
        return True

    except FileNotFoundError:
        print(f"Błąd: Nie znaleziono kompilatora '{compiler}'. Upewnij się, że masz zainstalowany TeX Live/MiKTeX.")
        return False

# --- PRZYKŁAD UŻYCIA ---
if __name__ == "__main__":
    # Ścieżka do Twojego głównego pliku Beamera
    path_to_main = "main.tex"

    # Możesz użyć 'lualatex' jeśli 'pdflatex' wyrzuca błędy pamięci przy TikZ
    compile_latex_to_pdf(path_to_main, compiler='pdflatex')