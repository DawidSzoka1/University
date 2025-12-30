import xml.etree.ElementTree as ET
import random
import os
import subprocess


class XmlToLatex:
    def __init__(self):
        self.latex_content = []
        # Definicje różnych stylów dla tekstu
        self.text_styles = [
            r"{\textit{#1}}",  # Kursywa
            r"{\textbf{#1}}",  # Pogrubienie
            r"{\textsc{#1}}",  # Kapitaliki
            r"{\textsf{#1}}",  # Bezszeryfowy
            r"{#1}",  # Zwykły
        ]
        self.price_style = r"{#1} \euro"
        self.kcal_style = r"{#1} kcal"
        # Definicje różnych stylów dla liczb
        self.num_styles = [
            r"\textbf{\expandafter{\romannumeral #1\relax}}",  # Rzymskie pogrubione
            r"{#1\textsuperscript{o}}",  # Z indeksem górnym
            r"\fbox{#1}",  # W ramce
            r"\textit{#1}",  # Kursywa liczbowa
            r"\underline{#1}",  # Podkreślenie
        ]

    def escape(self, text):
        if text is None: return ""
        chars = {
            '&': r'\&', '$': r'\$', '%': r'\%', '#': r'\#',
            '_': r'\_', '{': r'\{', '}': r'\}'
        }
        for char, escaped in chars.items():
            text = text.replace(char, escaped)
        return text.strip()

    def clean_tag_for_cmd(self, tag):
        return tag.replace("_", "").replace("-", "").upper()

    def is_number(self, s):
        try:
            float(s)
            return True
        except (ValueError, TypeError):
            return False

    def generate_preamble(self, groups):
        all_tags = {}
        for elements in groups.values():
            for el in elements:
                for child in el:
                    tag_name = child.tag.upper()
                    if tag_name not in all_tags:
                        all_tags[tag_name] = self.is_number(child.text)

        preamble = [
            r"\documentclass{article}",
            r"\usepackage{amsmath, amsthm, amssymb}",
            r"\usepackage[utf8]{inputenc}",
            r"\usepackage{eurosym}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage{booktabs, graphicx, xcolor, tcolorbox}",
            r"\usepackage[margin=1in]{geometry}",
        ]

        for tag, is_num in all_tags.items():
            cmd_name = self.clean_tag_for_cmd(tag)
            style = random.choice(self.num_styles if is_num else self.text_styles)
            if tag == "PRICE":
                style = self.price_style
            elif tag == "CALORIES":
                style = self.kcal_style
            preamble.append(f"\\newcommand{{\\{cmd_name}}}[1]{{{style}}}")

        preamble.append("\n\\begin{document}")
        return "\n".join(preamble)

    def render_as_table(self, title, elements, fields):
        latex = f"\\section{{{self.escape(title)} -- Styl Tabeli}}\n"
        col_def = "c|" + "r" * (len(fields))
        latex += r"\begin{tabular}{" + col_def + "}\n"
        latex += "Nr & " + " & ".join([f.capitalize() for f in fields]) + r"\\\hline" + "\n"
        for i, el in enumerate(elements, 1):
            row = [str(i)]
            for f in fields:
                val = self.escape(el.findtext(f))
                cmd = self.clean_tag_for_cmd(f)
                row.append(f"\\{cmd}{{{val}}}")
            latex += " & ".join(row) + r"\\" + "\n"
        latex += r"\end{tabular}" + "\n\n"
        return latex

    def render_as_list(self, title, elements, fields):
        latex = f"\\section{{{self.escape(title)} -- Styl Listy}}\n"
        latex += r"\begin{enumerate}" + "\n"
        for el in elements:
            items = []
            for f in fields:
                val = self.escape(el.findtext(f))
                cmd = self.clean_tag_for_cmd(f)
                items.append(f"\\{cmd}{{{val}}}")
            latex += f"\\item {', '.join(items)}\n"
        latex += r"\end{enumerate}" + "\n\n"
        return latex

    def render_as_cards(self, title, elements, fields):
        latex = f"\\section{{{self.escape(title)} -- Styl Kart}}\n"
        for el in elements:
            latex += r"\begin{tcolorbox}[colback=white, colframe=black!15, title=Element]"
            for f in fields:
                val = self.escape(el.findtext(f))
                cmd = self.clean_tag_for_cmd(f)
                latex += f"\\textbf{{{self.escape(f.capitalize())}}}: \\{cmd}{{{val}}} \\\\ \n"
            latex += r"\end{tcolorbox}"
        return latex

    def compile_to_pdf(self, tex_file):
        print(f"Rozpoczynanie kompilacji pliku: {tex_file}")
        try:
            for i in range(2):
                result = subprocess.run(
                    ['pdflatex', '-interaction=nonstopmode', tex_file],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

            if result.returncode == 0:
                print("--- Sukces! Plik PDF został utworzony. ---")
            else:
                print("--- Błąd kompilacji LaTeX! ---")
                print("Sprawdź plik .log dla szczegółów.")
        except FileNotFoundError:
            print(
                "Błąd: Nie znaleziono polecenia 'pdflatex'. Upewnij się, że masz zainstalowany LaTeX (np. MiKTeX, TeX Live).")

    def process(self, input_file):
        tree = ET.parse(input_file)
        root = tree.getroot()

        groups = {}
        for child in root:
            if child.tag not in groups: groups[child.tag] = []
            groups[child.tag].append(child)

        full_latex = [self.generate_preamble(groups)]

        available_styles = [self.render_as_table, self.render_as_list, self.render_as_cards]

        for i, (tag_name, elements) in enumerate(groups.items()):
            fields = []
            for e in elements:
                for sub in e:
                    if sub.tag not in fields: fields.append(sub.tag)

            style_func = available_styles[i % len(available_styles)]
            full_latex.append(style_func(tag_name, elements, fields))

        full_latex.append(r"\end{document}")
        tex_name = input_file.replace(".xml", ".tex")
        with open(tex_name, "w", encoding="utf-8") as f:
            f.write("\n".join(full_latex))

        self.compile_to_pdf(tex_name)

if __name__ == "__main__":
    p = input("Podaj nazwę pliku XML: ")
    if os.path.exists(p):
        XmlToLatex().process(p)
