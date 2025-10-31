import pandas as pd
import time
import matplotlib.pyplot as plt
from pathlib import Path

from metodaNewtona import calculate as newton
from metodaBisekcji import calculate as bisekcji
from metodaDwudzielna import calculate as dwudzielna
from metodaFibonaciego import calculate as fibonnaci
from metodaSiecznych import calculate as siecznych
from myFunction import funkcja, poch_funkcja, poch2_funkcja, poch3_funkcja

# --- Ustawienia ---
a, b = 0.6, 5.8
methods = [newton, fibonnaci, bisekcji, dwudzielna, siecznych]
methods_name = ["Newton", "Fibonaci", "Bisekcja", "Dwudzielna", "Siecznych"]

epsilons = [1e-1, 1e-3, 1e-5, 1e-7, 1e-9]
iterations_list = [10, 100, 1000, 10000, 100000]
results = []

# --- Eksperyment A: różne e, iteration=100000 ---
for e in epsilons:
    for method, name in zip(methods, methods_name):
        start = time.time()
        try:
            res = method(a=a, b=b, e=e, iteration=100000, maks=False,
                         function=funkcja, poch_f=poch_funkcja,
                         poch2=poch2_funkcja, poch3=poch3_funkcja)
            elapsed = time.time() - start

            if isinstance(res, (list, tuple)):
                x_opt, f_opt, iter_used = (res + (None,))[:3]
            else:
                x_opt, f_opt, iter_used = res, None, None

            results.append({
                "Eksperyment": "A_epsilon",
                "Metoda": name,
                "Epsilon": e,
                "Iteracje_max": 100000,
                "Iteracje_wyk": iter_used,
                "x_opt": x_opt,
                "f(x_opt)": f_opt,
                "Czas_s": elapsed,
            })
        except Exception as ex:
            results.append({
                "Eksperyment": "A_epsilon",
                "Metoda": name,
                "Epsilon": e,
                "Iteracje_max": 100000,
                "Iteracje_wyk": None,
                "x_opt": None,
                "f(x_opt)": None,
                "Czas_s": None,
                "Błąd": str(ex),
            })

# --- Eksperyment B: różne iteration, e=1e-7 ---
for iteration in iterations_list:
    for method, name in zip(methods, methods_name):
        start = time.time()
        try:
            res = method(a=a, b=b, e=1e-7, iteration=iteration, maks=False,
                         function=funkcja, poch_f=poch_funkcja,
                         poch2=poch2_funkcja, poch3=poch3_funkcja)
            elapsed = time.time() - start

            if isinstance(res, (list, tuple)):
                x_opt, f_opt, iter_used = (res + (None,))[:3]
            else:
                x_opt, f_opt, iter_used = res, None, None

            results.append({
                "Eksperyment": "B_iteracje",
                "Metoda": name,
                "Epsilon": 1e-7,
                "Iteracje_max": iteration,
                "Iteracje_wyk": iter_used,
                "x_opt": x_opt,
                "f(x_opt)": f_opt,
                "Czas_s": elapsed,
            })
        except Exception as ex:
            results.append({
                "Eksperyment": "B_iteracje",
                "Metoda": name,
                "Epsilon": 1e-7,
                "Iteracje_max": iteration,
                "Iteracje_wyk": None,
                "x_opt": None,
                "f(x_opt)": None,
                "Czas_s": None,
                "Błąd": str(ex),
            })

# --- Zapis danych ---
df = pd.DataFrame(results)
Path("plots").mkdir(exist_ok=True)
df.to_csv("plots/wyniki_dual_v2.csv", index=False)
print("✅ Zapisano wyniki do plots/wyniki_dual_v2.csv")

# --- Wykresy ---
def plot_logx(df_subset, x, y, title, xlabel, ylabel, filename):
    plt.figure(figsize=(7, 5))
    for name, group in df_subset.groupby("Metoda"):
        plt.plot(group[x], group[y], marker="o", label=name)
    plt.xscale("log")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/{filename}.png")

# === Eksperyment A ===
expA = df[df["Eksperyment"] == "A_epsilon"]
plot_logx(expA, "Epsilon", "Iteracje_wyk",
          "Liczba faktycznych iteracji vs dokładność ε (100000 iteracji)",
          "Dokładność ε (log)", "Faktyczne iteracje", "iteracje_vs_eps")

plot_logx(expA, "Epsilon", "Czas_s",
          "Czas działania vs dokładność ε (100000 iteracji)",
          "Dokładność ε (log)", "Czas [s]", "czas_vs_eps")

plot_logx(expA, "Epsilon", "f(x_opt)",
          "Wartość funkcji vs dokładność ε (100000 iteracji)",
          "Dokładność ε (log)", "f(x_opt)", "fopt_vs_eps")

# === Eksperyment B ===
expB = df[df["Eksperyment"] == "B_iteracje"]
plot_logx(expB, "Iteracje_max", "Iteracje_wyk",
          "Faktyczne iteracje vs maksymalna liczba iteracji (ε = 1e-7)",
          "Maksymalna liczba iteracji (log)", "Faktyczne iteracje", "iter_wyk_vs_itermax")

plot_logx(expB, "Iteracje_max", "Czas_s",
          "Czas działania vs maksymalna liczba iteracji (ε = 1e-7)",
          "Maksymalna liczba iteracji (log)", "Czas [s]", "czas_vs_itermax")

plot_logx(expB, "Iteracje_max", "x_opt",
          "x_opt vs maksymalna liczba iteracji (ε = 1e-7)",
          "Maksymalna liczba iteracji (log)", "x_opt", "xopt_vs_itermax")

print("✅ Wygenerowano 6 wykresów w folderze 'plots/'")
