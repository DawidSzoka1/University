import pandas as pd
import time
import matplotlib.pyplot as plt
from pathlib import Path

from metodaNewtona import calculate as newton
from metodaBisekcji import calculate as bisekcji
from metodaDwudzielna import calculate as dwudzielna
from metodaFibonaciego import calculate as fibonnaci
from metodaSiecznych import calculate as siecznych
from metodaZlotegoPodzialu import calculate as zlotego_podzialu
from myFunction import funkcja, poch_funkcja, poch2_funkcja, poch3_funkcja

# --- Ustawienia ---
a, b = 0.6, 5.8
methods = [newton, fibonnaci, bisekcji, dwudzielna, siecznych, zlotego_podzialu]
methods_name = ["Newton", "Fibonaci", "Bisekcja", "Dwudzielna", "Siecznych", "Złotego podziału"]

epsilons = [0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001]
iterations_list = [2, 5, 7, 10, 100, 1000]
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

from pathlib import Path

df = pd.read_csv("plots/wyniki_dual_v2.csv")

DECIMALS = 9

def format_fixed(value, decimals=DECIMALS):
    if pd.isna(value):
        return "NaN"
    return f"{value:.{decimals}f}"

def format_with_iter(value, iters):
    val_str = format_fixed(value)
    if pd.isna(iters):
        return f"{val_str} (-)"
    return f"{val_str} ({int(iters)})"

def eps_to_pow(eps):
    try:
        exp = int(f"{eps:.0e}".split("e")[1])
        return f"10^{exp}"
    except Exception:
        return str(eps)


expA = df[df["Eksperyment"] == "A_epsilon"].copy()
expA["Epsilon_label"] = expA["Epsilon"].apply(eps_to_pow)
expA["display_val"] = [
    format_with_iter(f, it) for f, it in zip(expA["x_opt"], expA["Iteracje_wyk"])
]
table_eps = expA.pivot(index="Epsilon_label", columns="Metoda", values="display_val")

expB = df[df["Eksperyment"] == "B_iteracje"].copy()
expB["display_val"] = [format_fixed(f) for f in expB["x_opt"]]
table_iter = expB.pivot(index="Iteracje_max", columns="Metoda", values="display_val")

# --- Błędy względne ---
x_true = 3.3
expA["blad_wzgledny"] = abs((x_true - expA["x_opt"] ) / x_true) * 100
expB["blad_wzgledny"] = abs((x_true - expB["x_opt"] ) / x_true) * 100


expA["blad_fmt"] = expA["blad_wzgledny"].apply(format_fixed)
expB["blad_fmt"] = expB["blad_wzgledny"].apply(format_fixed)

table_eps_err = expA.pivot(index="Epsilon_label", columns="Metoda", values="blad_fmt")
table_iter_err = expB.pivot(index="Iteracje_max", columns="Metoda", values="blad_fmt")

# --- Wyświetlenie w konsoli ---
print("\n=== Tabela 1: Wyniki dla różnych ε (wartość + iteracje) ===")
print(table_eps)
print("\n=== Tabela 2: Wyniki dla różnych liczby iteracji ===")
print(table_iter)
print("\n=== Tabela 3: Błędy względne dla różnych ε ===")
print(table_eps_err)
print("\n=== Tabela 4: Błędy względne dla różnych liczby iteracji ===")
print(table_iter_err)

# --- Zapis do Excela ---
Path("plots").mkdir(exist_ok=True)
with pd.ExcelWriter("plots/tabele_analiza.xlsx") as writer:
    table_eps.to_excel(writer, sheet_name="Wyniki_vs_epsilon")
    table_iter.to_excel(writer, sheet_name="Wyniki_vs_iteracje")
    table_eps_err.to_excel(writer, sheet_name="Bledy_vs_epsilon")
    table_iter_err.to_excel(writer, sheet_name="Bledy_vs_iteracje")

print("✅ Zapisano tabele z formatowaniem do plots/tabele_analiza.xlsx")


plt.figure(figsize=(8,6))
for name, group in expA.groupby("Metoda"):
    print(group)
    plt.plot(group["Epsilon"], group["blad_wzgledny"], marker="s", label=name)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Dokładność ε (log)")
plt.ylabel("Błąd względny [%] (log)")
plt.title("Porównanie błędów względnych dla różnych metod")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig("plots/bledy_vs_eps.png", dpi=300)

plt.figure(figsize=(8,6))
for name, group in expB.groupby("Metoda"):
    plt.plot(group["Iteracje_max"], group["blad_wzgledny"], marker="^", label=name)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Maksymalna liczba iteracji (log)")
plt.ylabel("Błąd względny [%] (log)")
plt.title("Porównanie błędów względnych w funkcji liczby iteracji")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig("plots/bledy_vs_iteracje.png", dpi=300)