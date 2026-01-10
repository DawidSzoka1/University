import pandas as pd
import numpy as np
import gradient
import newton
import gauss_seidla_dok
import hook_jeeves

X_ANALYTICAL = (1 + 145 ** (1/2)) / 36
Y_ANALYTICAL = 9 * X_ANALYTICAL ** 2

# --- KONFIGURACJA STARTOWA ---
X0_START = 2
Y0_START = 2


# --- FUNKCJA LICZĄCA BŁĄD WZGLĘDNY [%] ---
def calculate_relative_error(analytical, measured):
    """Liczy błąd względny w procentach."""
    if analytical == 0:
        return 0.0  # Zabezpieczenie przed dzieleniem przez zero
    return abs((analytical - measured) / analytical) * 100.0


# Wrapery metod - TERAZ PRZEKAZUJĄ PARAMETR check_e
def run_gradient(x0, y0, eps, max_iter, check_e=True):
    # Przekazujemy check_e do funkcji w gradient.py
    return gradient.calculate(x0, y0, epsilon=eps, iterations=max_iter, check_e=check_e)


def run_newton(x0, y0, eps, max_iter, check_e=True):
    # Przekazujemy check_e do funkcji w newton.py
    return newton.calculate(x0, y0, e=eps, iterations=max_iter, check_e=check_e)


def run_gaussa(x0, y0, eps, max_iter, check_e=True):
    # Przekazujemy check_e do funkcji w gauss_seidla_dok.py
    return gauss_seidla_dok.calculate(x0, y0, epsilon=eps, iterations=max_iter, check_e=check_e)


def run_hook(x0, y0, eps, max_iter, check_e=True):
    return hook_jeeves.calculate(x0, y0, epsilon=eps, iterations=max_iter, check_e=check_e, beta=0.5, e_start=0.5)


METHODS = {
    "Metoda Najszybszego Spadku": run_gradient,
    "Metoda Newtona": run_newton,
    "Metoda Gaussa-seidla": run_gaussa,
    "Metoda Hooke’a - Jeevesa": run_hook,
}


# --- FUNKCJE GENERUJĄCE DANE W FORMACIE MULTIINDEX ---

def create_accuracy_dataframe(methods_dict, epsilons):
    """
    Tabela 1: Różne dokładności (epsilon).
    Tutaj check_e musi być True, aby algorytm zatrzymał się po osiągnięciu dokładności.
    """
    rows_data = []
    indices = []

    for method_name, run_func in methods_dict.items():
        row_x = []
        row_y = []
        row_iter = []

        for eps in epsilons:
            # Ustawiamy duży limit iteracji, żeby algorytm nie przerwał zbyt wcześnie z powodu licznika
            safe_max_iter = 10000

            # WŁĄCZAMY sprawdzanie epsilona (check_e=True)
            xk, yk, iters, _ = run_func(X0_START, Y0_START, eps, safe_max_iter, check_e=True)

            row_x.append(f"{xk:.9f}")
            row_y.append(f"{yk:.9f}")
            row_iter.append(str(iters))

        rows_data.append(row_x)
        rows_data.append(row_y)
        rows_data.append(row_iter)

        indices.append((method_name, 'x'))
        indices.append((method_name, 'y'))
        indices.append((method_name, 'iteracje'))

    multi_idx = pd.MultiIndex.from_tuples(indices, names=['Metoda', 'Dane'])
    df = pd.DataFrame(rows_data, columns=epsilons, index=multi_idx)
    df.columns.name = 'Epsilon (ε)'
    return df


def create_iteration_dataframe(methods_dict, iterations_list):
    """
    Tabela 2: Różne liczby iteracji.
    Tutaj check_e musi być False, aby algorytm wykonał DOKŁADNIE zadaną liczbę kroków
    niezależnie od tego, czy już znalazł wynik.
    """
    rows_data = []
    indices = []

    # Epsilon jest tutaj nieistotny, bo check_e=False, ale musimy coś podać
    dummy_epsilon = 0.01

    for method_name, run_func in methods_dict.items():
        row_x = []
        row_y = []

        for max_iter in iterations_list:
            # WYŁĄCZAMY sprawdzanie epsilona (check_e=False)
            xk, yk, iters, _ = run_func(X0_START, Y0_START, dummy_epsilon, max_iter, check_e=False)

            row_x.append(f"{xk:.12f}")
            row_y.append(f"{yk:.12f}")

        rows_data.append(row_x)
        rows_data.append(row_y)

        indices.append((method_name, 'x'))
        indices.append((method_name, 'y'))

    multi_idx = pd.MultiIndex.from_tuples(indices, names=['Metoda', 'Dane'])
    df = pd.DataFrame(rows_data, columns=iterations_list, index=multi_idx)
    df.columns.name = 'Liczba iteracji'
    return df


def create_accuracy_error_dataframe(methods_dict, epsilons):
    """Tabela błędów dla różnych Epsilonów"""
    rows_data = []
    indices = []

    for method_name, run_func in methods_dict.items():
        row_err_x = []
        row_err_y = []

        for eps in epsilons:
            # check_e=True bo badamy wpływ epsilona
            xk, yk, _, _ = run_func(X0_START, Y0_START, eps, 10000, check_e=True)

            err_x = calculate_relative_error(X_ANALYTICAL, xk)
            err_y = calculate_relative_error(Y_ANALYTICAL, yk)

            row_err_x.append(f"{err_x:.9f} %")
            row_err_y.append(f"{err_y:.9f} %")

        rows_data.append(row_err_x)
        rows_data.append(row_err_y)
        indices.append((method_name, 'δx [%]'))
        indices.append((method_name, 'δy [%]'))

    multi_idx = pd.MultiIndex.from_tuples(indices, names=['Metoda', 'Błąd względny'])
    df = pd.DataFrame(rows_data, columns=epsilons, index=multi_idx)
    df.columns.name = 'Epsilon (ε)'
    return df


def create_iteration_error_dataframe(methods_dict, iterations_list):
    """Tabela błędów dla różnej liczby iteracji"""
    rows_data = []
    indices = []
    dummy_epsilon = 0.01

    for method_name, run_func in methods_dict.items():
        row_err_x = []
        row_err_y = []

        for max_iter in iterations_list:
            # check_e=False bo badamy konkretną liczbę iteracji
            xk, yk, _, _ = run_func(X0_START, Y0_START, dummy_epsilon, max_iter, check_e=False)

            err_x = calculate_relative_error(X_ANALYTICAL, xk)
            err_y = calculate_relative_error(Y_ANALYTICAL, yk)

            row_err_x.append(f"{err_x:.12f} %")
            row_err_y.append(f"{err_y:.12f} %")

        rows_data.append(row_err_x)
        rows_data.append(row_err_y)
        indices.append((method_name, 'δx [%]'))
        indices.append((method_name, 'δy [%]'))

    multi_idx = pd.MultiIndex.from_tuples(indices, names=['Metoda', 'Błąd względny'])
    df = pd.DataFrame(rows_data, columns=iterations_list, index=multi_idx)
    df.columns.name = 'Liczba iteracji'
    return df


if __name__ == "__main__":
    # Parametry zgodne ze sprawozdaniem
    test_epsilons = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
    test_iterations = [1, 2, 3, 5, 6, 9]

    # Generowanie
    print("Generowanie tabel wyników...")
    df_acc = create_accuracy_dataframe(METHODS, test_epsilons)
    df_iter = create_iteration_dataframe(METHODS, test_iterations)

    print("Generowanie tabel błędów...")
    df_err_acc = create_accuracy_error_dataframe(METHODS, test_epsilons)
    df_err_iter = create_iteration_error_dataframe(METHODS, test_iterations)

    # Wyświetlenie próbki błędów
    print("\n--- Przykładowa tabela błędów (Epsilon) ---")
    print(df_err_acc)

    # --- ZAPIS DO PLIKU ---
    output_file = 'Kompletne_Tabele_Sprawozdanie.xlsx'
    try:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Arkusz 1: Wyniki dla Epsilona (Pkt 4)
            df_acc.to_excel(writer, sheet_name='Wyniki_Epsilon')
            # Arkusz 2: Błędy dla Epsilona (Pkt 6 cz.1)
            df_err_acc.to_excel(writer, sheet_name='Błędy_Epsilon')

            # Arkusz 3: Wyniki dla Iteracji (Pkt 5)
            df_iter.to_excel(writer, sheet_name='Wyniki_Iteracje')
            # Arkusz 4: Błędy dla Iteracji (Pkt 6 cz.2)
            df_err_iter.to_excel(writer, sheet_name='Błędy_Iteracje')

        print(f"\nSUKCES! Wygenerowano 4 arkusze w pliku: {output_file}")
        print("Możesz kopiować tabele prosto do Worda.")
    except Exception as e:
        print(f"\nBłąd zapisu: {e}")
