import math  # Importuje moduł math, używany do obliczeń matematycznych, np. silnia (factorial)

# Funkcja oblicza różnice w przód (forward differences) dla danego ciągu wartości y
def compute_forward_differences(y_values):
    n = len(y_values)  # Liczba punktów
    differences = [y_values.copy()]  # Tworzy listę różnic, zaczynając od oryginalnych wartości y
    for k in range(1, n):  # Dla każdej kolejnej różnicy (pierwsza, druga, itd.)
        prev = differences[-1]  # Bierzemy ostatni poziom różnic
        # Obliczamy różnice między kolejnymi elementami poprzedniego poziomu
        current = [prev[i+1] - prev[i] for i in range(len(prev) - 1)]
        differences.append(current)  # Dodajemy nowy poziom różnic do listy
    return differences  # Zwracamy całą tabelę różnic w przód

# Funkcja interpoluje wartość w punkcie x za pomocą interpolacji Newtona (w przód)
def newton_interpolation(x_values, y_values, x):
    h = x_values[1] - x_values[0]  # Zakładamy równe odstępy między punktami (węzłami)
    differences = compute_forward_differences(y_values)  # Obliczamy różnice w przód

    n = len(x_values)  # Liczba punktów (węzłów)
    result = y_values[0]  # Wynik interpolacji zaczynamy od f(x₀)
    product = 1  # Będzie przechowywał iloczyn (x - x₀)(x - x₁)... do danego rzędu

    for k in range(1, n):  # Iterujemy po kolejnych wyrazach wielomianu Newtona
        product *= (x - x_values[k-1])  # Aktualizujemy iloczyn (x - x₀)...(x - x_{k-1})
        coefficient = differences[k][0] / (math.factorial(k) * (h ** k))  # Liczymy współczynnik z różnicy dzielonej
        result += coefficient * product  # Dodajemy wyraz do wyniku

    return result  # Zwracamy wynik interpolacji

# Przykładowe dane wejściowe:
x_nodes = [-4, -2, 0, 2, 4]  # Węzły x (równomiernie rozmieszczone)
y_nodes = [802, 78, 2, -50, -318]  # Wartości funkcji w tych węzłach
point = 3  # Punkt, w którym interpolujemy wartość

# Wywołujemy funkcję interpolującą i wypisujemy wynik
output = newton_interpolation(x_nodes, y_nodes, point)
print(f"W({point}) = {output}")  # Wyświetla wynik interpolacji dla zadanego punktu
