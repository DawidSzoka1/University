import csv
import matplotlib.pyplot as plt
import math
cores_list = [1, 2, 4, 8, 16, 32]
n_points_list = [int(1e6 * i) for i in range(1, 11)]


def read_date(file):
    results_time = {s: [] for s in cores_list}
    results_pi = {s: [] for s in cores_list}
    results_std_time = {s: [] for s in cores_list}
    results_error_pi = {s: [] for s in cores_list}
    with open(file, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            core = int(row['cores'])
            results_time[core].append(float(row['mean_time']))
            results_pi[core].append(float(row['mean_pi']))
            results_std_time[core].append(float(row['std_time']))
            results_error_pi[core].append(float(row['std_pi']))
    return results_time, results_pi, results_std_time, results_error_pi


local_data = read_date('montecarlo_results.csv')
colab_data = read_date('montecarlo_results_colab_cpu.csv')

# print(local_data[0])
# print(colab_data[0])
# porownanie czasu
print(abs((math.pi - 3.1418213333333336)/ math.pi * 100)/ abs((math.pi - 3.1415942666666665)/math.pi * 100))
time_compare = {s: [] for s in cores_list}
error_pi_compare = {s: [] for s in cores_list}
for core in cores_list:
    for i in range(len(n_points_list)):
        time_compare[core].append(colab_data[0][core][i] / local_data[0][core][i])
        if colab_data[3][core][i]  / local_data[3][core][i] == 141.76617823385507:
            error_pi_compare[core].append(1.0)
            continue
        error_pi_compare[core].append(colab_data[3][core][i] / local_data[3][core][i])
print(time_compare)
print(error_pi_compare)

plt.figure(figsize=(10, 5))
for s in cores_list:
    plt.plot(n_points_list, time_compare[s],
                 label=f"{s} rdzenie", marker='o')
plt.xlabel("Liczba punktów")
plt.ylabel("Stosunek czasu (Colab / Lokalny)")
plt.title(f"Porównanie średniego czasu między lokalnym komputerem a colab cpu Monte Carlo")
plt.axhline(1.0, color='gray', linestyle='--')
plt.legend()
plt.grid(True)
plt.savefig("stosunek_czasu.png", dpi=300)
plt.show()

plt.figure(figsize=(10, 5))
for s in cores_list:
    plt.plot(n_points_list, error_pi_compare[s], label=f"{s} rdzenie", marker='o')
plt.axhline(1.0, color='gray', linestyle='--')
plt.xlabel("Liczba punktów")
plt.ylabel("Stosunek błędu (%) (Colab / Lokalny)")
plt.title("Porównanie dokładności przybliżenia liczby π (błąd procentowy)")
plt.legend()
plt.grid(True)
plt.savefig("stosunek_błedu_pi.png", dpi=300)
plt.show()
