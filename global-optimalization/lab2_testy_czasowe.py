import csv
import random
import statistics
import time

# funkcje z lab2.py bez kodu na poziomie modułu i bez print(points) (debug)
src = open('/home/dawid/Projects/University/global-optimalization/lab2.py').read()
exec(src.split('zmienne_3d')[0].replace('    print(points)\n', ''))

NS = [3, 5, 7, 9]
DS = [1, 2, 3, 6]
CZESCI = [16, 4, 2, 1]  # N = 2^m / czesc
R = 5

with open('lab2_wyniki/czasy.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['n', 'd', 'N', 't_sredni_s', 't_odch_std_s', 't_min_s', 't_max_s'])
    for d in DS:
        m = find_m(2 * 10 ** d + 1)
        for N in [2 ** m // c for c in CZESCI]:
            for n in NS:
                data = [{'a': -1, 'b': 1, 'd': d} for _ in range(n)]
                ts = []
                for r in range(R):
                    random.seed(r)
                    t = time.perf_counter()
                    calc(data, N, func_nd)
                    ts.append(time.perf_counter() - t)
                w.writerow([n, d, N, f'{statistics.mean(ts):.6f}', f'{statistics.stdev(ts):.6f}',
                            f'{min(ts):.6f}', f'{max(ts):.6f}'])
                fh.flush()
                print(n, d, N, round(statistics.mean(ts), 4), flush=True)
