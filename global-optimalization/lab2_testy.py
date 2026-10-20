import math, random, statistics, time, csv
from multiprocessing import Pool

# funkcje z lab2.py bez kodu na poziomie modułu i bez print(points) (debug)
src = open('/home/dawid/Projects/University/global-optimalization/lab2.py').read()
exec(src.split('zmienne_3d')[0].replace('    print(points)\n', ''))

NS = [3, 5, 7, 9]
DS = [1, 2, 3, 6]
CZESCI = [16, 4, 2, 1]  # N = 2^m / czesc, czyli 1/16, 1/4, 1/2 i cała siatka
R = 10


def populacje(d):
    m = find_m(2 * 10 ** d + 1)
    return [2 ** m // c for c in CZESCI]


def job(a):
    n, d, N, seed = a
    random.seed(seed)
    data = [{'a': -1, 'b': 1, 'd': d} for _ in range(n)]
    p = calc(data, N, func_nd)
    return n, d, N, func_nd(*p), math.sqrt(sum(x * x for x in p))


if __name__ == '__main__':
    jobs = [(n, d, N, r * 7919 + n * 31 + d * 17 + N) for n in NS for d in DS for N in populacje(d) for r in range(R)]
    jobs.sort(key=lambda j: -j[2] * j[0])
    res = []
    with Pool(12, maxtasksperchild=1) as pool:
        for i, r in enumerate(pool.imap_unordered(job, jobs), 1):
            res.append(r)
            if i % 20 == 0:
                print(f'{i}/{len(jobs)}', flush=True)
    agg = {}
    for n, d, N, f, dist in res:
        agg.setdefault((n, d, N), []).append((f, dist))
    with open('lab2_wyniki/wyniki.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['n', 'd', 'N', 'czesc_siatki', 'f_srednia', 'f_odch_std', 'f_min', 'f_max', 'odl_od_0_srednia'])
        for (n, d, N), v in sorted(agg.items()):
            fs = [x[0] for x in v]
            m = find_m(2 * 10 ** d + 1)
            w.writerow([n, d, N, f'1/{2 ** m // N}', round(statistics.mean(fs), 4), round(statistics.stdev(fs), 4),
                        round(min(fs), 4), round(max(fs), 4), round(statistics.mean(x[1] for x in v), 4)])
