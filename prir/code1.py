import threading
import requests
import time
import json
import os
import re
import matplotlib.pyplot as plt

URLS = [
    ('File1.json', 'https://realpython.com/python-thread-lock/'),
    ('File2.json', 'https://wolnelektury.pl/media/book/txt/pan-tadeusz.txt'),
    ('File3.json', 'https://wolnelektury.pl/media/book/txt/potop-tom-pierwszy.txt')
]
KEYWORDS = ['python', 'lock', 'tadeusz', 'potop', 'thread']

def download_file(filename, url):
    print(f"Pobieranie {url} ...")
    try:
        response = requests.get(url, timeout=15)
        content = response.text
        data = {
            "url": url,
            "status_code": response.status_code,
            "length": len(content),
            "text": content
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"{filename} zapisany ({len(content)} znaków).")
    except Exception as e:
        print(f"Błąd pobierania {url}: {e}")


def measure_no_threads():
    start = time.time()
    for filename, url in URLS:
        download_file(filename, url)
    end = time.time()
    return end - start


def measure_with_threads():
    start = time.time()
    threads = []
    for filename, url in URLS:
        t = threading.Thread(target=download_file, args=(filename, url))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    end = time.time()
    return end - start


def analyze_files():
    results = []
    for filename, url in URLS:
        if not os.path.exists(filename):
            print(f"Plik {filename} nie istnieje, pomijam.")
            continue
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            text = data.get("text", "")
            text_plain = re.sub(r"<[^>]+>", " ", text)
            char_count = len(text_plain)
            words = re.findall(r"\w+", text_plain.lower())
            word_count = len(words)
            keyword_counts = {kw: words.count(kw.lower()) for kw in KEYWORDS}
            results.append({
                "file": filename,
                "url": url,
                "chars": char_count,
                "words": word_count,
                **keyword_counts
            })
    return results


def plot_analysis(results):
    files = [r["file"] for r in results]
    word_counts = [r["words"] for r in results]
    char_counts = [r["chars"] for r in results]

    plt.figure(figsize=(8,4))
    plt.bar(files, char_counts, color='orange', label='znaki')
    plt.bar(files, word_counts, color='green', alpha=0.6, label='słowa')
    plt.title('Liczba znaków i słów w plikach')
    plt.ylabel('Liczba')
    plt.legend()
    plt.tight_layout()
    plt.show()

    for kw in KEYWORDS:
        plt.figure(figsize=(6,3))
        plt.bar(files, [r[kw] for r in results], color='skyblue')
        plt.title(f"Wystąpienia słowa '{kw}'")
        plt.ylabel('Liczba wystąpień')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    print("\n=== Pobieranie stron bez wielowątkowości ===")
    time_normal = measure_no_threads()

    print("\n=== Pobieranie stron z wielowątkowością ===")
    time_threaded = measure_with_threads()

    print("\n=== Porównanie czasu ===")
    print(f"Czas bez wielowątkowości: {time_normal:.2f} s")
    print(f"Czas z wielowątkowością: {time_threaded:.2f} s")

    plt.bar(['Bez wątków', 'Z wątkami'], [time_normal, time_threaded], color=['gray', 'green'])
    plt.title('Porównanie czasu pobierania')
    plt.ylabel('Czas [s]')
    plt.show()

    print("\n=== Analiza pobranych plików ===")
    results = analyze_files()
    for r in results:
        print(r)

    plot_analysis(results)

