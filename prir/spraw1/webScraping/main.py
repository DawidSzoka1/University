import re

import requests
import json
import time
import os
import matplotlib.pyplot as plt
import threading


URLS = [
    "https://www.python.org",
    "https://www.wikipedia.org",
    "https://www.bbc.com",
    "https://edition.cnn.com",
    "https://www.openai.com",
    "https://www.nasa.gov",
]

OUTPUT_DIR = "downloaded_pages"
KEYWORDS = ["python", "ai", "data", "science", "news", "space"]
os.makedirs(OUTPUT_DIR, exist_ok=True)

def fetch_page(url, filename):
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


def analyze_json(data, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    stats = []
    with open(path, "w", encoding="utf-8") as f:
        text = data.get("content").lower()
        word_count = len(re.findall(r"\w+", text))
        char_count = len(text)
        keyword_count = {kw: text.count(kw.lower()) for kw in KEYWORDS}
        stats.append({
            "url": data.get("url", ""),
            "chars": char_count,
            "words": word_count,
            **keyword_count
        })
        json.dump(data, f, indent=2, ensure_ascii=False)


def download_sequential(urls):
    start = time.time()
    for i,url in enumerate(urls):
        fetch_page(url, f"file_{i+1}_seq.json")
    return time.time() - start


def download_parallel(urls):
    start = time.time()
    threads = []
    for i,url in enumerate(urls):
        t = threading.Thread(target=fetch_page, args=(url, f"file_{i+1}_parallel.json"))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()
    end = time.time()
    return end - start



def analyze_json():
    results = []
    files = []
    for f in os.listdir('.'):
        if not f.endswith('.json'):
            continue
        if f[:6] not in files:
            files.append(f[:6])
    for file in files:
        print(f"Analyzing {file}")
        with open(f'{file}_seq.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            text = data.get("text").lower()
            word_count = len(re.findall(r"\w+", text))
        char_count = len(text)
        keyword_count = {kw: text.count(kw.lower()) for kw in KEYWORDS}
        results.append({
            "url": data.get("url", ""),
            "filename": file,
            "chars": char_count,
            "words": word_count,
            **keyword_count
        })
    return results


def plot_analysis(results):
    files = [r['filename'] for r in results]
    word_count = [r['words'] for r in results]
    chars = [r['chars'] for r in results]
    plt.figure(figsize=(8,4))
    plt.bar(files, chars, color='orange', label='znaki')
    plt.bar(files, word_count, color='green', alpha=0.6, label='słowa')
    plt.title('Liczba znaków i słów w plikach')
    plt.ylabel('Liczba')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"chars_words_count.png")
    plt.show()
    for kw in KEYWORDS:
        plt.figure(figsize=(12, 6))
        plt.bar(files, [r[kw] for r in results], color='skyblue')
        plt.title(f"Wystapienia slowa '{kw}'")
        plt.ylabel("liczba wystapien")
        plt.tight_layout()
        plt.savefig(f"files_{kw}.png")
        plt.show()

if __name__ == "__main__":
    time_seq = download_sequential(URLS)
    time_parallel = download_parallel(URLS)

    plt.bar(['Bez wątków', 'Z wątkami'],[time_seq, time_parallel], color=['gray', 'green'])
    plt.title('Prównanie czasu pobierania')
    plt.ylabel('Czas [s]')
    plt.savefig('time_compare.png')
    plt.show()

    print(f"Czas bez wielowątkowości: {time_seq:.2f} s")
    print(f"Czas z wielowątkowością: {time_parallel:.2f} s")

    results = analyze_json()
    plot_analysis(results)

