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



if __name__ == "__main__":
    time_seq = download_sequential(URLS)
    time_parallel = download_parallel(URLS)

    print(f"Czas bez wielowątkowości: {time_seq:.2f} s")
    print(f"Czas z wielowątkowością: {time_parallel:.2f} s")

