# Asystent AI „Bielik” — uruchomienie i test

Asystent kursu to polski model językowy **Bielik** (SpeakLeash) uruchamiany lokalnie przez
**Ollamę**. Platforma działa także bez niego. Wtedy okno czatu pokazuje komunikat
„Asystent jest chwilowo niedostępny”, a wpisane pytanie zostaje w polu.

- Model domyślny: `hf.co/speakleash/Bielik-Minitron-7B-v3.0-Instruct-GGUF:Q4_K_M` (ok. 4–5 GB).
  Ollama pobiera go bezpośrednio z Hugging Face
  ([karta modelu](https://huggingface.co/speakleash/Bielik-Minitron-7B-v3.0-Instruct-GGUF)).
- Konfiguracja w `.env`:

| Zmienna | Domyślnie | Znaczenie |
|---------|-----------|-----------|
| `OLLAMA_BASE_URL` | `http://ollama:11434` | adres serwera Ollama widziany z kontenera `web` |
| `OLLAMA_MODEL` | `hf.co/speakleash/Bielik-Minitron-7B-v3.0-Instruct-GGUF:Q4_K_M` | nazwa modelu w Ollamie |
| `OLLAMA_TIMEOUT` | `180` | limit czasu odpowiedzi (sekundy) |
| `OLLAMA_MODEL_LABEL` | puste | nazwa modelu wyświetlana uczniowi (puste = wyliczona z `OLLAMA_MODEL`, np. „Bielik-Minitron-7B-v3.0-Instruct (Q4_K_M)”) |
| `CHATBOT_QUIZ_LOCK_MINUTES` | `60` | ile minut po otwarciu testu asystent odmawia pomocy (o ile test nie zostanie wcześniej wysłany) |

Po każdej zmianie `.env` przeładuj kontener: `docker compose up -d web`.

---

## Wariant A — Ollama w Dockerze (najprościej, na procesorze)

Nie wymaga instalowania niczego poza Dockerem. Model liczy jednak na CPU, więc odpowiedź
pojawia się wolniej (widać ją na bieżąco, fragment po fragmencie).

```bash
docker compose --profile ai up -d ollama
docker compose exec ollama ollama pull hf.co/speakleash/Bielik-Minitron-7B-v3.0-Instruct-GGUF:Q4_K_M
docker compose up -d web
```

`.env` domyślnie wskazuje na `http://ollama:11434`, więc nic więcej nie trzeba zmieniać.
Model zostaje w wolumenie `ollama_data` i nie trzeba go pobierać ponownie.

Zatrzymanie: `docker compose stop ollama`.

## Wariant B — Ollama na systemie z kartą graficzną (szybciej)

Zalecany przy karcie NVIDIA, gdy Docker nie ma `nvidia-container-toolkit`.

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull hf.co/speakleash/Bielik-Minitron-7B-v3.0-Instruct-GGUF:Q4_K_M
```

Domyślnie Ollama nasłuchuje tylko na `127.0.0.1`, więc kontener `web` jej nie widzi.
Trzeba ją przestawić na wszystkie interfejsy:

```bash
sudo systemctl edit ollama
```

W otwartym pliku dopisz:

```ini
[Service]
Environment="OLLAMA_HOST=0.0.0.0"
```

```bash
sudo systemctl restart ollama
```

W `.env` ustaw:

```env
OLLAMA_BASE_URL=http://host.docker.internal:11434
```

i przeładuj aplikację: `docker compose up -d web`. Mapowanie `host.docker.internal` jest
już w `docker-compose.yml`.

> `OLLAMA_HOST=0.0.0.0` otwiera port 11434 także w sieci lokalnej. W obcej sieci zablokuj go
> firewallem (np. `sudo ufw deny 11434` i zezwól tylko sieci Dockera).

Przy uruchamianiu Django bez Dockera (`.venv`) wystarczy `OLLAMA_BASE_URL=http://localhost:11434`,
bez zmiany `OLLAMA_HOST`.

---

## Sprawdzenie, czy model odpowiada

```bash
curl http://localhost:11434/api/tags          # lista pobranych modeli
curl http://localhost:11434/api/chat -d '{
  "model": "hf.co/speakleash/Bielik-Minitron-7B-v3.0-Instruct-GGUF:Q4_K_M",
  "messages": [{"role": "user", "content": "Cześć, kim jesteś?"}],
  "stream": false
}'
```

Z wnętrza kontenera (sprawdza adres z `.env`):

```bash
docker compose exec web python -c "import ollama, os; print(ollama.Client(host=os.environ['OLLAMA_BASE_URL']).list())"
```

Pierwsza odpowiedź po starcie trwa dłużej, bo model ładuje się do pamięci. Potem zostaje w niej
30 minut.

## Scenariusz testu na platformie

Konta z danych demo (`python manage.py seed_demo`): `student1` / `test123!`, `student2` / `test123!`.

1. **Lekcja.** Zaloguj się jako `student1` → Moje kursy → „Python dla początkujących” →
   lekcja „Zmienne i typy danych” → **Zapytaj asystenta o tę lekcję**. Zapytaj np. „Czym różni
   się int od float?”. Odpowiedź powinna nawiązywać do treści lekcji.
2. **Panel obok kursu.** Na desktopie (od 1024 px) rozmowa otwiera się z prawej, a lekcja
   zostaje widoczna po lewej. Kliknij „Dalej”: panel zostaje otwarty, a w nagłówku widać nową lekcję.
3. **Blokada w trakcie testu.** Otwórz „Test: podstawy Pythona” i zapytaj o pytanie z testu.
   Asystent odmawia („Podczas testu nie mogę udzielać pomocy…”), a model nie jest w ogóle pytany.
   To samo dzieje się po przejściu na lekcję w innej karcie.
4. **Wyjaśnienie po teście.** Wyślij test, a na stronie wyniku kliknij **Poproś asystenta o wyjaśnienie**
   i zapytaj, czemu odpowiedź była zła. Dopóki zostały podejścia, asystent naprowadza bez
   podawania poprawnej odpowiedzi. Po zaliczeniu albo wyczerpaniu podejść może ją omówić.
5. **Historia.** Menu **Asystent** pokazuje osobne rozmowy dla każdego kursu oraz rozmowę ogólną.
   `student2` nie widzi rozmów `student1`.
6. **Brak modelu.** `docker compose stop ollama` (albo `sudo systemctl stop ollama`), a potem pytanie.
   Pojawia się polski komunikat o niedostępności, a pytanie zostaje w polu.

## Rozwiązywanie problemów

| Objaw | Przyczyna / rozwiązanie |
|-------|------------------------|
| „Asystent jest chwilowo niedostępny — serwer modelu (Ollama) nie odpowiada” | Ollama nie działa albo zły `OLLAMA_BASE_URL`. W wariancie B sprawdź `OLLAMA_HOST=0.0.0.0` i przeładuj `web`. |
| „Model asystenta nie jest jeszcze pobrany…” | Uruchom `ollama pull` z nazwą z `OLLAMA_MODEL` (nazwy muszą być identyczne). |
| „Asystent odpowiadał zbyt długo” | Model na CPU jest za wolny: zwiększ `OLLAMA_TIMEOUT` albo użyj wariantu B. |
| „Asystent odpowiada jeszcze na poprzednie pytanie” | Uczeń może mieć tylko jedno pytanie w toku. Poczekaj albo kliknij „Zatrzymaj”. |
| Asystent odmawia pomocy, choć test jest już zamknięty | Test otwarto i nie wysłano. Blokada mija po `CHATBOT_QUIZ_LOCK_MINUTES` albo po wysłaniu testu. |
| `pull` zgłasza brak modelu | Wybierz inny plik GGUF Bielika z Hugging Face (`hf.co/speakleash/<repo>-GGUF:<kwantyzacja>`) i wpisz go w `OLLAMA_MODEL`. |

## Jak to działa (dla prowadzącego)

- Widżet pobiera krótkotrwały token **JWT** (`POST /chatbot/token/`, sesja + CSRF). Endpointy
  `/chatbot/api/history/`, `/chatbot/api/message/` i `/chatbot/api/clear/` przyjmują wyłącznie
  nagłówek `Authorization: Bearer …`.
- Odpowiedź modelu przychodzi strumieniem NDJSON. Obie wiadomości zapisują się w `ChatSession` /
  `ChatMessage`: jedna rozmowa na ucznia i kurs oraz jedna ogólna.
- Prompt systemowy zawiera opis i program kursu, treść otwartej lekcji (bez HTML), a na stronie
  wyniku także pytania, odpowiedzi ucznia i komentarze nauczyciela.
- Otwarcie testu tworzy `QuizSitting`, a wysłanie go usuwa. Dopóki istnieje, każde pytanie
  dostaje stałą odmowę bez kontaktu z modelem.
- Kod: `apps/chatbot/` (`api.py`, `llm.py`, `services.py`, `conversation.py`), widżet
  `templates/chatbot/widget.html` + `static/js/chatbot.js`, testy `apps/chatbot/tests.py`.
