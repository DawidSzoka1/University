# Uruchomienie na laptopie z Windowsem (Docker już zainstalowany, `.env` z komputera domowego)

Krótka ściąga na prezentację. Zakłada, że na laptopie działa **Docker Desktop** (WSL 2)
i że przenosisz plik **`.env` z komputera, na którym projekt już działa**.
Pełna instrukcja od zera (instalacja WSL i Dockera, wszystkie problemy): [`uruchomienie-windows.md`](uruchomienie-windows.md).

Komendy wpisujesz w **PowerShellu**, w katalogu projektu (tam, gdzie leży `docker-compose.yml`).

---

## 0. Przed wyjściem z domu (na komputerze z Linuksem)

1. **Wypchnij najnowszy kod do repozytorium** (`git add` / `commit` / `push`) — razem ze wszystkimi nowymi plikami,
   zwłaszcza migracjami: `apps/learning/migrations/0005_*.py` … `0008_*.py`, `apps/accounts/migrations/0006_*.py`,
   `apps/catalog/migrations/0009_*.py`, nowymi modułami (`apps/learning/styles.py`, `style_views.py`,
   `progress_views.py`, `apps/core/stats.py`, `apps/accounts/lockout.py`), szablonami w `templates/` i katalogiem `docs/`.
   Najprościej sprawdzić `git status` — nie może zostać nic z `??` poza `.env`.
   Bez migracji laptop wyrzuci błąd w stylu `relation "learning_surveyquestion" does not exist`.
2. **Skopiuj plik `.env`** na pendrive / do prywatnej chmury. **Nie wrzucaj go do repozytorium** — są w nim
   klucze Stripe i portali OAuth (`.env` jest w `.gitignore`, więc `git` go nie zabierze).

## 1. Pobierz projekt na laptop

```powershell
cd C:\
mkdir projekty -ErrorAction SilentlyContinue
cd C:\projekty
git clone <adres-repozytorium> projectELearning
cd projectELearning
```

Jeśli projekt już tam jest — tylko `git pull`.

> Ścieżka bez spacji i polskich liter (dobrze: `C:\projekty\projectELearning`).

## 2. Wstaw `.env` i popraw w nim 2 rzeczy

Skopiuj plik `.env` do `C:\projekty\projectELearning\` (obok `docker-compose.yml`), potem otwórz go:

```powershell
notepad .env
```

**a) Wyczyść sekret webhooka Stripe** — na laptopie `stripe-cli` dostanie inny sekret i sam go zapisze:

```env
STRIPE_WEBHOOK_SECRET=
```

(Aplikacja najpierw bierze wartość z `.env`; stara wartość z domu powodowałaby odrzucanie webhooków
z komunikatem „Nieprawidłowy podpis webhooka”.)

**b) Dopisz na końcu pliku linię**, żeby jedno `docker compose up` uruchamiało też asystenta AI i webhook Stripe:

```env
COMPOSE_PROFILES=ai,stripe
```

Resztę zostaw bez zmian. Sprawdź, że plik nazywa się dokładnie `.env`, a nie `.env.txt`:

```powershell
Get-ChildItem -Force .env*
```

Jeśli widzisz `.env.txt` → `Rename-Item .env.txt .env`.

## 3. Uruchom

Docker Desktop musi być włączony (zielone **Engine running**).

```powershell
docker compose up --build -d
docker compose ps
```

Powinno działać 5 usług: `db`, `redis`, `web`, `ollama`, `stripe-cli`. Pierwsze budowanie trwa kilka minut.
Na koniec poczekaj na start aplikacji:

```powershell
docker compose logs -f web
```

Gdy pojawi się `Starting development server at http://0.0.0.0:8000/` → `Ctrl+C` (aplikacja działa dalej).

## 4. Dane demo i konta

```powershell
docker compose exec -T web python scripts/seed_data.py --reset
```

Tworzy kursy, testy, style uczenia się, zamówienia, galerię itd. i wypisuje konta. Najważniejsze:

| Login | Hasło | Rola |
|-------|-------|------|
| `admin` | `admin123!` | administrator |
| `instructor` | `test123!` | nauczyciel (moderator) |
| `student1` | `test123!` | uczeń — styl „Wzrokowiec” pasuje |
| `student2` | `test123!` | uczeń — system poleca zmianę stylu na „Praktyk” |

Platforma: **<http://localhost:8000>** · panel: **<http://localhost:8000/admin/>**.
Co i w jakiej kolejności pokazywać: [`prezentacja.md`](prezentacja.md).

> Uruchamiaj na **porcie 8000**. Logowanie przez Discord / GitLab / Bitbucket ma w portalach zapisany
> adres zwrotny `http://localhost:8000/...` — na innym porcie lub adresie OAuth nie zadziała.

## 5. Asystent AI — pobierz model (raz, ok. 4,5 GB — zrób to w domu albo na dobrym Wi-Fi)

```powershell
docker compose exec ollama ollama pull hf.co/speakleash/Bielik-Minitron-7B-v3.0-Instruct-GGUF:Q4_K_M
docker compose exec ollama ollama list
```

Model zostaje w wolumenie Dockera — przy kolejnych uruchomieniach nie trzeba go pobierać.

- W Dockerze model liczy **na procesorze**: odpowiedź może trwać 20–60 s. Przed pokazem zadaj jedno
  pytanie na rozgrzewkę (pierwsze ładuje model do pamięci).
- Model potrzebuje ok. **6 GB wolnej pamięci dla Dockera**. Jeśli laptop ma 8 GB RAM i asystent nie
  odpowiada — w pliku `C:\Users\<Ty>\.wslconfig` ustaw:

  ```ini
  [wsl2]
  memory=7GB
  ```

  potem `wsl --shutdown` i ponowne uruchomienie Docker Desktop.
- **Szybciej z kartą graficzną (opcjonalnie):** zainstaluj Ollamę dla Windows (<https://ollama.com/download/windows>),
  pobierz model poleceniem `ollama pull …` jak wyżej (bez `docker compose exec ollama`), w `.env` ustaw
  `OLLAMA_BASE_URL=http://host.docker.internal:11434`, z `COMPOSE_PROFILES` usuń `ai` (zostaw `COMPOSE_PROFILES=stripe`)
  i wykonaj `docker compose up -d`.

## 6. Płatność testowa (Stripe)

`stripe-cli` startuje sam (profil `stripe`) i zapisuje sekret do `.stripe\webhook_secret`. Sprawdź:

```powershell
docker compose logs stripe-cli
```

Powinno być `Ready!` i `Sekret webhooka zapisany`. Karta testowa: `4242 4242 4242 4242`, dowolna przyszła data, dowolne CVC.
Laptop musi mieć **internet** (Stripe, filmy z YouTube, logowanie przez portale).

> Nawet gdyby webhook nie dotarł, zamówienie zmieni się na „Opłacone” po wejściu na jego stronę —
> aplikacja sama zapyta Stripe o stan płatności.

## 7. Sprawdzenie przed prezentacją (2 minuty)

- [ ] <http://localhost:8000> otwiera się, logowanie `student2` / `test123!` działa.
- [ ] Menu konta → **Styl uczenia się** → widać „Polecamy styl »Praktyk«”.
- [ ] Kurs Pythona → lekcja „Zmienne i typy danych” → przełącznik wersji (Wzrokowiec, Słuchowiec…).
- [ ] Asystent odpowiada na pytanie (rozgrzany model).
- [ ] `docker compose logs stripe-cli` → `Ready!`.
- [ ] (Opcjonalnie) testy: `docker compose exec web python manage.py test` → `OK` (393 testy, ok. 2–3 min).

## Po prezentacji

```powershell
docker compose stop        # zatrzymuje wszystko, dane zostają
```

Następnym razem wystarczy `docker compose up -d`. Usunięcie wszystkiego (baza, pliki, **pobrany model**):
`docker compose down -v`.

## Najczęstsze problemy na laptopie

| Objaw | Co zrobić |
|-------|-----------|
| `port is already allocated` (8000) | Coś innego zajmuje port 8000: `netstat -ano \| findstr :8000`, zamknij ten program (PID w ostatniej kolumnie). Nie zmieniaj portu — OAuth działa tylko na 8000. |
| `Bind for 127.0.0.1:5433 failed` / `6380` / `11434` | W `.env` zmień `DB_HOST_PORT=5434` / `REDIS_HOST_PORT=6381`; dla `11434` — działa już Ollama zainstalowana w Windows: użyj wariantu z kartą graficzną z punktu 5. |
| `relation "learning_…" does not exist` | Brak nowych migracji w repozytorium — zrób punkt 0.1 i `git pull`, potem `docker compose exec web python manage.py migrate`. |
| Webhook: „Nieprawidłowy podpis webhooka” w logach `web` | Nie wyczyszczony `STRIPE_WEBHOOK_SECRET` (punkt 2a) → popraw `.env`, `docker compose up -d web`. |
| `stripe-cli` kończy się od razu | Brak `STRIPE_SECRET_KEY` w `.env` albo brak internetu. |
| Asystent: „serwer modelu nie odpowiada” | `docker compose ps` — czy `ollama` działa? Jeśli nie: `docker compose up -d ollama`. |
| Asystent: „model nie jest jeszcze pobrany” | Punkt 5 (`ollama pull …`). |
| Logowanie Discord/GitLab: `redirect_uri mismatch` | Otwierasz inny adres niż `http://localhost:8000` (np. `127.0.0.1:8000`) — wejdź przez `localhost`. |
| Strona działa bardzo wolno | Docker Desktop → Settings → Resources (więcej CPU/RAM), folder projektu w wyjątkach antywirusa. |
