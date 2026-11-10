# Uruchomienie na Windowsie przez Dockera — krok po kroku

Instrukcja dla Windows 10 (22H2) i Windows 11. Nie trzeba instalować Pythona, PostgreSQL
ani Redisa — wszystko działa w kontenerach Dockera. Pierwsze uruchomienie zajmuje ok. 10–15 minut
(pobieranie obrazów i budowanie), każde kolejne kilka sekund.

Komendy wpisuje się w **PowerShellu** (menu Start → „PowerShell” albo „Terminal”).

---

## Krok 1. Włącz WSL 2 (raz na komputerze)

Docker Desktop na Windowsie działa na WSL 2 (wbudowany w Windows podsystem Linuksa).

1. Kliknij prawym przyciskiem menu Start → **Terminal (administrator)** albo **Windows PowerShell (administrator)**.
2. Wpisz:

   ```powershell
   wsl --install
   ```

3. **Uruchom komputer ponownie.** Po restarcie może otworzyć się okno Ubuntu z prośbą o nazwę
   użytkownika — podaj dowolną (do projektu nie jest potrzebna) albo po prostu je zamknij.
4. Sprawdź, czy działa:

   ```powershell
   wsl --status
   ```

   Powinno być `Domyślna wersja: 2` (ang. `Default Version: 2`).

> Jeśli `wsl --install` zgłasza błąd wirtualizacji — włącz ją w BIOS/UEFI (opcja **Intel VT-x** /
> **Intel Virtualization Technology** albo **AMD-V / SVM Mode**). W Menedżerze zadań → Wydajność →
> Procesor musi być widoczne „Wirtualizacja: Włączone”.

## Krok 2. Zainstaluj Docker Desktop

1. Pobierz instalator: <https://www.docker.com/products/docker-desktop/> (**Download for Windows – AMD64**;
   na komputerach z procesorem ARM, np. Snapdragon — wersja ARM64).
2. Uruchom instalator, zostaw zaznaczone **Use WSL 2 instead of Hyper-V (recommended)**.
3. Po instalacji uruchom **Docker Desktop** z menu Start i zaakceptuj warunki. Logowanie do konta
   Docker nie jest wymagane (*Skip* / *Continue without signing in*).
4. Poczekaj, aż w lewym dolnym rogu okna Docker Desktop pojawi się zielone **Engine running**.
5. Sprawdź w PowerShellu (zwykłym, nie administratora):

   ```powershell
   docker --version
   docker compose version
   ```

   Obie komendy muszą wypisać numer wersji.

> Docker Desktop musi być **uruchomiony** za każdym razem, gdy chcesz korzystać z aplikacji.

## Krok 3. Pobierz projekt

**Wariant A — Git** (<https://git-scm.com/download/win>, przy instalacji domyślne opcje):

```powershell
cd C:\
mkdir projekty
cd C:\projekty
git clone <adres-repozytorium> projectELearning
cd projectELearning
```

**Wariant B — archiwum ZIP:** rozpakuj projekt np. do `C:\projekty\projectELearning`, a potem:

```powershell
cd C:\projekty\projectELearning
```

> Najlepiej ścieżka **bez spacji i polskich znaków** (np. nie `C:\Users\Łukasz\Pulpit\mój projekt`).
> Upewnij się, że jesteś w katalogu z plikiem `docker-compose.yml`: komenda `dir` powinna go pokazać.

## Krok 4. Utwórz plik konfiguracyjny `.env`

Plik `.env` nie jest w repozytorium (zawiera hasła i klucze). Skopiuj wzór:

```powershell
Copy-Item .env.example .env
```

Otwórz go w Notatniku:

```powershell
notepad .env
```

i zmień `DJANGO_SECRET_KEY` na dowolny długi losowy ciąg. Można go wygenerować w PowerShellu:

```powershell
-join ((48..57) + (65..90) + (97..122) | Get-Random -Count 50 | ForEach-Object { [char]$_ })
```

Wklej wynik po `DJANGO_SECRET_KEY=` i zapisz plik. Reszta wartości działa bez zmian.

> Klucze Stripe, portali OAuth i adres Ollamy są opcjonalne — bez nich platforma działa,
> a te moduły pokazują komunikat, że nie są skonfigurowane (patrz [Krok 9](#krok-9-opcjonalnie-płatności-asystent-ai-logowanie-przez-portale)).

## Krok 5. Zbuduj i uruchom aplikację

```powershell
docker compose up --build -d
```

- `--build` — buduje obraz aplikacji (pierwszy raz kilka minut),
- `-d` — działa w tle, okno PowerShella zostaje wolne.

Docker uruchomi trzy kontenery: `db` (PostgreSQL), `redis` i `web` (Django). Przy starcie `web`
sam tworzy tabele w bazie (`migrate`).

Sprawdź stan:

```powershell
docker compose ps
```

Wszystkie trzy usługi powinny mieć status `running` (`db` i `redis` także `healthy`).
Podgląd logów aplikacji (wyjście: `Ctrl+C`, aplikacja działa dalej):

```powershell
docker compose logs -f web
```

Gotowe, gdy w logach pojawi się `Starting development server at http://0.0.0.0:8000/`.

## Krok 6. Wczytaj dane demonstracyjne

```powershell
docker compose exec web python scripts/seed_data.py
```

Tworzy konta testowe, drzewo kategorii, 10 kursów z lekcjami, filmami i testami, galerię i slajder,
komentarze, zamówienia we wszystkich statusach, rozmowy z asystentem i powiadomienia (ok. 1 minuty).
Na końcu wypisuje listę kont.

## Krok 7. Otwórz platformę

W przeglądarce: **<http://localhost:8000>**

| Rola | Login | Hasło | Co pokazać |
|------|-------|-------|------------|
| Administrator | `admin` | `admin123!` | Moderacja, kategorie, galeria i slajder, motywy, panel `/admin/` |
| Nauczyciel (moderator) | `instructor` | `test123!` | tworzenie kursu, kolejki akceptacji, ocenianie testów, zamówienia |
| Uczeń | `student1` | `test123!` | Moje kursy, lekcje, testy, dyplom, asystent AI, koszyk |
| Uczeń | `student2` | `test123!` | koszyk z kumulacją, zamówienia |

Pozostali nauczyciele i uczniowie z danych demo mają hasło `test123!` (lista w wyniku kroku 6).

Ważne adresy:

- <http://localhost:8000/admin/> — panel administracyjny Django,
- <http://localhost:8000/wcag-anty-wzorzec/> — **strona Anty-WCAG** (celowo łamie dostępność),
  opis naruszeń: <http://localhost:8000/wcag-anty-wzorzec/opis/>.

## Krok 8. Codzienna praca

| Co | Komenda |
|----|---------|
| Zatrzymać aplikację (dane zostają) | `docker compose stop` |
| Uruchomić ponownie | `docker compose up -d` |
| Zatrzymać i usunąć kontenery (dane w bazie zostają) | `docker compose down` |
| Przywrócić dane demo od zera | `docker compose exec web python scripts/seed_data.py --reset` |
| Uruchomić testy automatyczne | `docker compose exec web python manage.py test` |
| Utworzyć własne konto administratora | `docker compose exec web python manage.py createsuperuser` |
| Przebudować po zmianie `requirements.txt` | `docker compose up --build -d` |
| Przeładować po zmianie `.env` | `docker compose up -d web` |
| **Usunąć wszystko łącznie z bazą i plikami** | `docker compose down -v` |

Kod projektu jest podpięty do kontenera, więc zmiany w plikach `.py`, szablonach i CSS
działają po odświeżeniu strony (`Ctrl+F5`) bez przebudowy.

## Krok 9. (Opcjonalnie) płatności, asystent AI, logowanie przez portale

### Płatności Stripe (sandbox)

Załóż bezpłatne konto testowe Stripe, wpisz klucze `sk_test_…` i `pk_test_…` do `.env`, a potem:

```powershell
docker compose --profile stripe up -d
```

Usługa `stripe-cli` sama odbiera webhooki płatności. Pełna instrukcja i karty testowe:
[`docs/platnosci-stripe.md`](platnosci-stripe.md).

### Asystent AI — Bielik

Model ma ok. 4,5 GB. Dwa sposoby:

- **Ollama zainstalowana w Windowsie (zalecane, korzysta z karty graficznej):**
  1. Pobierz i zainstaluj <https://ollama.com/download/windows>.
  2. W PowerShellu pobierz model:

     ```powershell
     ollama pull hf.co/speakleash/Bielik-Minitron-7B-v3.0-Instruct-GGUF:Q4_K_M
     ```

  3. W `.env` ustaw `OLLAMA_BASE_URL=http://host.docker.internal:11434`.
  4. `docker compose up -d web`.
- **Ollama w Dockerze (bez instalacji, ale na procesorze — wolniej):**

  ```powershell
  docker compose --profile ai up -d ollama
  docker compose exec ollama ollama pull hf.co/speakleash/Bielik-Minitron-7B-v3.0-Instruct-GGUF:Q4_K_M
  ```

  `OLLAMA_BASE_URL=http://ollama:11434` (wartość domyślna z `.env.example`).

Szczegóły i scenariusz sprawdzenia: [`docs/asystent-bielik.md`](asystent-bielik.md).

### Logowanie przez Discord, GitLab, Bitbucket

Utwórz aplikację OAuth w portalu, jako adres zwrotny podaj adres z komentarza w `.env.example`
(np. `http://localhost:8000/accounts/social/discord/login/callback/`), a identyfikator i sekret
wpisz do `.env` (`DISCORD_CLIENT_ID`, `DISCORD_CLIENT_SECRET` itd.). Potem `docker compose up -d web`.

---

## Rozwiązywanie problemów

| Objaw | Rozwiązanie |
|-------|-------------|
| `error during connect … dockerDesktopLinuxEngine` / `The system cannot find the file specified` | Docker Desktop nie jest uruchomiony. Włącz go i poczekaj na **Engine running**. |
| `env file … \.env not found` | Nie ma pliku `.env` — wykonaj [Krok 4](#krok-4-utwórz-plik-konfiguracyjny-env). Sprawdź też, czy Notatnik nie zapisał go jako `.env.txt` (`dir -Force` pokaże pełną nazwę; popraw: `Rename-Item .env.txt .env`). |
| `no configuration file provided: not found` | Jesteś w złym katalogu — przejdź do folderu z `docker-compose.yml` (`cd C:\projekty\projectELearning`). |
| `Bind for 0.0.0.0:8000 failed: port is already allocated` | Port 8000 zajmuje inny program. Znajdź go: `netstat -ano \| findstr :8000` (ostatnia kolumna to PID) i zamknij w Menedżerze zadań. |
| `Bind for 127.0.0.1:5433 failed` (albo `6380`) | Zmień w `.env` `DB_HOST_PORT` (np. na `5434`) albo `REDIS_HOST_PORT`, potem `docker compose up -d`. |
| `WSL 2 installation is incomplete` / `Virtualization support not detected` | Wróć do [Kroku 1](#krok-1-włącz-wsl-2-raz-na-komputerze): `wsl --update`, włączona wirtualizacja w BIOS, restart. |
| Strona `localhost:8000` się nie otwiera zaraz po starcie | Pierwsze uruchomienie trwa dłużej (migracje). Sprawdź `docker compose logs -f web` i poczekaj na `Starting development server`. |
| `DisallowedHost` / „Bad Request (400)” | Otwierasz inny adres niż `localhost` / `127.0.0.1` — dopisz go do `DJANGO_ALLOWED_HOSTS` w `.env` i `docker compose up -d web`. |
| Brak kursów i kont po starcie | Nie wczytano danych demo — [Krok 6](#krok-6-wczytaj-dane-demonstracyjne). |
| `relation … does not exist` | Migracje nie przeszły — `docker compose exec web python manage.py migrate`, potem krok 6. |
| Asystent: „Asystent jest chwilowo niedostępny” | Ollama nie działa albo model nie jest pobrany — [Krok 9](#asystent-ai--bielik). |
| Bardzo wolne działanie / budowanie | Daj Dockerowi więcej zasobów: Docker Desktop → *Settings → Resources*; dodaj folder projektu do wyjątków antywirusa. |
| Chcę zacząć zupełnie od nowa | `docker compose down -v`, potem kroki 5–6. **Usuwa bazę i wgrane pliki.** |
