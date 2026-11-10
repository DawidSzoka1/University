# Płatności Stripe (sandbox) — konto, klucze i test

Sklep przyjmuje płatności przez **Stripe Checkout** w trybie testowym. Pieniądze nie są
pobierane: płaci się kartami testowymi. Bez kluczy platforma też działa. Zamówienie zostaje
wtedy przyjęte jako „Oczekuje na płatność”, a status zmienia ręcznie moderator albo administrator.

Przepływ:

1. **Koszyk** — ten sam kurs dodany drugi raz zwiększa ilość pozycji (kumulacja).
2. **Zamówienie** — dane podstawione z profilu. Można je zmienić i zapisać w profilu.
   Po złożeniu zamówienie ma status **„Oczekuje na płatność”**, a na e-mail kupującego idzie potwierdzenie.
3. **Stripe Checkout** — strona płatności Stripe (karta, BLIK, Przelewy24 — zależnie od ustawień konta).
4. **Potwierdzenie** — webhook `checkout.session.completed` (oraz sprawdzenie sesji po powrocie
   ze Stripe) zmienia status na **„Opłacone”**, odblokowuje kurs i wysyła e-mail z potwierdzeniem.
   Każda sztuka ponad pierwszą to kod dostępu do przekazania innej osobie.

---

## 1. Załóż konto Stripe (bezpłatne, bez firmy)

1. Wejdź na <https://dashboard.stripe.com/register>.
2. Podaj e-mail, imię i nazwisko, kraj **Polska** i hasło, potem potwierdź adres e-mail
   (link w wiadomości od Stripe).
3. Po zalogowaniu Stripe proponuje „aktywację konta” (dane firmy, konto bankowe).
   **Pomiń to** — do testów wystarczy tryb testowy (sandbox), który działa od razu.
4. Upewnij się, że jesteś w trybie testowym. W panelu musi być widoczny przełącznik albo napis
   **Test mode / Sandbox**. Klucze testowe zaczynają się od `pk_test_` i `sk_test_`.

## 2. Skopiuj klucze API

1. W panelu: **Developers → API keys** (bezpośrednio: <https://dashboard.stripe.com/test/apikeys>).
2. Skopiuj **Publishable key** (`pk_test_…`) i **Secret key** (`sk_test_…`, przycisk *Reveal test key*).
3. Wpisz je do pliku `.env` w katalogu projektu:

```env
STRIPE_SECRET_KEY=sk_test_...
STRIPE_PUBLISHABLE_KEY=pk_test_...
SITE_URL=http://localhost:8000
```

`SITE_URL` to adres, pod który Stripe odeśle kupującego po płatności, i ten sam, który trafia do linków w e-mailach.

> Klucz `sk_test_…` jest tajny — nie wrzucaj `.env` do repozytorium (plik jest w `.gitignore`, sprawdź to przed commitem).

## 3. Webhook lokalnie — Stripe CLI w `docker compose` (zalecane)

Stripe nie widzi `localhost`, więc zdarzenia o płatnościach przekazuje **Stripe CLI**. Jest dodany
do `docker-compose.yml` jako usługa `stripe-cli` (profil `stripe`). Uruchamia się razem z aplikacją,
bierze klucz z `.env`, **sam pobiera sekret webhooka** i zapisuje go do `.stripe/webhook_secret`,
skąd czyta go Django. Niczego nie trzeba kopiować ani instalować:

```bash
docker compose --profile stripe up -d
docker compose logs -f stripe-cli     # podgląd: --> checkout.session.completed / <-- [200]
```

Żeby zwykłe `docker compose up` od razu startowało webhook, odkomentuj w `.env`:

```env
COMPOSE_PROFILES=stripe
```

Warunki: w `.env` jest `STRIPE_SECRET_KEY`, a `DJANGO_ALLOWED_HOSTS` zawiera `web`
(CLI wysyła zdarzenia na `http://web:8000`; w dostarczonym `.env` już tak jest).

### Inne sposoby (gdy nie chcesz usługi w Dockerze)

- **Stripe CLI zainstalowane na systemie** (<https://docs.stripe.com/stripe-cli>):
  `stripe login`, potem `stripe listen --forward-to localhost:8000/orders/webhook/stripe/`. Wypisany
  `whsec_…` wpisz do `.env` jako `STRIPE_WEBHOOK_SECRET` i przeładuj `web`.
- **Bez webhooka w ogóle.** Płatność i tak zostanie potwierdzona automatycznie:
  - po powrocie ze Stripe serwer sam sprawdza sesję płatności w API Stripe,
  - jeśli kupujący zamknął kartę przed powrotem, status zaktualizuje się przy pierwszym otwarciu
    strony zamówienia (serwer znowu pyta Stripe o oczekujące zamówienie).

  Webhook jest potrzebny tylko do zmiany statusu „w tle”, bez udziału przeglądarki,
  np. przy płatnościach asynchronicznych.

> Na prezentacji wystarczy `docker compose --profile stripe up -d`. Gdyby usługa nie ruszyła
> (np. brak internetu w trakcie startu), zakup i tak przejdzie dzięki sprawdzaniu po powrocie.

## 4. Przeładuj aplikację

Biblioteka `stripe` doszła do `requirements.txt`, więc obraz trzeba raz przebudować:

```bash
docker compose build web
docker compose --profile stripe up -d
docker compose exec -T web python manage.py migrate
docker compose exec -T web python scripts/seed_data.py      # zamówienia demo (opcjonalnie)
```

Po każdej zmianie `.env` wystarczy `docker compose up -d web`.

## 5. (Opcjonalnie) BLIK i Przelewy24

**Settings → Payment methods** (<https://dashboard.stripe.com/test/settings/payment_methods>):
włącz **BLIK** i **Przelewy24**. Stripe Checkout pokaże je automatycznie, bo zamówienia są w PLN.
Karta działa bez tej konfiguracji.

## 6. Test zakupu

1. Zaloguj się jako `student1` / `test123!`.
2. Otwórz płatny kurs (np. „JavaScript w przeglądarce”) i kliknij **Dodaj do koszyka** dwa razy.
   W koszyku jest jedna pozycja z ilością **2** (kumulacja).
3. **Przejdź do zamówienia**. Dane są z profilu — zmień np. miasto i zaznacz „Zapisz te dane w moim profilu”.
4. Zaakceptuj regulamin i kliknij **Zamawiam i płacę**. Wtedy:
   - w logach `docker compose logs -f web` pojawi się e-mail „Zamówienie ORD-… przyjęte — oczekuje
     na płatność” (backend konsolowy),
   - przeglądarka przejdzie na stronę Stripe.
5. Zapłać kartą testową:

| Karta | Wynik |
|-------|-------|
| `4242 4242 4242 4242` | płatność udana |
| `4000 0025 0000 3155` | wymaga potwierdzenia 3D Secure (kliknij „Complete”) |
| `4000 0000 0000 9995` | odrzucona (brak środków) |

   Data ważności: dowolna przyszła (np. `12/34`), CVC: dowolne 3 cyfry, reszta dowolna.
   BLIK w trybie testowym: dowolny 6-cyfrowy kod.
6. Po powrocie zamówienie ma status **Opłacone**. W historii statusu jest wpis „Płatność potwierdzona
   przez Stripe”, kurs jest w „Moich kursach”, a w zamówieniu widać **kod dostępu** dla drugiej sztuki.
   W logach jest drugi e-mail z kodem, a `docker compose logs stripe-cli` pokazuje `[200]` dla `checkout.session.completed`.
7. Zaloguj się jako `student2` i wpisz kod na stronie **Moje zamówienia → Zrealizuj kod**.
8. Jako `admin` / `admin123!` (albo nauczyciel `instructor` / `test123!`): **Moderacja → Zamówienia**.
   Widać tam wszystkie zamówienia z filtrem statusu. W szczegółach można zmienić status
   (np. „Zrealizowane”), a kupujący dostanie powiadomienie.

Płatności widać też w panelu Stripe: **Payments** (tryb testowy).

## Prawdziwe e-maile zamiast logów (opcjonalnie)

Domyślnie wiadomości wypisują się w logach kontenera `web`. Żeby wysyłać je naprawdę, np. przez
skrzynkę testową [Mailtrap](https://mailtrap.io) albo Gmail z hasłem aplikacji, ustaw w `.env`:

```env
EMAIL_BACKEND=django.core.mail.backends.smtp.EmailBackend
EMAIL_HOST=sandbox.smtp.mailtrap.io
EMAIL_PORT=587
EMAIL_HOST_USER=...
EMAIL_HOST_PASSWORD=...
EMAIL_USE_TLS=True
DEFAULT_FROM_EMAIL=E-Learning <sklep@elearning.local>
```

## Rozwiązywanie problemów

| Objaw | Rozwiązanie |
|-------|-------------|
| Przycisk „Zapłać” nieaktywny, komunikat „Płatności online nie są skonfigurowane” | Brak `STRIPE_SECRET_KEY` w `.env` albo kontener nie został przeładowany (`docker compose up -d web`). |
| „Bramka płatności jest chwilowo niedostępna” | Zły klucz (literówka, klucz `live`) albo brak internetu. Szczegóły są w `docker compose logs web`. |
| Stripe CLI pokazuje `[400]` | Wpisany ręcznie `STRIPE_WEBHOOK_SECRET` nie pasuje do bieżącego `stripe listen`. Wyczyść go w `.env` (wtedy aplikacja użyje `.stripe/webhook_secret`) i przeładuj `web`. |
| `docker compose logs stripe-cli`: „Brak STRIPE_SECRET_KEY” | Wpisz klucz `sk_test_…` do `.env` i uruchom ponownie `docker compose --profile stripe up -d`. |
| Stripe CLI pokazuje `[400]` z `Invalid HTTP_HOST header: 'web:8000'` | Dopisz `web` do `DJANGO_ALLOWED_HOSTS` w `.env` i przeładuj `web`. |
| Po płatności status dalej „Oczekuje na płatność” | Płatność asynchroniczna (P24) czeka na potwierdzenie. Uruchom webhook (pkt 3) albo odśwież stronę po chwili. |
| `ModuleNotFoundError: stripe` | Obraz nie został przebudowany: `docker compose build web`. |

## Bezpieczeństwo (dla prowadzącego)

- Webhook `/orders/webhook/stripe/` to jedyny widok bez CSRF. Każde zdarzenie jest weryfikowane
  podpisem `Stripe-Signature` (`STRIPE_WEBHOOK_SECRET`), a nieprawidłowy podpis daje odpowiedź 400.
- Status zmienia się tylko wtedy, gdy `payment_status == "paid"` oraz kwota i waluta sesji zgadzają
  się z zamówieniem. Powtórzone zdarzenie niczego nie dubluje (operacja idempotentna).
- Powrót ze Stripe (`success_url`) nie ufa adresowi: serwer pobiera sesję z API Stripe i sprawdza ją tak samo.
- Ceny są kopiowane do zamówienia w chwili złożenia, więc późniejsza zmiana ceny kursu nie zmienia zamówienia.
- Kod: `apps/orders/` (`cart.py`, `services.py`, `payments.py`, `views.py`), testy: `apps/orders/tests.py`.
