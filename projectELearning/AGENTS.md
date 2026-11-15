# AGENTS.md — projectELearning

## Zanim cokolwiek zrobisz

1. Wczytaj skill **`elearning-pai`** (`.claude/skills/elearning-pai/`). Zawiera pełną
   specyfikację, wymagania usability/WCAG, docelowy model danych i audyt stanu.
   (Stary skill opencode `project-architecture` jest nieaktualny.)
2. Sprawdź `references/current-state.md` w tym skillu — pokazuje, co już istnieje,
   co jest stubem i jakie są blokery.
3. Nie zakładaj, że funkcja działa, bo istnieje plik. Wiele widoków to
   `HttpResponse("TODO: ...")`, a część aplikacji nie ma modeli.

## Projekt

Platforma e-learningowa (projekt PAI, cel 4.5+/5): kursy, koszyk, testy ABC/otwarte,
chatbot Bielik, moderowana społeczność, galeria/slajder, motywy kolorów, strona Anty-WCAG.

- Backend: Django 5.1, PostgreSQL 16, Redis 7, Docker Compose.
- Auth: `django.contrib.auth` + `django-allauth` (Discord, GitLab, Bitbucket; Facebook wyłączony) + DRF/SimpleJWT.
- Treści: CKEditor 5 + `bleach`; kategorie: MPTT; obrazy: Pillow; AI: Ollama/Bielik.
- UI: własny `static/css/main.css` — **bez Bootstrapa i gotowych szablonów**.
- Język UI: polski. Kod: angielski. Strefa czasowa: `Europe/Warsaw`.

## Aktualny etap (2026-10-29)

**Szkielet uruchomiony + motywy + redesign (2026-10-30): navbar, strona główna, logowanie, rejestracja.**
**Moduł 4 (2026-10-30): komentarze w wątkach z akceptacją, edycją i głosami, kolejki moderacji, wątek zgłoszenia kursu.**
**E-learning (2026-10-30): testy w programie (po lekcji / końcowe, obowiązkowe blokują dalszą naukę), ABC auto + otwarte oceniane ręcznie, dyplomy PDF.**
**Anty-WCAG (2026-10-30): `/wcag-anty-wzorzec/` (izolowany szablon + `static/anti_wcag/`), opis naruszeń `/wcag-anty-wzorzec/opis/`.**
**Galeria (2026-10-30): zdjęcia ze skalowaniem Pillow, układy slajdera z DnD (+ ↑↓), zapętlony slajder 1/2/3 karty.**
**Sklep (2026-11-01): koszyk z kumulacją (także gość), zamówienie z adresem z profilu, e-mail, Stripe Checkout sandbox + webhook → „Opłacone”, kody dostępu, obsługa statusów w Moderacji. Konfiguracja: `docs/platnosci-stripe.md`.**
**Asystent AI (2026-11-01): okno czatu Bielik (Ollama) w prawym dolnym rogu z kontekstem lekcji/wyniku testu, blokada w trakcie testu, API JWT ze strumieniem, historia rozmów per uczeń i kurs.**
**Style uczenia się (2026-11-05): `LearningStyle` (4 style VARK z migracji, admin dodaje kolejne), wersje lekcji `LessonVariant` + `Lesson.same_for_all_styles`, wybór stylu `/learning/styl-uczenia/`, analiza wyników i rekomendacja (`apps/learning/styles.py`). Ankieta stylu, polecanie kursów, panel uczniów kursu, statystyki `/statystyki/`, zmiana hasła i blokada logowania. Scenariusz pokazu: `docs/prezentacja.md`; laptop z Windows: `docs/uruchomienie-laptop-windows.md`.**
**Panele startowe (2026-11-15): `/` po zalogowaniu = panel ucznia / nauczyciela / administratora (`apps/core/dashboard.py`), prezentacja platformy pod `/o-platformie/`.**
Działa: `manage.py check`, migracje, seed, `runserver`, strona główna (hero, korzyści,
pasek motywów, CTA), wszystkie ścieżki z navbara, admin Django, wybór i edycja motywów.
Logowanie i rejestracja to jeden przycisk w navbarze. Płatności: Stripe (klucze testowe w `.env`,
instrukcja `docs/platnosci-stripe.md`). Asystent wymaga uruchomionej Ollamy z pobranym modelem — instrukcja: `docs/asystent-bielik.md`.
Szczegóły: skill → `references/current-state.md`.

## Zasady nienegocjowalne

- Cała aplikacja poza `/wcag-anty-wzorzec/` spełnia WCAG: pełna obsługa klawiaturą,
  widoczny `:focus-visible`, brak pułapek, `Esc` zamyka modale, każda kontrolka ma `<label for>`.
- Żadnego `alert()`, `confirm()`, `prompt()`. Komunikaty przez `messages`/HTML.
- Wszystkie listy paginowane (`Paginator`, partial `includes/pagination.html`).
- Dokładnie 5 kolorów bazowych (`--color-primary`, `--color-secondary`, `--color-accent`, `--color-surface`, `--color-surface-variant`);
  reszta odcieni przez `color-mix()` / `hsl(from ...)`. Nie dodawaj kolorów poza tokenami.
- Motywy to rekordy `core.ColorTheme`; admin dodaje/edytuje je w `/admin/core/colortheme/`.
  Kolor tekstu (`--color-on-*`) i tryb ciemny (`data-theme-mode`) wylicza
  `apps/core/colors.py` z kontrastu WCAG — nie wpisuj ich ręcznie. W stanie domyślnym
  arkusz `static/css/main.css` bezpośrednio steruje wyglądem (umożliwiając weryfikację
  ręczną zmianą pliku css).
- Wybór motywu jest globalny (navbar `<details>` + `/accounts/select-theme/`);
  zalogowany zapisuje go w `CustomUser.selected_theme`, gość w sesji.
- Role: `student`, `moderator`, `admin`. **Nie używaj `instructor`** — nie istnieje.
- Opisy HTML sanityzowane `bleach`; uploady walidowane i skalowane Pillow.
- API chronione JWT; operacje moderacyjne przez POST + CSRF.
- Każda zmiana modelu = migracja. Nie zostawiaj stubów `TODO` w ścieżce użytkownika.
- Strony Anty-WCAG nie „naprawiaj” — ma łamać dostępność celowo.

## Komendy

```bash
docker compose up --build                 # db + web + redis
python manage.py check                    # walidacja projektu
python manage.py makemigrations && python manage.py migrate
python manage.py createsuperuser
python manage.py seed_demo               # dane demo (dopisuje brakujące)
python manage.py seed_demo --reset       # odtwarza dane demo od zera
docker compose exec -T web python scripts/seed_data.py --reset   # to samo w Dockerze
pytest                                    # po dodaniu pytest.ini i testów
```

Lokalnie bez Dockera: `POSTGRES_HOST=localhost` (`.env` ma `db`), aktywne `.venv/`.
Pierwsze uruchomienie: `.env.example` → `.env`; instrukcja dla Windowsa (WSL 2 + Docker Desktop): `docs/uruchomienie-windows.md`.

## Mapa kodu

```
config/            settings, urls, wsgi/asgi
apps/core/         motywy, powiadomienia, strona główna
apps/accounts/     CustomUser, role, profil, OAuth2
apps/catalog/      kategorie MPTT, kursy, komentarze, zgłoszenia
apps/gallery/      galeria + slajder (upload, układy DnD, slajder)
apps/orders/       koszyk, zamówienia, Stripe, webhook, kody dostępu
apps/learning/     lekcje, quizy, odpowiedzi (BRAK modeli)
apps/chatbot/      asystent Bielik: widżet, API JWT, Ollama, historia
templates/         base + includes (brak szablonów widoków)
static/css/        design system
static/js/         nawigacja, powiadomienia, helper fetch
scripts/           seed danych
```

## Definicja ukończenia zmiany

- `python manage.py check` bez błędów, migracje czyste.
- Brak `NoReverseMatch`; nowe widoki mają URL + szablon + test.
- Ścieżka użytkownika przechodzi Tabem; modal zamyka się `Esc`; fokus wraca.
- Zmiana palety 3 tokenów faktycznie zmienia całą witrynę.
- Zaktualizowany `references/current-state.md`, jeśli zmienił się etap.
