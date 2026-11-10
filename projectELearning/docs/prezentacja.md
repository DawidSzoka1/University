# Scenariusz prezentacji — E-Learning

Ściąga do pokazania projektu prowadzącemu: kolejność, konta, co kliknąć i co powiedzieć.
Pełny pokaz trwa ok. 30 minut; jeśli czasu jest mniej, pokaż część **0**, **4** (style uczenia się, ankieta, polecane kursy), **5** (statystyki)
i po jednym punkcie z pozostałych.

---

## Przygotowanie (dzień wcześniej i 10 minut przed)

> Prezentujesz na laptopie z Windowsem? Najpierw przejdź [`uruchomienie-laptop-windows.md`](uruchomienie-laptop-windows.md)
> (przeniesienie `.env`, model AI, Stripe). Tam `COMPOSE_PROFILES=ai,stripe` w `.env` sprawia, że
> trzy pierwsze polecenia poniżej zastępuje jedno `docker compose up -d`.

```bash
docker compose up -d                                               # db, redis, web (http://localhost:8000)
docker compose exec -T web python scripts/seed_data.py --reset     # dane demo w znanym stanie
docker compose --profile ai up -d ollama                           # asystent Bielik (model jest już pobrany)
docker compose --profile stripe up -d                              # webhook płatności testowych
```

- `--reset` odtwarza **tylko** dane demo (kursy, konta z tabeli niżej). Twoje własne konto i kursy zostają.
- Jeśli port 8000 jest zajęty przez inny projekt, w `docker-compose.yml` zmień port `web` albo zatrzymaj tamten projekt.
- Pierwsza odpowiedź Bielika po starcie trwa dłużej (ładowanie modelu) — zadaj jedno pytanie na rozgrzewkę przed pokazem.
- Otwórz trzy okna przeglądarki (np. zwykłe + dwa prywatne / inne profile), żeby nie przelogowywać się co chwilę:
  **uczeń**, **nauczyciel**, **admin**.

## Konta

| Rola | Login | Hasło | Do czego na pokazie |
|------|-------|-------|---------------------|
| Administrator | `admin` | `admin123!` | kategorie i moderatorzy, galeria/slajder, motywy, panel `/admin/`, **dodanie nowego stylu uczenia się** |
| Nauczyciel (moderator) | `instructor` | `test123!` | kreator kursu, **wersje lekcji dla stylów**, moderacja, ocenianie testów, zamówienia |
| Uczeń — styl pasuje | `student1` | `test123!` | lekcje w wersji „Wzrokowiec”, analiza: „Twój styl sprawdza się najlepiej” |
| Uczeń — system poleca zmianę | `student2` | `test123!` | **rekomendacja stylu „Praktyk”**, koszyk z kumulacją, dyplom |
| Uczennica, autorka zgłoszenia | `ola.krawczyk` | `test123!` | zgłoszony kurs „Git i GitHub…” z wątkiem dyskusji |
| Uczeń bez wybranego stylu | `kuba.dabrowski` | `test123!` | wynik ankiety (Praktyk 80%) → „Ustaw jako mój styl”, polecane kursy tylko z ankiety |
| Pozostali uczniowie | np. `michal.kaczmarek` | `test123!` | komentarze, głosy, różne motywy |

Logowanie przez Discord / GitLab / Bitbucket działa na prawdziwych kontach w tych serwisach (klucze są w `.env`).

---

## 0. Start — o co chodzi w projekcie (1 min, bez logowania)

1. Strona główna: „Platforma kursów: sklep, nauka z testami, moderowana społeczność, polski asystent AI.
   Wszystko w jednym systemie Django, bez Bootstrapa — własny CSS”.
2. Pokaż, że liczby na stronie głównej (kursy, kategorie, motywy) są z bazy, nie wymyślone.
3. **Dostępność od pierwszej sekundy**: naciśnij `Tab` — pierwszy pojawia się link „Przejdź do treści”,
   każdy element ma widoczną ramkę fokusu.

## 1. Motywy kolorów (2 min)

- Gość: przełącznik motywu w nagłówku (ikonka z paskami) → strzałki ↑↓, `Enter`, `Esc` zamyka.
- „Cały wygląd liczy się z **5 kolorów bazowych**; reszta odcieni przez `color-mix()`.
  Kolor tekstu i tryb jasny/ciemny system wylicza z kontrastu WCAG — admin nie zrobi nieczytelnego motywu.”
- **admin** → `/admin/core/colortheme/` → edycja motywu próbnikami kolorów → zapis → platforma i panel w nowych kolorach.
- Zalogowany użytkownik ma motyw zapisany w profilu (działa na każdym urządzeniu), gość — w sesji.

## 2. Konta, role, logowanie (2 min)

- „Zaloguj się” → split-screen. Wpisz błędne hasło → komunikat tekstem, **login zostaje w polu**.
- Rejestracja: dostępność loginu/e-maila sprawdzana na żywo.
- Przyciski Discord / GitLab / Bitbucket (OAuth2, pobranie imienia, nazwiska, e-maila, awatara).
- Role: uczeń / nauczyciel (moderator kategorii) / administrator. API czatu i koszyka chronione JWT.

## 3. Katalog, kurs, nauka, testy (4 min) — `student1`

1. **Kursy** → filtry (kategoria, poziom, cena) wysyłane przyciskiem „Pokaż” (nie same — WCAG 3.2.2), paginacja.
   Okruszki z drzewa kategorii (MPTT).
2. „Python dla początkujących” → program kursu z sekcjami, test obowiązkowy w środku programu.
3. **Moje kursy** → kurs z paskiem postępu → lekcja: treść, kod, zdjęcie, film, pliki do pobrania.
4. Test: pytania ABC, wielokrotne i otwarte (tekst/kod/zdjęcie/plik). ABC liczone od razu, otwarte ocenia nauczyciel.
   „Test obowiązkowy blokuje dalsze lekcje — sprawdzane na serwerze, nie tylko ukrytym linkiem.”
5. **Mój profil** → dyplom PDF (kurs HTML i CSS) z numerem; publiczna strona weryfikacji dyplomu.

## 4. ★ Style uczenia się, ankieta i polecane kursy (8–10 min) — NOWE FUNKCJE

**Co powiedzieć na start:** „Każdy uczy się inaczej. Uczeń wybiera swój styl, autor kursu może przygotować
osobną treść lekcji dla każdego stylu, a system analizuje wyniki testów i podpowiada, który styl
najlepiej działa u tego ucznia.”

### 4a. Uczeń wybiera styl — `student1`
1. Menu konta (awatar w prawym górnym rogu) → **Styl uczenia się** (albo Profil → „Styl uczenia się: … · zmień”).
2. Cztery domyślne style (model VARK): **Wzrokowiec, Słuchowiec, Czytanie i pisanie, Praktyk** — każdy z opisem.
   `student1` ma wybranego Wzrokowca. Pokaż, że wybór robi się klawiaturą (Tab → strzałki → Enter).
3. Sekcja „02 Jak Ci idzie”: komunikat **„Twój styl »Wzrokowiec« sprawdza się najlepiej”** + tabela:
   lekcje przerobione w danej wersji, liczba ocenionych testów, średni wynik, odsetek zaliczonych, wynik ważony.

### 4b. Lekcja dopasowana do stylu — `student1`
1. Moje kursy → Python dla początkujących → lekcja **2.2 „Zmienne i typy danych”**.
2. Pasek **„Wersja lekcji”**: Podstawowa | **Wzrokowiec (Twój styl)** | Słuchowiec | Czytanie i pisanie | Praktyk.
   Domyślnie otwiera się wersja dla stylu ucznia (tabela „zmienna to podpisane pudełko”).
3. Kliknij **Słuchowiec** → inna treść + film. **Praktyk** → ćwiczenie w konsoli. **Podstawowa** → oryginał.
   „Uczeń może podejrzeć każdą wersję, ale domyślnie dostaje swoją.”
4. Lekcja **1.1 „Instalacja na Windows”** ma tylko 2 wersje, a np. **1.2 macOS** — żadnej:
   „Lekcja bez wersji jest taka sama dla wszystkich — autor nie musi pisać czterech wersji wszystkiego.”
5. „Oznacz jako ukończoną” zapisuje, **w której wersji** uczeń przerobił lekcję — to dane do analizy.

### 4c. Rekomendacja — `student2` (najważniejszy moment)
1. Zaloguj się jako `student2` → **Styl uczenia się**.
2. Ma wybranego **Słuchowca**, ale widać komunikat **„Polecamy styl »Praktyk«”**, a przy Praktyku znacznik
   „polecany dla Ciebie”.
3. Wyjaśnij tabelę: kurs Pythona przerobił w wersji **Praktyk** → 2 testy, 100%; kurs Dockera w wersji
   **Słuchowiec** → 1 test, 0%.
4. **Jak to liczy (powiedz krótko):**
   - każde podejście do testu dostaje styl, w którym uczeń przerabiał lekcje tego kursu;
   - dla stylu liczona jest średnia wyników, ale „ściągnięta” w stronę ogólnej średniej ucznia
     (wynik ważony = (n·średnia stylu + 2·średnia ogólna) / (n + 2)) — jeden szczęśliwy test nie decyduje;
   - styl jest polecany, gdy ma ≥ 2 ocenione testy i bije obecny styl o ≥ 5 punktów procentowych.
5. Wejdź w dowolną lekcję — na górze ta sama podpowiedź z linkiem „Zobacz analizę i zmień styl”.
6. Zmień styl na Praktyk → zapis → komunikat; lekcje z wersjami od razu pokazują wersję praktyczną.
   „Styl można zmienić w każdej chwili.”

### 4d. Autor przygotowuje wersje — `instructor`
1. Utwórz kurs / Moje kursy → Python dla początkujących → **Program kursu**: przy lekcjach napis
   „4 wersje dla stylów” i przycisk **Style**.
2. Lekcja „Zmienne i typy danych” → **Style**: lista stylów, przy każdym „ma własną wersję” albo
   „treść podstawowa” + **wskazówka dla autora** (co warto dodać w tej wersji).
3. **Edytuj** Praktyk → edytor CKEditor (treść sanityzowana na serwerze) + opcjonalny film z YouTube/Vimeo.
   Przy **Dodaj wersję** edytor startuje od treści podstawowej — autor tylko ją przerabia.
4. Przycisk **„Ta sama treść dla każdego stylu”** (z dialogiem potwierdzenia, nie `confirm()`):
   wszyscy widzą wersję podstawową, a zapisane wersje zostają w bazie i można je włączyć z powrotem.
   To samo pole jest w formularzu lekcji (sekcja „05 Style uczenia się”).

### 4e. Admin dodaje nowy styl — `admin`
1. `/admin/` → menu boczne **Katalog & Nauka → Style uczenia się**.
2. **Dodaj**: nazwa np. „Społeczny”, opis dla ucznia, wskazówka dla autora, kolejność → zapisz.
3. Pokaż od razu u ucznia (strona stylu) i u autora (lista wersji lekcji) — nowy styl jest na obu.
4. Odznaczenie „Aktywny” chowa styl z wyboru i z lekcji, ale wersje lekcji zostają w bazie.

### 4f. Bonus: asystent zna styl
Jako `student1` otwórz **Asystent** na lekcji i zapytaj „Wyjaśnij mi typy danych” — prompt zawiera styl
ucznia, więc wzrokowiec dostaje raczej zestawienie/tabelę. (Wymaga uruchomionej Ollamy.)

### 4g. Ankieta diagnostyczna — `kuba.dabrowski` albo nowe konto
1. Styl uczenia się → **02 Ankieta**: `kuba.dabrowski` ma już wynik (Praktyk 80%, paski procentowe) i przycisk
   **„Ustaw »Praktyk« jako mój styl”** — nie wybrał stylu sam.
2. Pokaż ankietę: **Wypełnij ankietę ponownie** → 10 pytań sytuacyjnych (żadne nie pyta wprost „jakim jesteś
   stylem”), każde to `fieldset` z `legend`, wybór strzałkami. Wyślij z jednym pustym pytaniem → komunikat
   przy pytaniu, pozostałe odpowiedzi zostają.
3. Na świeżo założonym koncie ankieta **od razu ustawia styl** (gdy uczeń nie miał stylu).
4. Admin: `/admin/` → *Pytania ankiety stylu* — edycja pytań i odpowiedzi (np. dla nowego stylu).

### 4h. Polecane kursy — `student1` i `kuba.dabrowski`
1. **Kursy**: na górze sekcja **„Polecane dla Ciebie”** z dopiskiem „Na podstawie: wybrany styl, wynik ankiety,
   wyniki testów” i powodem przy każdym kursie („Dla Ciebie: 4 z 7 lekcji ma wersję dla stylu…”,
   „Autor poleca kurs dla stylu…”).
2. Sortowanie **„Dopasowane do mnie (styl uczenia się)”** — cały katalog wg dopasowania.
3. Te same polecenia: „Moje kursy” → „Polecane dla Ciebie” i strona stylu → „04 Kursy dla Ciebie”.
4. Strona kursu (np. Django) — w panelu po prawej: „Dla Ciebie: …” i „Wersje dla stylów uczenia się”.
5. **Jak liczone:** profil ucznia = wybrany styl (waga 1) + rozkład z ankiety (waga 1) + styl najlepszych
   testów (waga 0,5), znormalizowane. Dopasowanie kursu do stylu = 0,7 × udział lekcji z wersją dla stylu +
   0,3 × czy autor zaznaczył styl. Wynik = suma (udział stylu u ucznia × dopasowanie kursu). Kursy, na które
   uczeń jest zapisany, nie są polecane.
6. Nauczyciel w formularzu kursu (sekcja **05 Style uczenia się**) zaznacza, dla jakich stylów kurs jest szczególnie dobry.

### 4i. Panel uczniów kursu — `instructor`
Moje kursy → „Tworzę” → **Uczniowie** (albo strona kursu → „Uczniowie i postępy”) przy „Python dla początkujących”:
- podsumowanie: **zaczęło / ukończyło (procent)**, średni postęp, średni wynik testów, testy do oceny;
- rozkład stylów uczniów kursu;
- tabela: postęp, lekcje x/y, zaliczone testy, średni wynik, wybrany styl, wynik ankiety; filtr Wszyscy / W trakcie / Ukończyli.

**Gdyby padło pytanie „skąd system wie, jak mi idzie?”** — z dwóch rzeczy zapisywanych automatycznie:
w której wersji przerobiłeś lekcję (`LessonProgress.style`) i jaki wynik miał test po tych lekcjach
(`Attempt.learning_style`). Analiza bierze tylko ocenione podejścia (otwarte pytania czekające na ocenę się nie liczą).

## 5. Statystyki platformy — `admin` (2 min)

Menu konta → **Statystyki platformy** (albo Moderacja → zakładka **Statystyki**): użytkownicy wg ról, kursy wg
statusu, **rozpoczęte i ukończone kursy z procentem ukończenia** (łącznie i w tabeli dla każdego kursu),
testy (zaliczone / do oceny), style uczenia się (kto jaki wybrał, wyniki ankiet, wersje lekcji), asystent AI,
sprzedaż. Na stronie każdego kursu widać też „Zaczęło: N osób / Ukończyło: M (P%)”.

## 6. Bezpieczeństwo konta (1 min)

- Profil → **Zmień hasło** (konto tylko z portalu: **Ustaw hasło**).
- Logowanie złym hasłem: komunikat „Pozostało prób przed czasową blokadą: 4”. Po 5 próbach konto (login lub
  e-mail) jest blokowane na 5 minut — nawet poprawne hasło nie przejdzie. Pokaż na koncie testowym, np.
  `bartek.sikora`, żeby nie zablokować sobie kont do pokazu.

## 7. Społeczność i moderacja (3 min)

- `instructor` → **Moderacja**: zakładki Zgłoszenia / Komentarze / Do oceny / Kategorie / Zamówienia z licznikami.
- **Zgłoszenia** → kurs „Git i GitHub w pracy zespołowej” (`ola.krawczyk`) → wątek dyskusji zapisany w bazie
  → Zaakceptuj / Odrzuć (odrzucenie wymaga powodu) → autorka dostaje powiadomienie.
- Pod kursem Pythona: komentarze w wątkach (4 poziomy wcięć), ocena 1–5, głosy lubię/nie lubię,
  ranga przy nazwisku. Nowy komentarz ucznia czeka na akceptację — nauczyciel akceptuje jednym kliknięciem.
- **Do oceny**: test z pytaniem otwartym → punkty + komentarz → uczeń dostaje powiadomienie.
- Dzwonek: panel 5 ostatnich powiadomień, `Esc` zamyka; pełna lista z filtrem i paginacją.

## 8. Sklep i płatność (3 min) — `student2`

1. **Koszyk**: „JavaScript w przeglądarce” × 2 — **kumulacja** (ponowne „Dodaj do koszyka” zwiększa ilość).
   Zmiana ilości bez przeładowania strony (API z JWT), bez JS działa zwykły formularz.
2. Zamówienie: adres podstawiony z profilu, można zmienić i zapisać do profilu.
3. Stripe (tryb testowy): karta `4242 4242 4242 4242`, dowolna przyszła data i CVC → powrót →
   status **„Opłacone”** (ustawia go webhook z podpisem, nie powrót przeglądarki) → kurs w „Moje kursy”,
   druga sztuka = **kod dostępu** dla innej osoby. E-maile widać w logach: `docker compose logs web`.
4. `instructor` → Moderacja → Zamówienia: ręczna zmiana statusu z historią zmian.

## 9. Galeria i slajder (2 min) — `admin`

- Galeria → **Zdjęcia w bazie**: upload ze skalowaniem Pillow (komunikat „przeskalowano z …”), wymagany tekst alternatywny.
- **Układy slajdera**: kilka układów, przeciąganie kart myszą **i** przyciskami ↑↓ (alternatywa klawiaturowa),
  zapis kolejności w tle (`fetch`), wybór układu wyświetlanego.
- Publiczna **Galeria**: zapętlony slajder, 1/2/3 karty zależnie od szerokości, przesuw o jedną kartę,
  pełne opisy, pauza, strzałki ←/→.

## 10. Asystent AI (2 min) — dowolny uczeń

- Przycisk **Asystent** (prawy dolny róg) → pytanie o lekcję → odpowiedź strumieniowo po polsku (Bielik 7B lokalnie).
- „Zapytaj asystenta o tę lekcję” — model dostaje treść lekcji jako kontekst.
- Otwórz test i zapytaj asystenta → **odmowa** (blokada w trakcie testu, bez pytania modelu).
- `Esc` zamyka okno, fokus wraca na przycisk.

## 11. Anty-WCAG (2 min) — bez logowania

- Stopka → **Strona Anty-WCAG (demonstracja)** → `/wcag-anty-wzorzec/`.
- Pokaż: brak fokusu, nieczytelny kontrast, pułapkę `Tab` w „Czat z konsultantem”, menu tylko na najechanie,
  formularz bez etykiet z resetem po 15 s, uciekający przycisk ×, popup newslettera bez zamknięcia.
- „Strona jest całkowicie odizolowana (własny szablon, CSS i JS); opis naruszeń: `/wcag-anty-wzorzec/opis/`.”

## 12. Na koniec — jakość

- `docker compose exec web python manage.py test` → **393 testy, wszystkie przechodzą**
  (w tym testy stylów, ankiety, polecania kursów, panelu uczniów, statystyk i blokady logowania).
- Dane demo jednym poleceniem (`seed_data.py --reset`), uruchomienie jednym `docker compose up`.
- Dokumentacja: `docs/Dokumentacja_E-Learning_Dawid_Szoka_89221.pdf`.

---

## Możliwe pytania i krótkie odpowiedzi

- **Dlaczego VARK?** — Prosty i znany podział na 4 style; lista nie jest zaszyta w kodzie — to tabela w bazie, admin dodaje kolejne.
- **Czy autor musi pisać 4 wersje każdej lekcji?** — Nie. Domyślnie lekcja ma jedną treść dla wszystkich;
  wersje dodaje się tylko tam, gdzie mają sens. Styl bez wersji dostaje treść podstawową.
- **Co jeśli uczeń nie wybierze stylu?** — Widzi wersję podstawową; gdy uzbiera wyniki, system podpowie styl („Najlepiej idzie Ci w stylu …”).
- **Czy rekomendacja zmienia styl sama?** — Nie, tylko podpowiada. Decyzja należy do ucznia.
- **Testy, pliki do pobrania — też w wersjach?** — Nie, są wspólne, żeby wyniki były porównywalne między stylami.
- **Bezpieczeństwo** — treść wersji czyszczona `bleach`, zmiany tylko POST + CSRF, wersje edytuje tylko autor kursu,
  nauczyciel kategorii i admin (inni dostają 404).
