"""Dane demonstracyjne — JEDYNE miejsce do edycji przy rozwoju projektu.

Komenda ``python manage.py seed_demo`` czyta te struktury i buduje z nich bazę:
konta (admin, nauczyciele, uczniowie), drzewo kategorii, 10 opublikowanych kursów z
programem (sekcje → lekcje z treścią, kodem, zdjęciami, filmami i plikami), testy,
zapisy z postępem, recenzje z głosami, zgłoszenia kursów i powiadomienia.

Jak rozszerzać:
* nowy kurs — dopisz słownik do ``COURSES`` (wzór: dowolny istniejący kurs),
* nowa funkcja (np. koszyk, oceny testów) — dodaj tu listę danych i jej obsługę w
  ``apps/core/management/commands/seed_demo.py`` (funkcja ``seed_<nazwa>``),
* po zmianie danych: ``python manage.py seed_demo --reset`` odtwarza wszystko od zera.

Filmy: ``"youtube:<id>"`` (linki sprawdzone przez oEmbed YouTube, materiały
anglojęzyczne) albo ``"file:<nazwa>"`` z katalogu ``apps/core/demo/assets/``.
Pliki do pobrania: ``(nazwa, treść)`` — treść tekstowa zapisywana jako plik.
Zdjęcia w lekcjach: ``image=(tekst alternatywny, podpis, [linie na obrazku])`` —
generowane Pillow przy seedzie i wstawiane do treści jako <figure>.
"""

from __future__ import annotations

PASSWORD = "test123!"
ADMIN_PASSWORD = "admin123!"


def lesson(title, minutes, html, *, preview=False, video=None, files=(), image=None):
    return {"title": title, "minutes": minutes, "html": html, "preview": preview,
            "video": video, "files": list(files), "image": image}


def single(text, correct, *wrong):
    """Pytanie jednokrotnego wyboru: poprawna odpowiedź + błędne."""
    return {"type": "closed_single", "text": text, "choices": [(correct, True)] + [(w, False) for w in wrong]}


def multiple(text, correct, wrong):
    return {"type": "closed_multiple", "text": text,
            "choices": [(c, True) for c in correct] + [(w, False) for w in wrong]}


def open_question(text, fmt="text", points=2, code=""):
    """Pytanie otwarte (ocenia nauczyciel). fmt: text / code / image / file."""
    return {"type": "open", "text": text, "choices": [], "format": fmt, "points": points, "code": code}


# ---------------------------------------------------------------------------
# Konta. Hasło każdego konta poza adminem: PASSWORD.
# theme — nazwa motywu (ColorTheme); None = domyślny.
# ---------------------------------------------------------------------------

ADMIN = {
    "username": "admin", "email": "admin@example.com", "first_name": "Ewa", "last_name": "Administratorka",
}

TEACHERS = [
    {"username": "instructor", "first_name": "Jan", "last_name": "Kowalski", "email": "jan.kowalski@example.com",
     "city": "Kraków", "theme": None},
    {"username": "m.wisniewska", "first_name": "Magdalena", "last_name": "Wiśniewska",
     "email": "m.wisniewska@example.com", "city": "Warszawa", "theme": "Las"},
    {"username": "p.zielinski", "first_name": "Piotr", "last_name": "Zieliński", "email": "p.zielinski@example.com",
     "city": "Gdańsk", "theme": "Ocean"},
    {"username": "k.lewandowska", "first_name": "Katarzyna", "last_name": "Lewandowska",
     "email": "k.lewandowska@example.com", "city": "Wrocław", "theme": "Neon"},
]

STUDENTS = [
    {"username": "student1", "first_name": "Anna", "last_name": "Nowak", "email": "student1@example.com",
     "street": "ul. Długa 12/4", "postal": "31-147", "city": "Kraków", "phone": "+48 600 100 200", "theme": None},
    {"username": "student2", "first_name": "Tomasz", "last_name": "Wójcik", "email": "student2@example.com",
     "street": "ul. Polna 3", "postal": "00-635", "city": "Warszawa", "phone": "+48 600 100 201", "theme": "Ocean"},
    {"username": "ola.krawczyk", "first_name": "Aleksandra", "last_name": "Krawczyk", "email": "ola.krawczyk@example.com",
     "street": "ul. Ogrodowa 8", "postal": "80-001", "city": "Gdańsk", "phone": "+48 600 100 202", "theme": "Las"},
    {"username": "michal.kaczmarek", "first_name": "Michał", "last_name": "Kaczmarek",
     "email": "michal.kaczmarek@example.com", "street": "ul. Lipowa 21", "postal": "50-001", "city": "Wrocław",
     "phone": "+48 600 100 203", "theme": None},
    {"username": "zuzia.mazur", "first_name": "Zuzanna", "last_name": "Mazur", "email": "zuzia.mazur@example.com",
     "street": "al. Róż 5", "postal": "90-001", "city": "Łódź", "phone": "+48 600 100 204", "theme": "Neon"},
    {"username": "kuba.dabrowski", "first_name": "Jakub", "last_name": "Dąbrowski",
     "email": "kuba.dabrowski@example.com", "street": "ul. Kwiatowa 2", "postal": "60-001", "city": "Poznań",
     "phone": "+48 600 100 205", "theme": None},
    {"username": "natalia.pawlak", "first_name": "Natalia", "last_name": "Pawlak", "email": "natalia.pawlak@example.com",
     "street": "ul. Słoneczna 14", "postal": "20-001", "city": "Lublin", "phone": "+48 600 100 206", "theme": "Las"},
    {"username": "bartek.sikora", "first_name": "Bartosz", "last_name": "Sikora", "email": "bartek.sikora@example.com",
     "street": "ul. Leśna 9", "postal": "70-001", "city": "Szczecin", "phone": "+48 600 100 207", "theme": None},
]

# ---------------------------------------------------------------------------
# Drzewo kategorii (MPTT) i przypisani nauczyciele-moderatorzy.
# ---------------------------------------------------------------------------

CATEGORIES = [
    {"name": "Programowanie", "moderators": ["instructor"], "children": [
        {"name": "Python"},
        {"name": "Aplikacje webowe"},
    ]},
    {"name": "Data Science", "moderators": ["m.wisniewska"], "children": [
        {"name": "Analiza danych"},
        {"name": "Uczenie maszynowe"},
    ]},
    {"name": "DevOps", "moderators": ["p.zielinski"], "children": [
        {"name": "Konteneryzacja"},
        {"name": "Linux"},
    ]},
    {"name": "Projektowanie UX/UI", "moderators": ["k.lewandowska"], "children": []},
    {"name": "Bezpieczeństwo IT", "moderators": ["p.zielinski"], "children": []},
]

# ---------------------------------------------------------------------------
# Kursy. status: approved (domyślnie) | pending | rejected.
# ---------------------------------------------------------------------------

COURSES = [
    # ------------------------------------------------------------------ 1
    {
        "title": "Python dla początkujących",
        "category": "Python", "author": "instructor", "price": "0.00", "level": "beginner",
        "description": "Kompletny kurs języka Python od podstaw: instalacja, zmienne, warunki, pętle i funkcje.",
        "about": "<p>Kurs dla osób, które nigdy nie programowały. Po każdej sekcji napiszesz działający "
                 "program, a na końcu sprawdzisz się w teście.</p><ul><li>instalacja Pythona na każdym systemie</li>"
                 "<li>pierwsze programy i praca z terminalem</li><li>zmienne, warunki, pętle i funkcje</li></ul>",
        "sections": [
            {"title": "Instalacja", "lessons": [
                lesson("Instalacja na Windows", 12, preview=True, html=
                       "<p>Pobierz instalator ze strony <a href=\"https://www.python.org/downloads/\">python.org</a> i uruchom go. "
                       "<strong>Zaznacz opcję „Add python.exe to PATH”</strong>, zanim klikniesz „Install Now”.</p>"
                       "<p>Sprawdź instalację w wierszu poleceń:</p><pre><code class=\"language-bash\">python --version</code></pre>",
                       image=("Okno instalatora z zaznaczoną opcją Add python.exe to PATH", "Najważniejsza opcja instalatora",
                              ["Python 3.13 Setup", "[x] Add python.exe to PATH", "> Install Now"])),
                lesson("Instalacja na macOS", 8,
                       "<p>Najprościej zainstalować Pythona menedżerem pakietów Homebrew:</p>"
                       "<pre><code class=\"language-bash\">brew install python\npython3 --version</code></pre>"),
                lesson("Instalacja na Linuksie", 6,
                       "<p>Większość dystrybucji ma Pythona w repozytorium. W Ubuntu i Debianie:</p>"
                       "<pre><code class=\"language-bash\">sudo apt install python3 python3-venv\npython3 --version</code></pre>"),
            ]},
            {"title": "Pierwsze linie kodu", "lessons": [
                lesson("Hello world", 15, video="file:python_hello_world.webm", html=
                       "<p>Pierwszy program w każdym języku wypisuje tekst na ekran. W Pythonie wystarczy jedna linijka:</p>"
                       "<pre><code class=\"language-python\">print(\"Hello, world!\")</code></pre>"
                       "<p>Zapisz ją w pliku <code>hello.py</code> i uruchom poleceniem <code>python hello.py</code>. "
                       "Film pokazuje cały proces w terminalu.</p>"),
                lesson("Zmienne i typy danych", 18,
                       "<p>Zmienna to nazwa, pod którą przechowujesz wartość. Python sam rozpoznaje typ:</p>"
                       "<pre><code class=\"language-python\">imie = \"Ola\"      # str\nwiek = 21          # int\nwzrost = 1.68      # float\nstudent = True     # bool</code></pre>"
                       "<table><thead><tr><th>Typ</th><th>Przykład</th><th>Do czego</th></tr></thead><tbody>"
                       "<tr><td>str</td><td>\"tekst\"</td><td>napisy</td></tr><tr><td>int</td><td>42</td><td>liczby całkowite</td></tr>"
                       "<tr><td>float</td><td>3.14</td><td>liczby z przecinkiem</td></tr><tr><td>bool</td><td>True</td><td>prawda/fałsz</td></tr>"
                       "</tbody></table>"),
                lesson("Zadanie: popraw program z pliku", 14, files=[(
                    "powitanie.py",
                    "# Zadanie: popraw dwa bledy, aby program wypisal powitanie.\n"
                    "imie = input(\"Jak masz na imie? \")\n"
                    "print(\"Czesc, \" + imie\n"
                    "print(\"Milo Cie poznac!)\n")], html=
                       "<p>Pobierz plik <code>powitanie.py</code> z sekcji „Pliki do pobrania”. Program ma wypisać "
                       "powitanie z Twoim imieniem, ale zawiera dwa błędy. Znajdź je, popraw i uruchom program.</p>"),
            ]},
            {"title": "Sterowanie programem", "lessons": [
                lesson("Instrukcja if", 16, video="youtube:kqtD5dpn9C8",
                       html="<p>Instrukcja <code>if</code> wykonuje kod tylko wtedy, gdy warunek jest prawdziwy:</p>"
                            "<pre><code class=\"language-python\">wiek = int(input(\"Ile masz lat? \"))\nif wiek &gt;= 18:\n    print(\"Pełnoletni\")\nelse:\n    print(\"Niepełnoletni\")</code></pre>"
                            "<p>Film (po angielsku) omawia warunki od 1:00:00.</p>"),
                lesson("Pętle for i while", 20,
                       "<p><code>for</code> przechodzi po elementach, <code>while</code> powtarza, dopóki warunek jest prawdziwy:</p>"
                       "<pre><code class=\"language-python\">for liczba in range(1, 6):\n    print(liczba * liczba)\n\nlicznik = 3\nwhile licznik &gt; 0:\n    print(licznik)\n    licznik -= 1</code></pre>"),
                lesson("Własne funkcje", 18,
                       "<p>Funkcja grupuje kod pod nazwą i może zwracać wynik:</p>"
                       "<pre><code class=\"language-python\">def pole_prostokata(a, b):\n    return a * b\n\nprint(pole_prostokata(3, 4))  # 12</code></pre>"),
            ]},
        ],
        "quiz": {"title": "Test: podstawy Pythona", "questions": [
            single("Które polecenie wypisuje tekst na ekran?", "print()", "echo()", "write()", "console.log()"),
            single("Jaki typ ma wartość 3.14?", "float", "int", "str", "bool"),
            multiple("Które z poniższych to pętle w Pythonie?", ["for", "while"], ["repeat", "loop"]),
            open_question("Napisz funkcję, która zwraca większą z dwóch liczb, i wyjaśnij, jak działa.", fmt="code", points=3),
        ]},
        # Test obowiązkowy w połowie kursu: bez zaliczenia dalsze lekcje są zablokowane
        "quizzes": [{
            "title": "Sprawdź się: pierwszy program", "after": "Hello world", "required": True, "attempts": 3,
            "questions": [
                single("Co wypisze print(\"2\" + \"2\")?", "22", "4", "błąd", "2 + 2"),
                single("Jak uruchomić plik program.py w terminalu?", "python program.py", "run program.py", "program.py --start", "open program.py"),
                multiple("Które zdania o funkcji print() są prawdziwe?", ["może wypisać kilka wartości naraz", "domyślnie kończy linię znakiem nowej linii"],
                         ["wymaga średnika na końcu", "działa tylko z tekstem"]),
            ],
        }],
    },
    # ------------------------------------------------------------------ 2
    {
        "title": "Python dla zaawansowanych: klasy, wyjątki i testy",
        "category": "Python", "author": "instructor", "price": "129.00", "level": "intermediate",
        "description": "Programowanie obiektowe, obsługa wyjątków, praca z plikami i testy z pytest.",
        "about": "<p>Dla osób, które znają podstawy Pythona i chcą pisać większe, czytelne programy.</p>",
        "sections": [
            {"title": "Programowanie obiektowe", "lessons": [
                lesson("Klasy i obiekty", 22, preview=True, html=
                       "<p>Klasa opisuje, jakie dane i zachowania ma obiekt:</p>"
                       "<pre><code class=\"language-python\">class Konto:\n    def __init__(self, wlasciciel, saldo=0):\n        self.wlasciciel = wlasciciel\n        self.saldo = saldo\n\n    def wplac(self, kwota):\n        self.saldo += kwota</code></pre>"),
                lesson("Dziedziczenie", 18,
                       "<p>Klasa potomna przejmuje zachowanie rodzica i może je rozszerzyć:</p>"
                       "<pre><code class=\"language-python\">class KontoOszczednosciowe(Konto):\n    def dolicz_odsetki(self, procent):\n        self.saldo *= 1 + procent / 100</code></pre>"),
            ]},
            {"title": "Wyjątki i pliki", "lessons": [
                lesson("Obsługa wyjątków", 16,
                       "<pre><code class=\"language-python\">try:\n    wynik = 10 / int(input(\"Dzielnik: \"))\nexcept ZeroDivisionError:\n    print(\"Nie dziel przez zero!\")\nexcept ValueError:\n    print(\"To nie jest liczba.\")</code></pre>"),
                lesson("Czytanie i zapis plików CSV", 20, files=[(
                    "oceny.csv", "uczen,matematyka,fizyka\nAnna,5,4\nTomasz,3,4\nOla,5,5\nMichal,4,3\n")], html=
                       "<p>Pobierz plik <code>oceny.csv</code> i policz średnią z matematyki modułem <code>csv</code>:</p>"
                       "<pre><code class=\"language-python\">import csv\n\nwith open(\"oceny.csv\", encoding=\"utf-8\") as plik:\n    oceny = [int(w[\"matematyka\"]) for w in csv.DictReader(plik)]\nprint(sum(oceny) / len(oceny))</code></pre>"),
            ]},
            {"title": "Testy automatyczne", "lessons": [
                lesson("Pierwszy test w pytest", 18,
                       "<p>Test to funkcja zaczynająca się od <code>test_</code> z instrukcją <code>assert</code>:</p>"
                       "<pre><code class=\"language-python\">def test_wplata():\n    konto = Konto(\"Ola\")\n    konto.wplac(100)\n    assert konto.saldo == 100</code></pre>"
                       "<pre><code class=\"language-bash\">pip install pytest\npytest</code></pre>"),
                lesson("Zadanie: przetestuj klasę Konto", 25, files=[(
                    "konto.py",
                    "class Konto:\n    def __init__(self, wlasciciel, saldo=0):\n        self.wlasciciel = wlasciciel\n        self.saldo = saldo\n\n"
                    "    def wplac(self, kwota):\n        self.saldo += kwota\n\n    def wyplac(self, kwota):\n        # Blad: brak sprawdzenia, czy wystarczy srodkow\n        self.saldo -= kwota\n")], html=
                       "<p>Napisz testy dla metody <code>wyplac</code> z pliku <code>konto.py</code>. Jeden z testów powinien "
                       "wykryć błąd — popraw go, aby wszystkie testy przechodziły.</p>"),
            ]},
        ],
        "quiz": {"title": "Test: Python zaawansowany", "questions": [
            single("Jak nazywa się metoda wywoływana przy tworzeniu obiektu?", "__init__", "__new_object__", "create", "constructor"),
            single("Który blok przechwytuje wyjątek?", "except", "catch", "rescue", "handle"),
            multiple("Co jest prawdą o pytest?", ["testy to funkcje test_…", "używa instrukcji assert"],
                     ["wymaga klas dziedziczących po TestCase", "działa tylko w PyCharm"]),
            open_question("Wyjaśnij różnicę między dziedziczeniem a kompozycją na własnym przykładzie."),
        ]},
    },
    # ------------------------------------------------------------------ 3
    {
        "title": "Django od zera do aplikacji",
        "category": "Aplikacje webowe", "author": "instructor", "price": "199.00", "level": "intermediate",
        "description": "Zbuduj aplikację webową w Django: modele, widoki, szablony, formularze i panel admina.",
        "about": "<p>Krok po kroku tworzymy aplikację z listą zadań — od instalacji po wdrożenie.</p>",
        "sections": [
            {"title": "Start projektu", "lessons": [
                lesson("Instalacja i pierwszy projekt", 15, preview=True, video="youtube:F5mRW0jo-U4", html=
                       "<pre><code class=\"language-bash\">python -m venv .venv\nsource .venv/bin/activate\npip install django\ndjango-admin startproject config .\npython manage.py runserver</code></pre>"
                       "<p>Film (po angielsku) to pełny kurs — pierwsze kroki znajdziesz w pierwszych 20 minutach.</p>"),
                lesson("Aplikacje i adresy URL", 14,
                       "<pre><code class=\"language-python\"># zadania/urls.py\nfrom django.urls import path\nfrom . import views\n\nurlpatterns = [path(\"\", views.lista, name=\"lista\")]</code></pre>"),
            ]},
            {"title": "Dane i widoki", "lessons": [
                lesson("Modele i migracje", 20,
                       "<pre><code class=\"language-python\">class Zadanie(models.Model):\n    tytul = models.CharField(max_length=200)\n    zrobione = models.BooleanField(default=False)</code></pre>"
                       "<pre><code class=\"language-bash\">python manage.py makemigrations\npython manage.py migrate</code></pre>"),
                lesson("Widoki i szablony", 22,
                       "<pre><code class=\"language-python\">def lista(request):\n    return render(request, \"zadania/lista.html\", {\"zadania\": Zadanie.objects.all()})</code></pre>",
                       files=[("lista.html",
                               "{% extends \"base.html\" %}\n{% block content %}\n<h1>Zadania</h1>\n<ul>\n"
                               "  {% for z in zadania %}<li>{{ z.tytul }}</li>{% empty %}<li>Brak zadań</li>{% endfor %}\n</ul>\n{% endblock %}\n")]),
                lesson("Formularze i walidacja", 24,
                       "<p><code>ModelForm</code> buduje formularz z modelu i sprawdza dane. Po błędzie formularz "
                       "zachowuje wpisane wartości — użytkownik nie musi wpisywać wszystkiego od nowa.</p>"),
            ]},
            {"title": "Panel admina i wdrożenie", "lessons": [
                lesson("Panel administracyjny", 12,
                       "<pre><code class=\"language-python\">from django.contrib import admin\nadmin.site.register(Zadanie)</code></pre>"),
                lesson("Docker dla aplikacji Django", 18,
                       "<p>Pobierz gotowy <code>Dockerfile</code> i uruchom aplikację w kontenerze.</p>",
                       files=[("Dockerfile",
                               "FROM python:3.13-slim\nWORKDIR /app\nCOPY requirements.txt .\nRUN pip install -r requirements.txt\n"
                               "COPY . .\nCMD [\"python\", \"manage.py\", \"runserver\", \"0.0.0.0:8000\"]\n")]),
            ]},
        ],
        "quiz": {"title": "Test: Django", "questions": [
            single("Które polecenie tworzy pliki migracji?", "makemigrations", "migrate", "startapp", "collectstatic"),
            single("Gdzie definiuje się pola bazy danych?", "w modelu", "w szablonie", "w urls.py", "w settings.py"),
            multiple("Co chroni formularze przed CSRF w Django?", ["tag {% csrf_token %}", "middleware CsrfViewMiddleware"],
                     ["metoda GET", "plik robots.txt"]),
            open_question("Opisz drogę żądania HTTP od adresu URL do wyrenderowanego szablonu."),
        ]},
    },
    # ------------------------------------------------------------------ 4
    {
        "title": "HTML i CSS: dostępne strony od podstaw",
        "category": "Aplikacje webowe", "author": "instructor", "price": "0.00", "level": "beginner",
        "description": "Semantyczny HTML, nowoczesny CSS i zasady WCAG — strony, które działają dla każdego.",
        "about": "<p>Uczysz się od razu dobrych nawyków: nagłówki, etykiety formularzy, kontrast i obsługa klawiaturą.</p>",
        "sections": [
            {"title": "HTML", "lessons": [
                lesson("Struktura dokumentu", 14, preview=True, video="youtube:UB1O30fR-EE", html=
                       "<pre><code class=\"language-html\">&lt;!DOCTYPE html&gt;\n&lt;html lang=\"pl\"&gt;\n&lt;head&gt;&lt;title&gt;Moja strona&lt;/title&gt;&lt;/head&gt;\n&lt;body&gt;\n  &lt;main&gt;&lt;h1&gt;Witaj&lt;/h1&gt;&lt;/main&gt;\n&lt;/body&gt;\n&lt;/html&gt;</code></pre>"),
                lesson("Formularze z etykietami", 16,
                       "<p>Każde pole potrzebuje etykiety powiązanej atrybutem <code>for</code>:</p>"
                       "<pre><code class=\"language-html\">&lt;label for=\"email\"&gt;Adres e-mail&lt;/label&gt;\n&lt;input id=\"email\" type=\"email\" autocomplete=\"email\"&gt;</code></pre>"),
            ]},
            {"title": "CSS", "lessons": [
                lesson("Selektory i kaskada", 18, video="youtube:yfoY53QXEnI",
                       html="<p>CSS dobiera styl do elementów selektorami. Gdy dwie reguły się kłócą, wygrywa ta bardziej szczegółowa.</p>"),
                lesson("Flexbox i Grid", 22, video="youtube:1Rs2ND1ryYc",
                       html="<pre><code class=\"language-css\">.siatka {\n  display: grid;\n  grid-template-columns: repeat(auto-fit, minmax(16rem, 1fr));\n  gap: 1rem;\n}</code></pre>"),
            ]},
            {"title": "Dostępność", "lessons": [
                lesson("Fokus i klawiatura", 16,
                       "<p>Nigdy nie usuwaj obrysu fokusu bez zamiennika:</p>"
                       "<pre><code class=\"language-css\">:focus-visible { outline: 3px solid #B45309; outline-offset: 2px; }</code></pre>",
                       image=("Przycisk z widocznym obrysem fokusu", "Fokus musi być widoczny",
                              ["[ Zapisz ]  <- Tab", "outline: 3px solid", "kontrast >= 3:1"])),
                lesson("Zadanie: popraw formularz", 20, files=[(
                    "formularz.html",
                    "<form>\n  <input type=\"text\" placeholder=\"Imię\">\n  <input type=\"email\" placeholder=\"E-mail\">\n"
                    "  <div onclick=\"wyslij()\">Wyślij</div>\n</form>\n")], html=
                       "<p>Formularz z pliku ma trzy problemy z dostępnością. Dodaj etykiety, zamień <code>div</code> "
                       "na prawdziwy przycisk i sprawdź, czy całość da się obsłużyć klawiaturą.</p>"),
            ]},
        ],
        "quiz": {"title": "Test: HTML, CSS i dostępność", "questions": [
            single("Który atrybut wiąże etykietę z polem?", "for", "name", "href", "alt"),
            single("Jaki minimalny kontrast wymaga WCAG AA dla zwykłego tekstu?", "4.5:1", "2:1", "3:1", "7:1"),
            multiple("Które elementy są interaktywne z klawiatury bez dodatkowego kodu?", ["button", "a z atrybutem href"],
                     ["div", "span"]),
            open_question("Wyjaśnij, dlaczego placeholder nie zastępuje etykiety pola."),
        ]},
    },
    # ------------------------------------------------------------------ 5
    {
        "title": "JavaScript w przeglądarce",
        "category": "Aplikacje webowe", "author": "instructor", "price": "99.00", "level": "beginner",
        "description": "Zmienne, funkcje, DOM i zdarzenia — interaktywne strony bez frameworków.",
        "about": "<p>Piszesz małe, dostępne komponenty: licznik, przełącznik i modal zamykany klawiszem Esc.</p>",
        "sections": [
            {"title": "Podstawy języka", "lessons": [
                lesson("Zmienne i funkcje", 16, preview=True, video="youtube:W6NZfCO5SIk", html=
                       "<pre><code class=\"language-javascript\">const imie = \"Ola\";\nlet licznik = 0;\nfunction przywitaj(osoba) {\n  return `Cześć, ${osoba}!`;\n}</code></pre>"),
                lesson("Tablice i obiekty", 18,
                       "<pre><code class=\"language-javascript\">const kursy = [{ tytul: \"Python\", cena: 0 }, { tytul: \"Django\", cena: 199 }];\nconst darmowe = kursy.filter((k) =&gt; k.cena === 0);</code></pre>"),
            ]},
            {"title": "DOM i zdarzenia", "lessons": [
                lesson("Wyszukiwanie elementów", 14,
                       "<pre><code class=\"language-javascript\">const przycisk = document.querySelector(\"#zapisz\");\nprzycisk.textContent = \"Zapisano\";</code></pre>"),
                lesson("Modal zamykany klawiszem Esc", 22,
                       "<p>Natywny element <code>&lt;dialog&gt;</code> zamyka się Esc i trzyma fokus w środku.</p>",
                       files=[("modal.html",
                               "<button id=\"otworz\">Otwórz</button>\n<dialog id=\"okno\">\n  <h2>Okno</h2>\n"
                               "  <button id=\"zamknij\">Zamknij</button>\n</dialog>\n<script>\n"
                               "const okno = document.getElementById(\"okno\");\n"
                               "document.getElementById(\"otworz\").onclick = () => okno.showModal();\n"
                               "document.getElementById(\"zamknij\").onclick = () => okno.close();\n</script>\n")]),
            ]},
        ],
        "quiz": {"title": "Test: JavaScript", "questions": [
            single("Które słowo deklaruje stałą?", "const", "let", "var", "static"),
            single("Która metoda zwraca pierwszy pasujący element?", "querySelector", "getAll", "findElement", "select"),
            open_question("Opisz, jak sprawić, by własny modal zamykał się klawiszem Esc i oddawał fokus."),
        ]},
    },
    # ------------------------------------------------------------------ 6
    {
        "title": "Analiza danych w pandas",
        "category": "Analiza danych", "author": "m.wisniewska", "price": "149.00", "level": "intermediate",
        "description": "Wczytywanie, czyszczenie i grupowanie danych w pandas na prawdziwym zbiorze sprzedaży.",
        "about": "<p>Pracujemy na jednym zbiorze danych od pierwszej do ostatniej lekcji.</p>",
        "sections": [
            {"title": "Wczytywanie danych", "lessons": [
                lesson("DataFrame w pigułce", 18, preview=True, video="youtube:vmEHCJofslg", html=
                       "<pre><code class=\"language-python\">import pandas as pd\n\ndf = pd.read_csv(\"sprzedaz.csv\")\nprint(df.head())\nprint(df.describe())</code></pre>",
                       files=[("sprzedaz.csv",
                               "data,miasto,produkt,ilosc,cena\n2026-01-03,Kraków,kurs Python,3,0\n2026-01-03,Warszawa,kurs Django,2,199\n"
                               "2026-01-04,Gdańsk,kurs pandas,5,149\n2026-01-05,Kraków,kurs pandas,1,149\n2026-01-05,Wrocław,kurs Django,4,199\n"
                               "2026-01-06,Warszawa,kurs Python,6,0\n")]),
                lesson("Czyszczenie danych", 20,
                       "<pre><code class=\"language-python\">df = df.dropna()\ndf[\"data\"] = pd.to_datetime(df[\"data\"])\ndf[\"przychod\"] = df[\"ilosc\"] * df[\"cena\"]</code></pre>"),
            ]},
            {"title": "Analiza", "lessons": [
                lesson("Grupowanie i agregacja", 22,
                       "<pre><code class=\"language-python\">df.groupby(\"miasto\")[\"przychod\"].sum().sort_values(ascending=False)</code></pre>",
                       image=("Wykres słupkowy przychodu w miastach", "Przychód według miast",
                              ["Wrocław   ███████ 796", "Warszawa  ████ 398", "Gdańsk    ████ 745", "Kraków    █ 149"])),
                lesson("Zadanie: raport sprzedaży", 25,
                       "<p>Na podstawie pliku <code>sprzedaz.csv</code> z pierwszej lekcji przygotuj raport: najlepszy produkt, "
                       "najlepsze miasto i przychód dzienny. Wynik zapisz do pliku Excel metodą <code>to_excel</code>.</p>"),
            ]},
        ],
        "quiz": {"title": "Test: pandas", "questions": [
            single("Która funkcja wczytuje plik CSV?", "pd.read_csv", "pd.open_csv", "pd.load", "pd.csv"),
            single("Co zwraca df.head()?", "pierwsze wiersze", "nagłówki kolumn", "ostatni wiersz", "liczbę wierszy"),
            multiple("Które metody służą do agregacji?", ["sum", "mean"], ["head", "dropna"]),
            open_question("Opisz, jak znajdziesz miasto z największym przychodem w zbiorze sprzedaz.csv."),
        ]},
    },
    # ------------------------------------------------------------------ 7
    {
        "title": "Wprowadzenie do Machine Learning",
        "category": "Uczenie maszynowe", "author": "m.wisniewska", "price": "149.99", "level": "intermediate",
        "description": "Podstawy uczenia maszynowego z wykorzystaniem scikit-learn.",
        "about": "<p>Od intuicji do pierwszego modelu klasyfikacji i oceny jego jakości.</p>",
        "sections": [
            {"title": "Intuicja", "lessons": [
                lesson("Czym jest uczenie maszynowe", 15, preview=True, video="youtube:Gv9_4yMHFhI", html=
                       "<p>Model uczy się zależności z przykładów, zamiast wykonywać ręcznie zapisane reguły. "
                       "Film StatQuest (po angielsku) świetnie tłumaczy intuicję.</p>"),
                lesson("Zbiór treningowy i testowy", 16,
                       "<pre><code class=\"language-python\">from sklearn.model_selection import train_test_split\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)</code></pre>"),
            ]},
            {"title": "Pierwszy model", "lessons": [
                lesson("Klasyfikacja drzewem decyzyjnym", 22, video="youtube:ukzFI9rgwfU",
                       html="<pre><code class=\"language-python\">from sklearn.tree import DecisionTreeClassifier\nmodel = DecisionTreeClassifier(max_depth=3).fit(X_train, y_train)\nprint(model.score(X_test, y_test))</code></pre>"),
                lesson("Ocena modelu", 18,
                       "<p>Sama dokładność bywa myląca. Sprawdź macierz pomyłek, precyzję i czułość.</p>",
                       image=("Macierz pomyłek 2 na 2", "Macierz pomyłek",
                              ["            przew. TAK  przew. NIE", "fakt TAK        42          8", "fakt NIE         5         45"])),
            ]},
        ],
        "quiz": {"title": "Test: podstawy ML", "questions": [
            single("Po co dzieli się dane na treningowe i testowe?", "żeby uczciwie ocenić model",
                   "żeby szybciej trenować", "żeby zmniejszyć plik", "nie ma takiej potrzeby"),
            multiple("Które miary oceniają klasyfikację?", ["precyzja", "czułość"], ["średnia arytmetyczna wieku", "liczba kolumn"]),
            open_question("Podaj przykład problemu, w którym dokładność 95% może być złym wynikiem."),
        ]},
    },
    # ------------------------------------------------------------------ 8
    {
        "title": "Docker i Kubernetes w praktyce",
        "category": "Konteneryzacja", "author": "p.zielinski", "price": "199.99", "level": "advanced",
        "description": "Konteneryzacja aplikacji i orkiestracja w chmurze.",
        "about": "<p>Od pierwszego obrazu do wdrożenia na klastrze Kubernetes.</p>",
        "sections": [
            {"title": "Docker", "lessons": [
                lesson("Obrazy i kontenery", 18, preview=True, video="youtube:pTFZFxd4hOI", html=
                       "<pre><code class=\"language-bash\">docker run --rm hello-world\ndocker ps -a\ndocker images</code></pre>"),
                lesson("Docker Compose", 22, files=[(
                    "docker-compose.yml",
                    "services:\n  db:\n    image: postgres:16-alpine\n    environment:\n      POSTGRES_PASSWORD: haslo\n"
                    "  web:\n    build: .\n    ports:\n      - \"8000:8000\"\n    depends_on:\n      - db\n")], html=
                       "<p>Compose uruchamia kilka kontenerów jednym poleceniem <code>docker compose up</code>. "
                       "Pobierz przykładowy plik i dopisz usługę Redis.</p>"),
            ]},
            {"title": "Kubernetes", "lessons": [
                lesson("Pody, deploymenty i serwisy", 24, video="youtube:s_o8dwzRlu4",
                       html="<pre><code class=\"language-bash\">kubectl create deployment web --image=nginx\nkubectl expose deployment web --port=80\nkubectl get pods</code></pre>"),
                lesson("Skalowanie aplikacji", 16,
                       "<pre><code class=\"language-bash\">kubectl scale deployment web --replicas=3</code></pre>"),
            ]},
        ],
        "quiz": {"title": "Test: Docker i Kubernetes", "questions": [
            single("Które polecenie pokazuje działające kontenery?", "docker ps", "docker show", "docker list", "docker run"),
            single("Co w Kubernetes zapewnia stały adres dla podów?", "Service", "ConfigMap", "Secret", "Volume"),
            open_question("Wyjaśnij różnicę między obrazem a kontenerem."),
            open_question("Wyślij swój Dockerfile dla aplikacji z lekcji (plik Dockerfile lub .txt).", fmt="file", points=3),
        ]},
    },
    # ------------------------------------------------------------------ 9
    {
        "title": "Linux i terminal na co dzień",
        "category": "Linux", "author": "p.zielinski", "price": "0.00", "level": "beginner",
        "certificate": False,  # kurs bez dyplomu — ukończenie bez PDF
        "description": "Pliki, uprawnienia, procesy i skrypty — terminal bez strachu.",
        "about": "<p>Praktyczne polecenia, których programista używa każdego dnia.</p>",
        "sections": [
            {"title": "Pierwsze polecenia", "lessons": [
                lesson("Poruszanie się po katalogach", 12, preview=True, video="file:linux_pierwsze_polecenia.webm", html=
                       "<pre><code class=\"language-bash\">pwd\nls -l\ncd ~/projekty\nmkdir nowy &amp;&amp; cd nowy</code></pre>"
                       "<p>Film pokazuje te polecenia w działaniu.</p>"),
                lesson("Kopiowanie, przenoszenie, usuwanie", 14,
                       "<pre><code class=\"language-bash\">cp notatki.txt kopia.txt\nmv kopia.txt archiwum/\nrm -i archiwum/kopia.txt</code></pre>"),
            ]},
            {"title": "System", "lessons": [
                lesson("Uprawnienia plików", 16, video="youtube:ROjZy1WbCIA",
                       html="<pre><code class=\"language-bash\">chmod 750 skrypt.sh\nls -l skrypt.sh  # -rwxr-x---</code></pre>"),
                lesson("Zadanie: pierwszy skrypt", 18, files=[(
                    "backup.sh",
                    "#!/usr/bin/env bash\n# Zadanie: dodaj date do nazwy archiwum i sprawdz, czy katalog istnieje.\n"
                    "tar -czf backup.tar.gz \"$1\"\necho \"Gotowe\"\n")], html=
                       "<p>Pobierz skrypt <code>backup.sh</code>, nadaj mu prawa wykonywania i rozbuduj go zgodnie z komentarzem.</p>"),
            ]},
        ],
        "quiz": {"title": "Test: Linux", "questions": [
            single("Które polecenie pokazuje bieżący katalog?", "pwd", "cd", "ls", "whoami"),
            single("Co oznacza chmod 750?", "rwx dla właściciela, r-x dla grupy, brak dla innych",
                   "pełne prawa dla wszystkich", "tylko odczyt dla wszystkich", "usunięcie pliku"),
            open_question("Opisz, co robi polecenie: grep -r \"TODO\" . | wc -l"),
        ]},
    },
    # ------------------------------------------------------------------ 10
    {
        "title": "Projektowanie UX: od badań do prototypu",
        "category": "Projektowanie UX/UI", "author": "k.lewandowska", "price": "179.00", "level": "beginner",
        "description": "Badania z użytkownikami, architektura informacji, makiety i testy użyteczności.",
        "about": "<p>Projektujesz małą aplikację od rozmów z użytkownikami po klikalny prototyp.</p>",
        "sections": [
            {"title": "Badania", "lessons": [
                lesson("Wywiad z użytkownikiem", 16, preview=True,
                       html="<p>Pytaj o konkretne sytuacje z przeszłości, nie o opinie. Zamiast „Czy kupiłbyś…?” "
                            "zapytaj „Opowiedz, kiedy ostatnio…”.</p>",
                       files=[("scenariusz_wywiadu.md",
                               "# Scenariusz wywiadu\n\n1. Opowiedz o ostatnim kursie online, który zaczęłaś/zacząłeś.\n"
                               "2. Co sprawiło, że go skończyłaś/skończyłeś (albo przerwałaś/przerwałeś)?\n"
                               "3. Jak wybierasz, czego uczyć się dalej?\n")]),
                lesson("Persony i mapa podróży", 18,
                       "<p>Persona streszcza wnioski z badań. Mapa podróży pokazuje, gdzie użytkownik napotyka trudności.</p>",
                       image=("Mapa podróży ucznia w pięciu krokach", "Mapa podróży ucznia",
                              ["1 Szuka kursu", "2 Porównuje", "3 Kupuje", "4 Uczy się  <- tu odpada", "5 Kończy"])),
            ]},
            {"title": "Projektowanie", "lessons": [
                lesson("Makiety i prototyp", 22, video="youtube:c9Wg6Cb_YlU",
                       html="<p>Zacznij od szkiców na papierze, potem makieta szarościami, a dopiero na końcu kolory.</p>"),
                lesson("Test użyteczności", 20,
                       "<p>Pięciu uczestników wystarczy, by znaleźć większość problemów. Daj zadanie, obserwuj i nie podpowiadaj.</p>"),
            ]},
        ],
        "quiz": {"title": "Test: podstawy UX", "questions": [
            single("Ile osób zwykle wystarcza do jakościowego testu użyteczności?", "około 5", "1", "50", "500"),
            multiple("Które pytania są dobre w wywiadzie?", ["Opowiedz, kiedy ostatnio…", "Co wtedy zrobiłeś/zrobiłaś?"],
                     ["Czy podoba Ci się nasz pomysł?", "Kupiłbyś to?"]),
            open_question("Zaprojektuj krótkie zadanie do testu użyteczności strony z katalogiem kursów."),
            open_question("Narysuj na kartce szkic ekranu logowania i prześlij jego zdjęcie.", fmt="image", points=3),
        ]},
    },
    # -------------------------------------------- szkic nauczycielki (widzi go tylko autorka)
    {
        "title": "Dostępność stron w praktyce",
        "category": "Projektowanie UX/UI", "author": "k.lewandowska", "price": "0.00", "level": "intermediate",
        "status": "draft",
        "description": "Audyt WCAG krok po kroku: klawiatura, kontrast, czytniki ekranu.",
        "about": "<p>Szkic kursu w przygotowaniu — przykład kroku „Program” przed publikacją.</p>",
        "sections": [
            {"title": "Audyt klawiaturą", "lessons": [
                lesson("Przejdź stronę samym Tabem", 10,
                       "<p>Odłóż mysz i przejdź całą stronę klawiszem Tab. Zanotuj miejsca, w których nie widać fokusu.</p>"),
            ]},
        ],
    },
    # -------------------------------------------- zgłoszenia uczniów (niepubliczne)
    {
        "title": "Git i GitHub w pracy zespołowej",
        "category": "Programowanie", "author": "ola.krawczyk", "price": "0.00", "level": "beginner",
        "status": "pending",
        "description": "Gałęzie, pull requesty i rozwiązywanie konfliktów — propozycja kursu od uczennicy.",
        "about": "<p>Kurs zgłoszony przez uczennicę, czeka na akceptację nauczyciela.</p>",
        "messages": [
            ("instructor", "Dzięki za zgłoszenie! Dodasz lekcję o rozwiązywaniu konfliktów przy merge?"),
            ("ola.krawczyk", "Jasne, dopiszę ją do sekcji „Gałęzie” do końca tygodnia."),
        ],
        "sections": [
            {"title": "Gałęzie", "lessons": [
                lesson("Tworzenie gałęzi", 12, "<pre><code class=\"language-bash\">git switch -c nowa-funkcja</code></pre>"),
            ]},
        ],
    },
    {
        "title": "Excel dla każdego",
        "category": "Analiza danych", "author": "michal.kaczmarek", "price": "49.00", "level": "beginner",
        "status": "rejected",
        "rejection_reason": "Kurs dubluje tematykę „Analizy danych w pandas” i nie ma jeszcze programu lekcji.",
        "description": "Formuły, tabele przestawne i wykresy.",
        "about": "<p>Propozycja odrzucona — przykład zgłoszenia z powodem odrzucenia.</p>",
        "sections": [],
    },
]

# ---------------------------------------------------------------------------
# Zapisy na kursy: (uczeń, kurs, ile pierwszych lekcji ukończonych | "all")
# Płatne kursy — tak jakby zamówienie było opłacone (koszyk dojdzie później).
# ---------------------------------------------------------------------------

ENROLLMENTS = [
    ("student1", "Python dla początkujących", 5),
    ("student1", "HTML i CSS: dostępne strony od podstaw", "all"),
    ("student1", "Django od zera do aplikacji", 2),
    ("student2", "Python dla początkujących", "all"),
    ("student2", "Docker i Kubernetes w praktyce", 1),
    ("ola.krawczyk", "Analiza danych w pandas", 3),
    ("ola.krawczyk", "Python dla początkujących", 9),
    ("michal.kaczmarek", "Linux i terminal na co dzień", "all"),
    ("michal.kaczmarek", "Python dla zaawansowanych: klasy, wyjątki i testy", 2),
    ("zuzia.mazur", "Projektowanie UX: od badań do prototypu", 2),
    ("zuzia.mazur", "HTML i CSS: dostępne strony od podstaw", 4),
    ("kuba.dabrowski", "JavaScript w przeglądarce", 1),
    ("kuba.dabrowski", "Python dla początkujących", 0),
    ("natalia.pawlak", "Wprowadzenie do Machine Learning", 3),
    ("natalia.pawlak", "Analiza danych w pandas", "all"),
    ("bartek.sikora", "Linux i terminal na co dzień", 2),
]

# ---------------------------------------------------------------------------
# Recenzje: (autor, kurs, ocena 1–5, treść, zatwierdzona?)
# ---------------------------------------------------------------------------

REVIEWS = [
    ("student2", "Python dla początkujących", 5, "Świetny start — zadanie z poprawianiem pliku bardzo pomogło.", True),
    ("ola.krawczyk", "Python dla początkujących", 4, "Jasno wytłumaczone pętle. Przydałoby się więcej zadań.", True),
    ("student1", "HTML i CSS: dostępne strony od podstaw", 5, "Pierwszy kurs, który uczy dostępności od pierwszej lekcji.", True),
    ("zuzia.mazur", "HTML i CSS: dostępne strony od podstaw", 5, "Zadanie z formularzem otworzyło mi oczy.", True),
    ("michal.kaczmarek", "Linux i terminal na co dzień", 4, "Konkretnie i praktycznie. Skrypt backup.sh — super ćwiczenie.", True),
    ("natalia.pawlak", "Analiza danych w pandas", 5, "Jeden zbiór danych przez cały kurs to dobry pomysł.", True),
    ("natalia.pawlak", "Wprowadzenie do Machine Learning", 4, "Macierz pomyłek w końcu zrozumiała.", True),
    ("student2", "Docker i Kubernetes w praktyce", 3, "Dobre, ale tempo w części o Kubernetes jest szybkie.", True),
    ("kuba.dabrowski", "Python dla początkujących", 2, "Za łatwe jak dla mnie.", False),  # czeka na akceptację
    # Zwykły komentarz (pytanie) bez oceny kursu
    ("bartek.sikora", "Linux i terminal na co dzień", None, "Czy polecenia z kursu zadziałają też w terminalu na macOS?", True),
]

# ---------------------------------------------------------------------------
# Wątki odpowiedzi: (kurs, autor komentarza głównego, [(autor, treść, [odpowiedzi…]), …]).
# Wątek w Pythonie ma 6 poziomów — widok wcina 4, głębsze pokazuje płasko z dopiskiem
# „w odpowiedzi do …”. Wszystkie odpowiedzi są zatwierdzone.
# ---------------------------------------------------------------------------

COMMENT_THREADS = [
    ("Python dla początkujących", "student2", [
        ("instructor", "Dziękuję! W kolejnej wersji dodam więcej zadań na plikach.", [
            ("student2", "Super — może coś z plikami CSV?", [
                ("ola.krawczyk", "Popieram, CSV na start byłby idealny.", [
                    ("instructor", "Zapisane: zadanie z CSV trafi do sekcji 2.", [
                        ("student2", "Dziękuję, czekam!", []),
                    ]),
                ]),
            ]),
        ]),
        ("zuzia.mazur", "Zgadzam się, zadanie z plikiem było najlepsze w całym kursie.", []),
    ]),
    ("Python dla początkujących", "ola.krawczyk", [
        ("kuba.dabrowski", "Zadania są w lekcji 5, wystarczy przewinąć do końca.", [
            ("ola.krawczyk", "Faktycznie, przeoczyłam. Dzięki!", []),
        ]),
    ]),
    ("Docker i Kubernetes w praktyce", "student2", [
        ("p.zielinski", "Dzięki za uwagę — rozbiję część o Kubernetes na dwie lekcje.", []),
    ]),
    ("Linux i terminal na co dzień", "bartek.sikora", [
        ("p.zielinski", "Tak, prawie wszystkie. Różnice (np. sed -i) opisuję w lekcji 3.", []),
    ]),
]

# Testy: podejścia demonstracyjne. Uczniowie z ukończonymi lekcjami mają zaliczone testy stojące
# po tych lekcjach (i końcowe, gdy przeszli cały kurs) — wtedy kurs z dyplomem wystawia PDF.
# Wyjątki: (uczeń, kurs, tytuł testu)
PENDING_ATTEMPTS = [  # czeka na ocenę nauczyciela (kolejka „Do oceny” konta instructor)
    ("ola.krawczyk", "Python dla początkujących", "Test: podstawy Pythona"),
]
FAILED_ATTEMPTS = [  # niezaliczone podejście
    ("kuba.dabrowski", "Python dla początkujących", "Sprawdź się: pierwszy program"),
    ("student2", "Docker i Kubernetes w praktyce", "Test: Docker i Kubernetes"),  # analiza stylów
]

# Głosy na odpowiedzi: (głosujący, kurs, autor odpowiedzi, początek treści, +1/−1)
REPLY_VOTES = [
    ("student1", "Python dla początkujących", "instructor", "Dziękuję! W kolejnej", 1),
    ("zuzia.mazur", "Python dla początkujących", "instructor", "Dziękuję! W kolejnej", 1),
    ("kuba.dabrowski", "Python dla początkujących", "student2", "Super — może", -1),
    ("student1", "Python dla początkujących", "kuba.dabrowski", "Zadania są w lekcji 5", 1),
    ("michal.kaczmarek", "Linux i terminal na co dzień", "p.zielinski", "Tak, prawie wszystkie", 1),
]

# Głosy pomocności: (głosujący, autor recenzji, kurs, +1/−1) → reputacja i rangi
REVIEW_VOTES = [
    (voter, "natalia.pawlak", course, 1)
    for voter in ["student1", "student2", "ola.krawczyk", "michal.kaczmarek", "zuzia.mazur", "kuba.dabrowski", "bartek.sikora"]
    for course in ["Analiza danych w pandas", "Wprowadzenie do Machine Learning"]
] + [
    (voter, "student2", "Python dla początkujących", 1) for voter in ["student1", "ola.krawczyk", "zuzia.mazur"]
] + [
    ("student1", "student2", "Docker i Kubernetes w praktyce", -1),
    ("bartek.sikora", "michal.kaczmarek", "Linux i terminal na co dzień", 1),
]

# ---------------------------------------------------------------------------
# Powiadomienia: (użytkownik, rodzaj Notification.Kind, tytuł, treść, przeczytane?)
# ---------------------------------------------------------------------------

NOTIFICATIONS = [
    ("student2", "comment_approved", "Twój komentarz został zaakceptowany",
     "Recenzja kursu „Python dla początkujących” jest już widoczna.", False),
    ("ola.krawczyk", "submission_message", "Nowa wiadomość od moderatora",
     "Jan Kowalski odpisał w zgłoszeniu „Git i GitHub w pracy zespołowej”.", False),
    ("michal.kaczmarek", "submission_status", "Zgłoszenie kursu odrzucone",
     "Kurs „Excel dla każdego” został odrzucony. Sprawdź powód w szczegółach kursu.", False),
    ("instructor", "course_submitted", "Nowe zgłoszenie kursu",
     "Aleksandra Krawczyk: „Git i GitHub w pracy zespołowej” w kategorii Programowanie.", False),
    ("instructor", "comment_pending", "Komentarz czeka na akceptację",
     "Nowa recenzja kursu „Python dla początkujących” od kuba.dabrowski.", False),
    ("instructor", "quiz_to_grade", "Nowy test do sprawdzenia",
     "Anna Nowak przesłała odpowiedź otwartą w teście „Test: podstawy Pythona”.", False),
    ("student1", "system", "Witaj na platformie!", "Zacznij od kursu „Python dla początkujących”.", True),
]


# ---------------------------------------------------------------------------
# Galeria: zdjęcia (generowane 2400×1600 — przy zapisie skalowane do 1600×1067),
# opisy różnej długości (karty wyrównują się do najwyższej) i zapisane układy slajdera.
# (tytuł, opis, tekst alternatywny, kurs lub None, kolor tła RGB, kolor akcentu RGB)
# ---------------------------------------------------------------------------

GALLERY = [
    ("Jesienna promocja", "Wszystkie kursy programowania 30% taniej do końca listopada.",
     "Plakat promocji: duży napis minus 30 procent na pomarańczowym tle", None, (180, 83, 9), (255, 251, 235)),
    ("Python od zera", "Pierwszy program w 15 minut. Kurs prowadzi Jan Kowalski — od instalacji po własne funkcje, "
     "z testem obowiązkowym w połowie i dyplomem na końcu.",
     "Okno terminala z poleceniem print i napisem Python", "Python dla początkujących", (17, 24, 39), (217, 119, 6)),
    ("Dostępne strony WWW", "Uczymy HTML i CSS tak, żeby strona działała z klawiaturą i czytnikiem ekranu.\n\n"
     "W kursie: etykiety formularzy, widoczny fokus, kontrast i modale zamykane klawiszem Esc. Każda lekcja kończy się "
     "krótkim zadaniem, które sprawdzisz narzędziem Lighthouse.",
     "Szkic formularza z zaznaczonymi etykietami pól", "HTML i CSS: dostępne strony od podstaw", (15, 118, 110), (240, 253, 250)),
    ("Docker w praktyce", "Kontenery bez tajemnic.",
     "Ikona kontenera na niebieskim tle", "Docker i Kubernetes w praktyce", (30, 64, 175), (219, 234, 254)),
    ("Analiza danych w pandas", "Jeden zbiór danych przez cały kurs: od wczytania pliku CSV, przez czyszczenie i grupowanie, "
     "aż po wykresy, które pokażesz w raporcie. Idealny start przed uczeniem maszynowym.",
     "Wykres słupkowy sprzedaży w kolejnych miesiącach", "Analiza danych w pandas", (88, 28, 135), (243, 232, 255)),
    ("Spotkanie społeczności", "Co miesiąc spotykamy się online: uczniowie pokazują projekty, a nauczyciele odpowiadają na pytania. "
     "Najbliższe spotkanie w czwartek o 18:00.",
     "Grupa osób przy laptopach podczas spotkania", None, (22, 101, 52), (220, 252, 231)),
    ("Nowość: testy z dyplomem", "Po ukończeniu kursu pobierzesz dyplom w PDF.",
     "Dyplom z wstążką i napisem E-Learning", None, (127, 29, 29), (254, 226, 226)),
]

# (nazwa, wyświetlany?, przewijanie automatyczne?, sekundy, [tytuły zdjęć w kolejności])
SLIDER_PRESETS = [
    ("Polecane na start", True, True, 6, ["Python od zera", "Dostępne strony WWW", "Analiza danych w pandas",
                                         "Nowość: testy z dyplomem", "Spotkanie społeczności"]),
    ("Promocja jesienna", False, True, 5, ["Jesienna promocja", "Docker w praktyce", "Python od zera", "Analiza danych w pandas"]),
    ("Kursy programowania", False, False, 8, ["Python od zera", "Docker w praktyce", "Analiza danych w pandas",
                                             "Dostępne strony WWW", "Jesienna promocja", "Spotkanie społeczności",
                                             "Nowość: testy z dyplomem"]),
]


# ---------------------------------------------------------------------------
# Asystent AI: przykładowe rozmowy (historia widoczna bez uruchomionej Ollamy).
# (login, tytuł kursu lub None = rozmowa ogólna, [(nadawca, treść, tytuł lekcji lub None, odmowa w teście)])
# ---------------------------------------------------------------------------

CHAT_CONVERSATIONS = [
    ("student1", "Python dla początkujących", [
        ("user", "Czym różni się int od float?", "Zmienne i typy danych", False),
        ("bot", "**int** to liczby całkowite, np. `7` albo `-3`. **float** to liczby z częścią ułamkową, "
                "np. `3.14`. Zobacz różnicę przy dzieleniu:\n\n```python\nprint(7 // 2)  # 3 — int\n"
                "print(7 / 2)   # 3.5 — float\n```\n\nSpróbuj sam: jaki typ zwróci `type(10 / 5)`?",
         "Zmienne i typy danych", False),
        ("user", "Która odpowiedź w pytaniu 2 jest dobra?", None, True),
        ("bot", "Trwa test „Test: podstawy Pythona”. Podczas testu nie mogę udzielać pomocy — rozwiąż go "
                "samodzielnie. Gdy wyślesz odpowiedzi, chętnie wyjaśnię pytania, z którymi był problem: "
                "otwórz wtedy wynik testu i zapytaj mnie o nie.", None, True),
    ]),
    ("student1", None, [
        ("user", "Jak najlepiej uczyć się programowania?", None, False),
        ("bot", "Kilka sprawdzonych zasad:\n\n- pisz kod codziennie, nawet 20 minut,\n"
                "- po każdej lekcji zrób zadanie bez podglądania rozwiązania,\n"
                "- czytaj komunikaty błędów do końca — zwykle mówią, co poprawić.\n\n"
                "Od którego kursu chcesz zacząć?", None, False),
    ]),
    ("student2", "Python dla początkujących", [
        ("user", "Wytłumacz jeszcze raz instrukcję if.", "Instrukcja if", False),
        ("bot", "`if` sprawdza warunek i wykonuje wcięty blok tylko wtedy, gdy warunek jest prawdziwy:\n\n"
                "```python\nwiek = 17\nif wiek >= 18:\n    print(\"Pełnoletni\")\nelse:\n    print(\"Niepełnoletni\")\n```\n\n"
                "Zwróć uwagę na dwukropek i wcięcie — bez nich Python zgłosi błąd.", "Instrukcja if", False),
    ]),
]


# ---------------------------------------------------------------------------
# Sklep: zamówienia w różnych statusach (opłacone tylko dla kursów, na które uczeń jest zapisany
# w ENROLLMENTS — dodatkowe sztuki to kody dostępu) i koszyk z kumulacją.
# (login, status, [(tytuł kursu, ilość)], notatka w historii)
# ---------------------------------------------------------------------------

ORDERS = [
    ("student1", "paid", [("Django od zera do aplikacji", 2)], "Płatność testowa Stripe (karta 4242…)."),
    ("student1", "pending_payment", [("JavaScript w przeglądarce", 1), ("Analiza danych w pandas", 1)], ""),
    ("student2", "completed", [("Docker i Kubernetes w praktyce", 1)], "Płatność testowa Stripe."),
    ("ola.krawczyk", "paid", [("Analiza danych w pandas", 1)], "Przelew tradycyjny — potwierdzony ręcznie."),
    ("natalia.pawlak", "paid", [("Wprowadzenie do Machine Learning", 1)], "Płatność testowa Stripe (BLIK)."),
    ("zuzia.mazur", "cancelled", [("Wprowadzenie do Machine Learning", 1)], "Anulowane przez kupującego."),
]

CARTS = [  # koszyk z dwiema sztukami tego samego kursu = kumulacja
    ("student2", [("JavaScript w przeglądarce", 2), ("Python dla zaawansowanych: klasy, wyjątki i testy", 1)]),
]

# ---------------------------------------------------------------------------
# Style uczenia się (style tworzy migracja learning.0006 — tu tylko ich slugi)
# ---------------------------------------------------------------------------

STUDENT_STYLES = {  # uczeń → wybrany styl (pozostali uczniowie nie wybrali stylu)
    "student1": "wzrokowiec",
    "student2": "sluchowiec",
    "ola.krawczyk": "czytanie-pisanie",
    "michal.kaczmarek": "praktyk",
    "zuzia.mazur": "wzrokowiec",
}

# Wersje lekcji dla stylów: (kurs, lekcja, styl, treść HTML, film YouTube lub "")
LESSON_VARIANTS = [
    ("Python dla początkujących", "Instalacja na Windows", "wzrokowiec",
     "<h3>Instalacja w trzech krokach</h3>"
     "<table><thead><tr><th>Krok</th><th>Co widzisz na ekranie</th><th>Co robisz</th></tr></thead><tbody>"
     "<tr><td>1</td><td>Strona python.org → Downloads</td><td>Pobierz instalator dla Windows</td></tr>"
     "<tr><td>2</td><td>Okno „Install Python 3.13”</td><td><strong>Zaznacz „Add python.exe to PATH”</strong></td></tr>"
     "<tr><td>3</td><td>Wiersz poleceń</td><td>Wpisz <code>python --version</code></td></tr>"
     "</tbody></table>"
     "<p>Jeśli w kroku 3 widzisz numer wersji, np. <code>Python 3.13.0</code>, instalacja się udała.</p>", ""),
    ("Python dla początkujących", "Instalacja na Windows", "praktyk",
     "<h3>Zrób to teraz — lista kontrolna</h3>"
     "<ol><li>Otwórz <a href=\"https://www.python.org/downloads/\">python.org/downloads</a> i pobierz instalator.</li>"
     "<li>Uruchom go i zaznacz <strong>Add python.exe to PATH</strong>.</li>"
     "<li>Kliknij <em>Install Now</em> i poczekaj na koniec instalacji.</li>"
     "<li>Otwórz wiersz poleceń i wykonaj:</li></ol>"
     "<pre><code class=\"language-bash\">python --version\npython -c \"print(2 + 2)\"</code></pre>"
     "<p><strong>Wyzwanie:</strong> zmień <code>2 + 2</code> na działanie, którego wyniku nie znasz na pamięć.</p>", ""),
    ("Python dla początkujących", "Zmienne i typy danych", "wzrokowiec",
     "<h3>Zmienna to podpisane pudełko</h3>"
     "<p>Nazwa zmiennej to etykieta, wartość to zawartość pudełka, a typ mówi, co może być w środku.</p>"
     "<table><thead><tr><th>Etykieta</th><th>Zawartość</th><th>Rodzaj pudełka (typ)</th></tr></thead><tbody>"
     "<tr><td>imie</td><td>\"Ola\"</td><td>str — tekst</td></tr><tr><td>wiek</td><td>21</td><td>int — liczba całkowita</td></tr>"
     "<tr><td>wzrost</td><td>1.68</td><td>float — liczba z przecinkiem</td></tr><tr><td>student</td><td>True</td><td>bool — tak/nie</td></tr>"
     "</tbody></table>"
     "<pre><code class=\"language-python\">imie = \"Ola\"\nprint(type(imie))  # &lt;class 'str'&gt;</code></pre>", ""),
    ("Python dla początkujących", "Zmienne i typy danych", "sluchowiec",
     "<h3>Posłuchaj, a potem powiedz na głos</h3>"
     "<p>Obejrzyj fragment filmu o zmiennych (po angielsku) i zatrzymuj go po każdym przykładzie.</p>"
     "<p>Przeczytaj kod na głos, mówiąc pełnymi zdaniami: „zmienna <em>wiek</em> dostaje wartość dwadzieścia jeden — to liczba całkowita”.</p>"
     "<pre><code class=\"language-python\">imie = \"Ola\"\nwiek = 21\nwzrost = 1.68\nstudent = True</code></pre>"
     "<p><strong>Do omówienia:</strong> wytłumacz komuś (albo asystentowi AI), czym różni się <code>21</code> od <code>\"21\"</code>.</p>",
     "https://www.youtube.com/watch?v=kqtD5dpn9C8"),
    ("Python dla początkujących", "Zmienne i typy danych", "czytanie-pisanie",
     "<h3>Definicje</h3>"
     "<ul><li><strong>Zmienna</strong> — nazwa przypisana do wartości w pamięci programu.</li>"
     "<li><strong>Przypisanie</strong> — instrukcja <code>nazwa = wartość</code>.</li>"
     "<li><strong>Typ</strong> — rodzaj wartości: <code>str</code>, <code>int</code>, <code>float</code>, <code>bool</code>.</li></ul>"
     "<pre><code class=\"language-python\">imie = \"Ola\"      # str\nwiek = 21          # int\nwzrost = 1.68      # float\nstudent = True     # bool</code></pre>"
     "<p><strong>Zadanie pisemne:</strong> w trzech zdaniach opisz własnymi słowami, kiedy użyjesz <code>int</code>, a kiedy <code>float</code>.</p>", ""),
    ("Python dla początkujących", "Zmienne i typy danych", "praktyk",
     "<h3>Eksperyment w konsoli</h3>"
     "<p>Uruchom <code>python</code> i wpisuj po kolei, obserwując wynik:</p>"
     "<pre><code class=\"language-python\">&gt;&gt;&gt; wiek = 21\n&gt;&gt;&gt; type(wiek)\n&gt;&gt;&gt; wiek = \"21\"\n&gt;&gt;&gt; type(wiek)\n&gt;&gt;&gt; wiek + 1</code></pre>"
     "<p>Ostatnia linia skończy się błędem. <strong>Napraw ją</strong> funkcją <code>int()</code> tak, żeby wynik wynosił 22.</p>", ""),
    ("Python dla początkujących", "Pętle for i while", "praktyk",
     "<h3>Napisz, uruchom, zmień</h3>"
     "<pre><code class=\"language-python\">for liczba in range(1, 6):\n    print(liczba * liczba)</code></pre>"
     "<ol><li>Uruchom program — powinien wypisać kwadraty liczb 1–5.</li>"
     "<li>Zmień go tak, żeby wypisał tylko liczby parzyste do 20.</li>"
     "<li>Przepisz to samo z pętlą <code>while</code>.</li></ol>", ""),
    ("Python dla początkujących", "Pętle for i while", "wzrokowiec",
     "<h3>Pętla krok po kroku</h3>"
     "<table><thead><tr><th>Obrót pętli</th><th>liczba</th><th>Wypisane</th></tr></thead><tbody>"
     "<tr><td>1</td><td>1</td><td>1</td></tr><tr><td>2</td><td>2</td><td>4</td></tr><tr><td>3</td><td>3</td><td>9</td></tr>"
     "<tr><td>4</td><td>4</td><td>16</td></tr><tr><td>5</td><td>5</td><td>25</td></tr></tbody></table>"
     "<pre><code class=\"language-python\">for liczba in range(1, 6):\n    print(liczba * liczba)</code></pre>", ""),
]

# Wersja lekcji, w której uczeń przerabiał kurs: (uczeń, kurs, styl) — podstawa analizy stylów.
# student2: Python w wersji „Praktyk” (zaliczone testy), Docker w wersji „Słuchowiec” (test niezaliczony)
# → system poleca mu zmianę stylu na „Praktyk”. student1 zostaje przy „Wzrokowcu”.
PROGRESS_STYLES = [
    ("student1", "Python dla początkujących", "wzrokowiec"),
    ("student1", "HTML i CSS: dostępne strony od podstaw", "wzrokowiec"),
    ("student2", "Python dla początkujących", "praktyk"),
    ("student2", "Docker i Kubernetes w praktyce", "sluchowiec"),
    ("michal.kaczmarek", "Linux i terminal na co dzień", "praktyk"),
]

# Kursy szczególnie dla stylów (pole „Kurs szczególnie dla stylów” w formularzu kursu)
COURSE_STYLES = {
    "Python dla zaawansowanych: klasy, wyjątki i testy": ["praktyk"],
    "Django od zera do aplikacji": ["praktyk"],
    "Docker i Kubernetes w praktyce": ["praktyk"],
    "Linux i terminal na co dzień": ["praktyk"],
    "JavaScript w przeglądarce": ["praktyk", "wzrokowiec"],
    "HTML i CSS: dostępne strony od podstaw": ["wzrokowiec"],
    "Projektowanie UX: od badań do prototypu": ["wzrokowiec"],
    "Analiza danych w pandas": ["wzrokowiec", "czytanie-pisanie"],
    "Wprowadzenie do Machine Learning": ["czytanie-pisanie"],
}

# Ankieta stylu: uczeń → styl wskazany w kolejnych 10 pytaniach.
# kuba.dabrowski nie wybrał stylu — polecane kursy wynikają u niego tylko z ankiety.
W, S, C, P = "wzrokowiec", "sluchowiec", "czytanie-pisanie", "praktyk"
SURVEYS = {
    "student1": [W, W, W, C, W, W, P, W, C, W],
    "student2": [S, P, S, S, P, S, P, P, S, P],
    "kuba.dabrowski": [P, P, P, W, P, P, P, S, P, P],
    "zuzia.mazur": [W, W, C, W, W, W, W, C, W, W],
    "natalia.pawlak": [C, C, W, C, C, C, C, C, W, C],
}
