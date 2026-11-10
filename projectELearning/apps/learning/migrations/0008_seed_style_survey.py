"""Ankieta diagnostyczna: 10 pytań sytuacyjnych. Żadne nie pyta wprost o styl — każda odpowiedź
opisuje zachowanie typowe dla jednego stylu (wzrokowiec, słuchowiec, czytanie i pisanie, praktyk).
Administrator może zmieniać pytania i dodawać odpowiedzi dla nowych stylów w panelu."""

from django.db import migrations

V, A, R, K = "wzrokowiec", "sluchowiec", "czytanie-pisanie", "praktyk"

QUESTIONS = [
    ("Musisz nauczyć się obsługi nowego programu. Od czego zaczynasz?", [
        (K, "Od razu go uruchamiam i klikam, sprawdzając, co się stanie."),
        (V, "Oglądam film albo zrzuty ekranu pokazujące, jak go używać."),
        (A, "Proszę kogoś, kto go zna, żeby mi go objaśnił."),
        (R, "Czytam instrukcję albo dokumentację."),
    ]),
    ("Przygotowujesz się do ważnego sprawdzianu. Co pomaga Ci najbardziej?", [
        (V, "Rysuję schematy, mapy myśli i zaznaczam kolorami."),
        (A, "Powtarzam materiał na głos albo omawiam go z kimś."),
        (R, "Robię notatki i streszczenia własnymi słowami."),
        (K, "Rozwiązuję jak najwięcej zadań i przykładów."),
    ]),
    ("Ktoś tłumaczy Ci drogę do nieznanego miejsca. Co wolisz?", [
        (V, "Mapę albo szkic z zaznaczoną trasą."),
        (A, "Ustny opis: skręć w prawo przy piekarni…"),
        (R, "Spisaną listę kroków z nazwami ulic."),
        (K, "Żeby ktoś przeszedł ze mną kawałek drogi."),
    ]),
    ("Na wykładzie najwięcej zapamiętujesz, gdy prowadzący…", [
        (V, "Pokazuje slajdy z diagramami i wykresami."),
        (A, "Ciekawie opowiada i zadaje pytania słuchaczom."),
        (R, "Udostępnia notatki i listę lektur."),
        (K, "Daje ćwiczenie do wykonania na zajęciach."),
    ]),
    ("Kupujesz nowe urządzenie. Jak poznajesz jego funkcje?", [
        (K, "Włączam i testuję wszystkie przyciski."),
        (V, "Oglądam rysunki w instrukcji i ikonki na obudowie."),
        (A, "Pytam sprzedawcę albo znajomego, który ma takie samo."),
        (R, "Czytam opis funkcji punkt po punkcie."),
    ]),
    ("Chcesz zapamiętać nowe słowo w obcym języku. Jak to robisz?", [
        (A, "Powtarzam je na głos kilka razy."),
        (V, "Wyobrażam sobie obrazek, który z nim kojarzę."),
        (R, "Zapisuję je kilka razy i układam z nim zdanie."),
        (K, "Od razu używam go w rozmowie albo w ćwiczeniu."),
    ]),
    ("W pracy grupowej najchętniej…", [
        (V, "Przygotowuję prezentację lub plakat."),
        (A, "Prowadzę dyskusję i referuję wyniki."),
        (R, "Spisuję ustalenia i redaguję tekst."),
        (K, "Buduję prototyp lub wykonuję doświadczenie."),
    ]),
    ("Gdy czegoś nie rozumiesz w materiale, najpierw…", [
        (V, "Szukam rysunku lub filmu, który to pokazuje."),
        (A, "Proszę kogoś, żeby mi to wytłumaczył innymi słowami."),
        (R, "Czytam to samo w innym źródle albo w definicji."),
        (K, "Próbuję na przykładzie i sprawdzam, co wyjdzie."),
    ]),
    ("Który sposób powtórki przed zajęciami jest dla Ciebie najlepszy?", [
        (V, "Przejrzenie swoich schematów i zakreśleń."),
        (A, "Wysłuchanie nagrania lub podcastu na ten temat."),
        (R, "Przeczytanie notatek i fiszek."),
        (K, "Zrobienie kilku szybkich zadań praktycznych."),
    ]),
    ("Masz opowiedzieć znajomym o czymś, czego się nauczyłeś. Jak to robisz?", [
        (V, "Pokazuję zdjęcia, rysunki albo wykres."),
        (A, "Po prostu opowiadam, z przykładami."),
        (R, "Wysyłam im link do dobrego artykułu i swoje notatki."),
        (K, "Pokazuję to w praktyce, na żywo."),
    ]),
]


def create_survey(apps, schema_editor):
    LearningStyle = apps.get_model("learning", "LearningStyle")
    SurveyQuestion = apps.get_model("learning", "SurveyQuestion")
    SurveyOption = apps.get_model("learning", "SurveyOption")
    styles = {style.slug: style for style in LearningStyle.objects.all()}
    if SurveyQuestion.objects.exists():
        return
    for q_order, (text, options) in enumerate(QUESTIONS, start=1):
        question = SurveyQuestion.objects.create(text=text, order=q_order)
        for o_order, (slug, option_text) in enumerate(options, start=1):
            if slug in styles:
                SurveyOption.objects.create(question=question, text=option_text, style=styles[slug], order=o_order)


def remove_survey(apps, schema_editor):
    apps.get_model("learning", "SurveyQuestion").objects.filter(text__in=[q[0] for q in QUESTIONS]).delete()


class Migration(migrations.Migration):
    dependencies = [("learning", "0007_style_survey")]

    operations = [migrations.RunPython(create_survey, remove_survey)]
