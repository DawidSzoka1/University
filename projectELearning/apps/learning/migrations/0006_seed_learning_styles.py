"""Cztery domyślne style uczenia się (model VARK). Administrator może dodać kolejne w panelu."""

from django.db import migrations

STYLES = [
    (
        "Wzrokowiec", "wzrokowiec", 1,
        "Najlepiej zapamiętujesz to, co widzisz: schematy, diagramy, tabele, mapy myśli, wyróżnienia "
        "i filmy pokazujące rozwiązanie krok po kroku.",
        "Dodaj schemat, tabelę porównawczą, zrzuty ekranu albo mapę pojęć. Krótkie akapity, "
        "najważniejsze rzeczy wyróżnione.",
    ),
    (
        "Słuchowiec", "sluchowiec", 2,
        "Łatwiej uczysz się, słuchając i rozmawiając: wyjaśnienia słowne, nagrania, podcasty, "
        "tłumaczenie materiału na głos.",
        "Dodaj film lub nagranie z omówieniem, pisz jak w rozmowie, zaproponuj przeczytanie "
        "kodu na głos i pytania do przedyskutowania.",
    ),
    (
        "Czytanie i pisanie", "czytanie-pisanie", 3,
        "Wolisz tekst: definicje, opisy, listy, własne notatki i streszczenia. Zapisywanie "
        "pomaga Ci zrozumieć i zapamiętać.",
        "Dodaj precyzyjne definicje, listę najważniejszych punktów, słowniczek i zadanie "
        "„opisz własnymi słowami”.",
    ),
    (
        "Praktyk", "praktyk", 4,
        "Uczysz się przez działanie: ćwiczenia, eksperymenty, przykłady do uruchomienia "
        "i zadania wykonywane krok po kroku.",
        "Zacznij od zadania do wykonania, dodaj listę kroków, przykład do zmodyfikowania "
        "i krótkie wyzwanie na koniec.",
    ),
]


def create_styles(apps, schema_editor):
    LearningStyle = apps.get_model("learning", "LearningStyle")
    for name, slug, order, description, hint in STYLES:
        LearningStyle.objects.update_or_create(
            slug=slug,
            defaults={"name": name, "order": order, "description": description, "author_hint": hint, "is_active": True},
        )


def remove_styles(apps, schema_editor):
    apps.get_model("learning", "LearningStyle").objects.filter(slug__in=[s[1] for s in STYLES]).delete()


class Migration(migrations.Migration):
    dependencies = [("learning", "0005_learning_styles")]

    operations = [migrations.RunPython(create_styles, remove_styles)]
