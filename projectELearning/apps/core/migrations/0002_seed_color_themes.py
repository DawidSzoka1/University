"""Seed the initial set of four colour themes.

Four visually distinct palettes (warm light, dark blue, light green,
dark purple) so users can pick more than just "light/dark".  The
migration is reversible: rolling back removes only the seeded names.
"""

from django.db import migrations

THEMES = [
    {
        "name": "Bursztyn",
        "primary": "#B45309",
        "secondary": "#0F766E",
        "surface": "#FFFBEB",
        "is_default": True,
    },
    {
        "name": "Ocean",
        "primary": "#38BDF8",
        "secondary": "#22D3EE",
        "surface": "#0B1220",
        "is_default": False,
    },
    {
        "name": "Las",
        "primary": "#166534",
        "secondary": "#A16207",
        "surface": "#F0FDF4",
        "is_default": False,
    },
    {
        "name": "Neon",
        "primary": "#C084FC",
        "secondary": "#F472B6",
        "surface": "#1E1033",
        "is_default": False,
    },
]


def create_themes(apps, schema_editor):
    ColorTheme = apps.get_model("core", "ColorTheme")
    for theme in THEMES:
        ColorTheme.objects.get_or_create(name=theme["name"], defaults=theme)


def remove_themes(apps, schema_editor):
    ColorTheme = apps.get_model("core", "ColorTheme")
    ColorTheme.objects.filter(name__in=[theme["name"] for theme in THEMES]).delete()


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0001_initial"),
    ]

    operations = [
        migrations.RunPython(create_themes, remove_themes),
    ]
