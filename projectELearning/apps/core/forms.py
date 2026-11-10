"""Formularze modułu core — edycja motywów kolorystycznych."""

from __future__ import annotations

from django import forms

from apps.core.colors import normalize_hex
from apps.core.models import ColorTheme

_COLOR_WIDGET = forms.TextInput(attrs={"type": "color"})


class AccessibleErrorsMixin:
    """Wiąże komunikat błędu z polem: ``aria-invalid`` + ``aria-describedby``.

    Szablon ``includes/form_field.html`` renderuje błąd z id ``<auto_id>-error``,
    a tekst pomocy z id ``<auto_id>_helptext`` (konwencja Django).
    """

    def full_clean(self) -> None:
        super().full_clean()
        for name in self.errors:
            if name not in self.fields:
                continue
            field = self.fields[name]
            auto_id = self[name].auto_id
            described_by = [f"{auto_id}_helptext"] if field.help_text else []
            described_by.append(f"{auto_id}-error")
            field.widget.attrs["aria-invalid"] = "true"
            field.widget.attrs["aria-describedby"] = " ".join(described_by)


class ColorThemeForm(forms.ModelForm):
    """Formularz motywu z natywnymi color pickerami przeglądarki.

    Pięć pól bazowych (primary, secondary, accent, surface, surface_variant)
    definiuje całą kolorystykę aplikacji. Reszta palety (w tym kolory tekstu
    i tryb ciemny) wylicza się automatycznie w modelu i arkuszu CSS.
    """

    class Meta:
        model = ColorTheme
        fields = [
            "name",
            "primary",
            "secondary",
            "accent",
            "surface",
            "surface_variant",
            "is_default",
        ]
        widgets = {
            "primary": _COLOR_WIDGET,
            "secondary": _COLOR_WIDGET,
            "accent": _COLOR_WIDGET,
            "surface": _COLOR_WIDGET,
            "surface_variant": _COLOR_WIDGET,
        }
        help_texts = {
            "primary": "Kolor główny (przyciski, linki, obramowania aktywne).",
            "secondary": "Kolor dodatkowy (akcenty drugorzędne, odznaki, tagi).",
            "accent": "Kolor wyróżnienia (akcenty specjalne, ikony, powiadomienia).",
            "surface": "Kolor tła strony. Ciemne tło automatycznie włącza tryb ciemny.",
            "surface_variant": "Kolor tła kontenerów, kart, paneli i sekcji.",
            "is_default": "Motyw używany, gdy użytkownik nie wybrał własnego.",
        }

    def _clean_color(self, field_name: str) -> str:
        value = self.cleaned_data[field_name]
        try:
            return normalize_hex(value)
        except ValueError as exc:
            raise forms.ValidationError(str(exc)) from exc

    def clean_primary(self) -> str:
        return self._clean_color("primary")

    def clean_secondary(self) -> str:
        return self._clean_color("secondary")

    def clean_accent(self) -> str:
        return self._clean_color("accent")

    def clean_surface(self) -> str:
        return self._clean_color("surface")

    def clean_surface_variant(self) -> str:
        return self._clean_color("surface_variant")

    def clean(self):
        """Motyw musi dawać kontrast WCAG AA we wszystkich parach kolorów używanych przez arkusz."""
        from apps.core.colors import theme_accessibility_problems

        cleaned = super().clean()
        names = ("primary", "secondary", "accent", "surface", "surface_variant")
        if all(cleaned.get(name) for name in names):
            for problem in theme_accessibility_problems(*(cleaned[name] for name in names)):
                self.add_error(None, problem)
        return cleaned
