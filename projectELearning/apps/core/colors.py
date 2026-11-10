"""Narzędzia do pracy z kolorami motywów.

Cała paleta aplikacji opiera się na pięciu kolorach bazowych:
(``primary``, ``secondary``, ``accent``, ``surface``, ``surface_variant``).
Pozostałe odcienie powstają w CSS przez ``color-mix()``, ale dwa elementy
muszą być wyliczone w Pythonie, ponieważ zależą od jasności koloru:

* kolor tekstu kładziony na kolor (``--color-on-primary`` itd.),
* informacja, czy motyw jest ciemny (``data-theme-mode="dark"``).

Moduł nie zależy od Django — można go testować w izolacji.
"""

from __future__ import annotations

import re

_HEX_RE = re.compile(r"^#?(?P<hex>[0-9a-fA-F]{6})$")

# Kolory tekstu wybierane na podstawie kontrastu WCAG.
_DARK_TEXT = "#111827"
_LIGHT_TEXT = "#FFFFFF"


def normalize_hex(value: str) -> str:
    """Zwraca kolor w postaci ``#RRGGBB`` (wielkie litery).

    Akceptuje zapis z ``#`` lub bez.  Podnosi ``ValueError`` dla
    niepoprawnej wartości — dzięki temu formularze i model nie zapiszą
    przypadkowego tekstu w polu koloru.
    """
    if not isinstance(value, str):
        raise ValueError("Kolor musi być tekstem w formacie #RRGGBB.")

    match = _HEX_RE.match(value.strip())
    if not match:
        raise ValueError("Kolor musi mieć format #RRGGBB (np. #4F46E5).")
    return f"#{match.group('hex').upper()}"


def is_valid_hex(value: str) -> bool:
    """Czy wartość jest poprawnym kolorem ``#RRGGBB``."""
    try:
        normalize_hex(value)
    except ValueError:
        return False
    return True


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    """Zamienia ``#RRGGBB`` na krotkę ``(r, g, b)`` w zakresie 0–255."""
    normalized = normalize_hex(value)
    return (
        int(normalized[1:3], 16),
        int(normalized[3:5], 16),
        int(normalized[5:7], 16),
    )


def _channel_luminance(channel: int) -> float:
    """Luminancja pojedynczego kanału wg WCAG 2.1."""
    srgb = channel / 255
    if srgb <= 0.04045:
        return srgb / 12.92
    return ((srgb + 0.055) / 1.055) ** 2.4


def relative_luminance(value: str) -> float:
    """Względna luminancja koloru (0 = czarny, 1 = biały)."""
    red, green, blue = hex_to_rgb(value)
    return (
        0.2126 * _channel_luminance(red)
        + 0.7152 * _channel_luminance(green)
        + 0.0722 * _channel_luminance(blue)
    )


def contrast_ratio(first: str, second: str) -> float:
    """Współczynnik kontrastu WCAG między dwoma kolorami (1–21)."""
    lum_a = relative_luminance(first)
    lum_b = relative_luminance(second)
    lighter, darker = max(lum_a, lum_b), min(lum_a, lum_b)
    return (lighter + 0.05) / (darker + 0.05)


def readable_text_color(background: str) -> str:
    """Zwraca czarny lub biały tekst o lepszym kontraście na tle.

    Dla każdego koloru jeden z wariantów ma kontrast co najmniej ~4.58,
    więc wynik zawsze spełnia WCAG AA dla zwykłego tekstu.
    """
    return (
        _DARK_TEXT
        if contrast_ratio(background, _DARK_TEXT)
        >= contrast_ratio(background, _LIGHT_TEXT)
        else _LIGHT_TEXT
    )


def is_dark_color(value: str) -> bool:
    """Czy kolor jest na tyle ciemny, że tekst na nim powinien być biały."""
    return readable_text_color(value) == _LIGHT_TEXT


def theme_palette(
    primary: str,
    secondary: str,
    surface: str,
    accent: str = "#D97706",
    surface_variant: str = "#FEF3C7",
) -> dict[str, object]:
    """Buduje komplet zmiennych motywu (5 kolorów bazowych + WCAG teksty)."""
    p = normalize_hex(primary)
    s = normalize_hex(secondary)
    acc = normalize_hex(accent)
    surf = normalize_hex(surface)
    surf_var = normalize_hex(surface_variant)
    return {
        "primary": p,
        "secondary": s,
        "accent": acc,
        "surface": surf,
        "surface_variant": surf_var,
        "on_primary": readable_text_color(p),
        "on_secondary": readable_text_color(s),
        "on_accent": readable_text_color(acc),
        "on_surface": readable_text_color(surf),
        "on_surface_variant": readable_text_color(surf_var),
        "is_dark": is_dark_color(surf),
    }


# ---------------------------------------------------------------------------
# Panel administracyjny (django-unfold)
# ---------------------------------------------------------------------------
# Unfold koloruje interfejs skalami Tailwinda 50–950: ``primary`` (akcje),
# ``base`` (tła, ramki, tekst) i ``font``. Wyprowadzamy je z kolorów motywu,
# żeby panel wyglądał jak platforma, i pilnujemy kontrastu WCAG tam, gdzie
# Unfold kładzie tekst: biały napis na ``primary-600`` (przyciski) oraz
# tekst ``base-500``/``base-600`` na tle strony i białych kartach.

_WHITE = "#FFFFFF"
_BLACK = "#000000"
_AA = 4.5


def mix(first: str, second: str, amount: float) -> str:
    """Miesza dwa kolory HEX (``amount`` = udział drugiego koloru, 0–1)."""
    a, b = hex_to_rgb(first), hex_to_rgb(second)
    return "#" + "".join(f"{round(x + (y - x) * amount):02X}" for x, y in zip(a, b))


def _tone_with_contrast(start: str, toward: str, backgrounds: list[str], from_amount: float) -> str:
    """Najsłabsze przesunięcie ``start`` → ``toward`` spełniające AA na wszystkich tłach."""
    amount = from_amount
    while amount < 1:
        candidate = mix(start, toward, amount)
        if all(contrast_ratio(candidate, bg) >= _AA for bg in backgrounds):
            return candidate
        amount += 0.02
    return normalize_hex(toward)


def unfold_palette(
    primary: str,
    surface: str,
    surface_variant: str | None = None,
) -> dict[str, dict[str, str]]:
    """Skale ``primary``, ``base`` i ``font`` dla Unfold z kolorów motywu."""
    p = normalize_hex(primary)
    surf = normalize_hex(surface)
    ink = readable_text_color(surf)
    dark = is_dark_color(surf)

    # Przycisk: Unfold zawsze pisze na nim białym tekstem.
    button = _tone_with_contrast(p, _BLACK, [_WHITE], 0.0)
    primary_scale = {
        "50": mix(p, _WHITE, 0.92),
        "100": mix(p, _WHITE, 0.82),
        "200": mix(p, _WHITE, 0.64),
        "300": mix(p, _WHITE, 0.42),
        "400": mix(p, _WHITE, 0.2),
        "500": p,
        "600": button,
        "700": mix(button, _BLACK, 0.2),
        "800": mix(button, _BLACK, 0.38),
        "900": mix(button, _BLACK, 0.55),
        "950": mix(button, _BLACK, 0.7),
    }

    if not dark:
        # Jasny panel: tło strony = surface, karty białe, tekst = ink.
        backgrounds = [surf, _WHITE]
        base = {
            "50": surf,
            "100": mix(surf, ink, 0.05) if surface_variant is None else mix(normalize_hex(surface_variant), surf, 0.4),
            "200": mix(surf, ink, 0.16),
            "300": mix(surf, ink, 0.3),
            "400": mix(surf, ink, 0.48),
            "500": _tone_with_contrast(surf, ink, backgrounds, 0.6),
            "600": _tone_with_contrast(surf, ink, backgrounds, 0.8),
            "700": mix(surf, ink, 0.9),
            "800": mix(surf, ink, 0.95),
            "900": ink,
            "950": mix(ink, _BLACK, 0.5),
        }
    else:
        # Ciemny panel: tło = surface (base-900), karty = base-800/900, tekst jasny.
        card = mix(surf, ink, 0.07)
        backgrounds = [surf, card]
        base = {
            "50": ink,
            "100": mix(surf, ink, 0.96),
            "200": mix(surf, ink, 0.9),
            "300": _tone_with_contrast(surf, ink, backgrounds, 0.82),
            "400": _tone_with_contrast(surf, ink, backgrounds, 0.66),
            "500": mix(surf, ink, 0.45),
            "600": mix(surf, ink, 0.3),
            "700": mix(surf, ink, 0.17),
            "800": card,
            "900": surf,
            "950": mix(surf, _BLACK, 0.35),
        }

    return {
        "primary": primary_scale,
        "base": base,
        "font": {
            "subtle-light": "var(--color-base-500)",
            "subtle-dark": "var(--color-base-400)",
            "default-light": "var(--color-base-600)",
            "default-dark": "var(--color-base-300)",
            "important-light": "var(--color-base-900)",
            "important-dark": "var(--color-base-100)",
        },
    }


def _active_theme(request=None):
    """Motyw aktywny dla żądania (użytkownik → sesja → domyślny)."""
    theme = None
    if request is not None:
        try:
            from apps.core.context_processors import _get_theme_data

            theme = _get_theme_data(request).get("current_theme")
        except Exception:
            theme = None
    if theme is None:
        try:
            from apps.core.models import ColorTheme

            theme = ColorTheme.objects.filter(is_default=True).first() or ColorTheme.objects.first()
        except Exception:
            theme = None
    return theme


def get_unfold_theme_colors(request=None) -> dict[str, dict[str, str]]:
    """``UNFOLD["COLORS"]`` — skale kolorów panelu z aktywnego motywu."""
    theme = _active_theme(request)
    return unfold_palette(
        getattr(theme, "primary", "#B45309"),
        getattr(theme, "surface", "#FFFBEB"),
        getattr(theme, "surface_variant", None),
    )


def get_unfold_theme_mode(request=None) -> str:
    """``UNFOLD["THEME"]`` — tryb jasny/ciemny panelu zgodny z motywem platformy."""
    theme = _active_theme(request)
    return "dark" if is_dark_color(getattr(theme, "surface", "#FFFBEB")) else "light"


# ---------------------------------------------------------------------------
# Kontrola dostępności motywu (walidacja w panelu administratora)
# ---------------------------------------------------------------------------
AA_TEXT = 4.5


def theme_accessibility_problems(primary: str, secondary: str, accent: str, surface: str, surface_variant: str) -> list[str]:
    """Lista problemów z kontrastem WCAG AA dla motywu (pusta = motyw poprawny).

    Sprawdza dokładnie te pary kolorów, których używa arkusz ``main.css``: tekst na tle
    i tle kontenerów (ten sam kolor tekstu na obu), tekst pomocniczy (76% tekstu + tło),
    napisy na przyciskach i kolor marki użyty jako tekst (82% koloru + tekst).
    """
    problems = []
    on = readable_text_color(surface)
    ratio = contrast_ratio(on, surface)
    if ratio < AA_TEXT:
        problems.append(f"Tło jest zbyt „pośrodku” skali jasności — żaden kolor tekstu nie daje na nim kontrastu 4,5:1 (najlepszy: {ratio:.2f}:1). Wybierz wyraźnie jaśniejsze albo ciemniejsze tło.")
    if readable_text_color(surface_variant) != on:
        problems.append("Tło kontenerów wymaga innego koloru tekstu niż tło strony (jedno jest jasne, drugie ciemne). Oba tła muszą być jasne albo oba ciemne.")
    elif contrast_ratio(on, surface_variant) < AA_TEXT:
        problems.append(f"Tekst na tle kontenerów ma kontrast {contrast_ratio(on, surface_variant):.2f}:1 (wymagane 4,5:1).")
    muted = mix(on, surface, 0.24)
    if ratio >= AA_TEXT and contrast_ratio(muted, surface) < AA_TEXT:
        problems.append(f"Tekst pomocniczy (opisy, daty) ma na tle kontrast {contrast_ratio(muted, surface):.2f}:1 — wybierz tło o większym kontraście z tekstem.")
    for label, color in (("główny", primary), ("dodatkowy", secondary), ("akcentowy", accent)):
        button = contrast_ratio(readable_text_color(color), color)
        if button < AA_TEXT:
            problems.append(f"Kolor {label}: napis na przycisku w tym kolorze ma kontrast {button:.2f}:1 (wymagane 4,5:1).")
    for label, color in (("główny", primary), ("dodatkowy", secondary)):
        as_text = contrast_ratio(mix(color, on, 0.18), surface)
        if as_text < AA_TEXT:
            problems.append(f"Kolor {label} użyty jako tekst (nadtytuły, linki) ma na tle kontrast {as_text:.2f}:1 — zbyt podobny do tła.")
    return problems
