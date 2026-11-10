"""
Context processors for the core app.

Provides the active colour theme and the notification counter to every
template.  A theme is a named triple of base colours stored in
``core.ColorTheme``; the remaining shades are derived in CSS.
"""

from __future__ import annotations

from django.db import DatabaseError
from django.http import HttpRequest

from apps.core.colors import theme_palette

# ---------------------------------------------------------------------------
# Fallback palette (used when the database has no themes yet)
# ---------------------------------------------------------------------------
_FALLBACK = theme_palette(
    primary="#B45309",
    secondary="#0F766E",
    surface="#FFFBEB",
    accent="#D97706",
    surface_variant="#FEF3C7",
)

SESSION_THEME_KEY = "theme_id"


def _get_theme_data(request: HttpRequest) -> dict[str, object]:
    """Resolve the active theme for the current request.

    Priority:
      1. logged-in user's ``selected_theme`` (saved across devices),
      2. theme stored in the session (anonymous visitors),
      3. the theme marked as default (or the first available theme).
    """
    from apps.core.models import ColorTheme  # noqa: WPS433 – lazy import

    try:
        themes = list(ColorTheme.objects.all().order_by("-is_default", "name"))
    except DatabaseError:  # pragma: no cover – DB not ready yet
        themes = []

    default_theme = next((theme for theme in themes if theme.is_default), None)
    if default_theme is None and themes:
        default_theme = themes[0]

    selected_theme = None
    source = "default"

    user = getattr(request, "user", None)
    if user is not None and user.is_authenticated:
        selected_theme = getattr(user, "selected_theme", None)
        if selected_theme is not None:
            source = "user"

    if selected_theme is None:
        session = getattr(request, "session", None)
        theme_id = session.get(SESSION_THEME_KEY) if session is not None else None
        if theme_id:
            selected_theme = next(
                (theme for theme in themes if theme.pk == theme_id), None
            )
            if selected_theme is not None:
                source = "session"

    current_theme = selected_theme or default_theme
    palette = current_theme.palette if current_theme is not None else _FALLBACK
    theme_is_custom = source in ("user", "session")

    return {
        # Base tokens consumed by base.html / templates (5 base tokens + WCAG on_*).
        "theme_primary": palette["primary"],
        "theme_secondary": palette["secondary"],
        "theme_accent": palette["accent"],
        "theme_background": palette["surface"],
        "theme_surface_variant": palette["surface_variant"],
        "theme_on_primary": palette["on_primary"],
        "theme_on_secondary": palette["on_secondary"],
        "theme_on_accent": palette["on_accent"],
        "theme_on_surface": palette["on_surface"],
        "theme_on_surface_variant": palette["on_surface_variant"],
        "theme_mode": "dark" if palette["is_dark"] else "light",
        "theme_is_custom": theme_is_custom,
        # Theme objects for the switcher / selection page.
        "current_theme": current_theme,
        "current_theme_id": current_theme.pk if current_theme is not None else None,
        "default_theme": default_theme,
        "available_themes": themes,
        "theme_source": source,
    }


def theme_colors(request: HttpRequest) -> dict[str, object]:
    """Inject theme variables and the list of available themes."""
    return _get_theme_data(request)


def notifications_count(request: HttpRequest) -> dict[str, object]:
    """Licznik nieprzeczytanych i 5 ostatnich powiadomień do panelu pod dzwonkiem.

    ``recent_notifications`` jest funkcją — szablon wywołuje ją dopiero, gdy
    renderuje panel, więc strony bez navbara nie wykonują dodatkowego zapytania.
    """
    user = getattr(request, "user", None)
    if user is None or not user.is_authenticated:
        return {"unread_notifications_count": 0, "recent_notifications": []}

    from apps.core.models import Notification  # noqa: WPS433

    mine = Notification.objects.filter(user=user)
    return {
        "unread_notifications_count": mine.filter(is_read=False).count(),
        "recent_notifications": lambda: list(mine.order_by("-created_at")[:5]),
    }
