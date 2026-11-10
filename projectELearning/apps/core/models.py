from __future__ import annotations

from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import models, transaction

from apps.core.colors import (
    is_dark_color,
    normalize_hex,
    readable_text_color,
    theme_palette,
)


class ColorTheme(models.Model):
    """Nazwany motyw kolorystyczny złożony z pięciu kolorów bazowych.

    Cała aplikacja używa ``primary``, ``secondary``, ``accent``, ``surface``
    oraz ``surface_variant``; odcienie pośrednie powstają w CSS przez ``color-mix()``.
    Kolory tekstu (``on_*``) są wyliczane z kontrastu WCAG, więc każdy nowy motyw
    dodany przez administratora od razu pozostaje czytelny.
    """

    name = models.CharField("Nazwa", max_length=100, unique=True)
    primary = models.CharField("Kolor główny", max_length=7, default="#B45309")
    secondary = models.CharField("Kolor dodatkowy", max_length=7, default="#0F766E")
    accent = models.CharField("Kolor akcentowy", max_length=7, default="#D97706")
    surface = models.CharField("Tło", max_length=7, default="#FFFBEB")
    surface_variant = models.CharField("Tło kontenerów", max_length=7, default="#FEF3C7")
    is_default = models.BooleanField("Domyślny", default=False)
    created_at = models.DateTimeField("Utworzono", auto_now_add=True)

    class Meta:
        ordering = ["name"]
        verbose_name = "Motyw kolorystyczny"
        verbose_name_plural = "Motywy kolorystyczne"

    def __str__(self) -> str:
        return self.name

    # --- Walidacja -----------------------------------------------------

    def clean(self) -> None:
        """Sprawdza format kolorów, żeby do bazy nie trafił dowolny tekst."""
        errors: dict[str, list[str]] = {}
        for field in ("primary", "secondary", "accent", "surface", "surface_variant"):
            try:
                normalize_hex(getattr(self, field))
            except ValueError as exc:
                errors[field] = [str(exc)]
        if errors:
            raise ValidationError(errors)

    def save(self, *args, **kwargs) -> None:
        """Normalizuje kolory i pilnuje, żeby istniał dokładnie jeden domyślny motyw."""
        self.primary = normalize_hex(self.primary)
        self.secondary = normalize_hex(self.secondary)
        self.accent = normalize_hex(self.accent)
        self.surface = normalize_hex(self.surface)
        self.surface_variant = normalize_hex(self.surface_variant)

        with transaction.atomic():
            if self.is_default:
                (
                    ColorTheme.objects.exclude(pk=self.pk)
                    .filter(is_default=True)
                    .update(is_default=False)
                )
            elif not (
                ColorTheme.objects.exclude(pk=self.pk)
                .filter(is_default=True)
                .exists()
            ):
                # Pierwszy motyw w bazie (lub jedyny) zostaje domyślnym.
                self.is_default = True

            super().save(*args, **kwargs)

    # --- Kolory pochodne ----------------------------------------------

    @property
    def is_dark(self) -> bool:
        """Czy motyw jest ciemny (tło wymaga jasnego tekstu)."""
        return is_dark_color(self.surface)

    @property
    def mode(self) -> str:
        """Wartość atrybutu ``data-theme-mode`` dla ``<html>``."""
        return "dark" if self.is_dark else "light"

    @property
    def on_primary(self) -> str:
        """Kolor tekstu na kolorze głównym (czarny lub biały)."""
        return readable_text_color(self.primary)

    @property
    def on_secondary(self) -> str:
        """Kolor tekstu na kolorze dodatkowym (czarny lub biały)."""
        return readable_text_color(self.secondary)

    @property
    def on_accent(self) -> str:
        """Kolor tekstu na kolorze akcentowym (czarny lub biały)."""
        return readable_text_color(self.accent)

    @property
    def on_surface(self) -> str:
        """Kolor tekstu na tle strony (czarny lub biały)."""
        return readable_text_color(self.surface)

    @property
    def on_surface_variant(self) -> str:
        """Kolor tekstu na tle kontenerów (czarny lub biały)."""
        return readable_text_color(self.surface_variant)

    @property
    def palette(self) -> dict[str, object]:
        """Komplet zmiennych motywu gotowy dla context processora."""
        return theme_palette(
            self.primary,
            self.secondary,
            self.surface,
            self.accent,
            self.surface_variant,
        )


class Notification(models.Model):
    """Powiadomienie widoczne na pulpicie użytkownika."""

    class Kind(models.TextChoices):
        COMMENT_APPROVED = "comment_approved", "Komentarz zaakceptowany"
        COMMENT_PENDING = "comment_pending", "Komentarz czeka na akceptację"
        COMMENT_REPLY = "comment_reply", "Nowa odpowiedź na komentarz"
        COMMENT_REJECTED = "comment_rejected", "Komentarz odrzucony lub usunięty"
        COURSE_SUBMITTED = "course_submitted", "Nowe zgłoszenie kursu"
        SUBMISSION_MESSAGE = "submission_message", "Nowa wiadomość w zgłoszeniu"
        SUBMISSION_STATUS = "submission_status", "Zmiana statusu zgłoszenia"
        QUIZ_TO_GRADE = "quiz_to_grade", "Nowy test do sprawdzenia"
        QUIZ_GRADED = "quiz_graded", "Test oceniony"
        COURSE_COMPLETED = "course_completed", "Kurs ukończony"
        ORDER_STATUS = "order_status", "Zmiana statusu zamówienia"
        SYSTEM = "system", "Komunikat systemowy"

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="notifications",
        verbose_name="Użytkownik",
    )
    kind = models.CharField(
        "Rodzaj",
        max_length=32,
        choices=Kind.choices,
        default=Kind.SYSTEM,
        db_index=True,
    )
    title = models.CharField("Tytuł", max_length=200)
    message = models.TextField("Treść", blank=True)
    url = models.CharField("Link", max_length=500, blank=True)
    is_read = models.BooleanField("Przeczytane", default=False)
    created_at = models.DateTimeField("Utworzono", auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Powiadomienie"
        verbose_name_plural = "Powiadomienia"

    def __str__(self) -> str:
        return self.title
