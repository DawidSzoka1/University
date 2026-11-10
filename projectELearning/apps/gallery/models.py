"""Galeria i slajder: zdjęcia w bazie (skalowane lokalnie Pillow) i zapisane układy slajdera."""

from __future__ import annotations

import uuid
from pathlib import Path

from django.conf import settings
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models, transaction


def gallery_image_path(instance, filename: str) -> str:
    return f"gallery/{uuid.uuid4().hex}/{Path(filename).name}"


class GalleryItem(models.Model):
    """Zdjęcie w bazie galerii. Przy dodaniu skalowane do maks. MAX_SIZE (Pillow, bez chmury)."""

    MAX_SIZE = (1600, 1200)

    title = models.CharField("Tytuł", max_length=200)
    description = models.TextField("Opis", help_text="Pełny opis pod zdjęciem — dowolnej długości, wyświetlany w całości.")
    image = models.ImageField("Zdjęcie", upload_to=gallery_image_path)
    image_alt = models.CharField(
        "Tekst alternatywny",
        max_length=255,
        default="",
        help_text="Co widać na zdjęciu — dla osób, które go nie widzą.",
    )
    related_course = models.ForeignKey(
        "catalog.Course",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="gallery_items",
        verbose_name="Powiązany kurs",
    )
    uploaded_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="gallery_items",
        verbose_name="Dodał(a)",
    )
    original_width = models.PositiveIntegerField("Szerokość oryginału", default=0)
    original_height = models.PositiveIntegerField("Wysokość oryginału", default=0)
    width = models.PositiveIntegerField("Szerokość po skalowaniu", default=0)
    height = models.PositiveIntegerField("Wysokość po skalowaniu", default=0)
    uploaded_at = models.DateTimeField("Dodano", auto_now_add=True)

    class Meta:
        ordering = ["-uploaded_at", "-pk"]
        verbose_name = "Zdjęcie galerii"
        verbose_name_plural = "Zdjęcia galerii"

    def __str__(self) -> str:
        return self.title

    @property
    def was_resized(self) -> bool:
        return bool(self.original_width) and (self.original_width, self.original_height) != (self.width, self.height)

    def save(self, *args, **kwargs) -> None:
        # Nowo przesłany plik: zapamiętaj rozdzielczość oryginału i przeskaluj lokalnie (Pillow)
        if self.image and not getattr(self.image, "_committed", True):
            from PIL import Image

            from apps.core.images import resize_image_field

            self.image.seek(0)
            with Image.open(self.image) as source:
                self.original_width, self.original_height = source.size
            resize_image_field(self.image, self.MAX_SIZE)
            self.image.seek(0)
            with Image.open(self.image) as scaled:
                self.width, self.height = scaled.size
            self.image.seek(0)
        super().save(*args, **kwargs)


class SliderPreset(models.Model):
    """Zapisany układ slajdera (kolejność kart). Na stronie galerii widać dokładnie jeden — wyświetlany."""

    name = models.CharField("Nazwa układu", max_length=200, unique=True)
    is_active = models.BooleanField("Wyświetlany w galerii", default=False)
    autoplay = models.BooleanField(
        "Przewijaj automatycznie",
        default=True,
        help_text="Osoby z ograniczeniem animacji w systemie i tak zobaczą slajder zatrzymany.",
    )
    interval_seconds = models.PositiveSmallIntegerField(
        "Czas wyświetlania karty (s)",
        default=6,
        validators=[MinValueValidator(3), MaxValueValidator(30)],
    )
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="slider_presets",
        verbose_name="Utworzył(a)",
    )
    created_at = models.DateTimeField("Utworzono", auto_now_add=True)
    updated_at = models.DateTimeField("Zmieniono", auto_now=True)

    class Meta:
        ordering = ["-is_active", "name"]
        verbose_name = "Układ slajdera"
        verbose_name_plural = "Układy slajdera"

    def __str__(self) -> str:
        return f"{self.name}{' (wyświetlany)' if self.is_active else ''}"

    def save(self, *args, **kwargs) -> None:
        with transaction.atomic():
            if self.is_active:
                SliderPreset.objects.filter(is_active=True).exclude(pk=self.pk).update(is_active=False)
            super().save(*args, **kwargs)

    @classmethod
    def get_active(cls) -> SliderPreset | None:
        return cls.objects.filter(is_active=True).first()


class SliderPresetItem(models.Model):
    """Karta w układzie: zdjęcie z bazy + pozycja. Zmiana kolejności aktualizuje ``order``
    (bez usuwania i ponownego dodawania rekordów)."""

    preset = models.ForeignKey(
        SliderPreset,
        on_delete=models.CASCADE,
        related_name="preset_items",
        verbose_name="Układ",
    )
    gallery_item = models.ForeignKey(
        GalleryItem,
        on_delete=models.CASCADE,
        related_name="preset_appearances",
        verbose_name="Zdjęcie",
    )
    order = models.PositiveIntegerField("Kolejność", default=0, db_index=True)

    class Meta:
        ordering = ["order", "pk"]
        verbose_name = "Karta układu"
        verbose_name_plural = "Karty układu"
        constraints = [
            models.UniqueConstraint(fields=["preset", "gallery_item"], name="unique_preset_item"),
        ]

    def __str__(self) -> str:
        return f"{self.preset.name} – {self.gallery_item.title} (#{self.order})"
