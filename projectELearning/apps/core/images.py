"""Walidacja i lokalne skalowanie obrazów (Pillow) — bez usług w chmurze."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from django.core.exceptions import ValidationError
from django.core.files.base import ContentFile
from PIL import Image, ImageOps, UnidentifiedImageError

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
MAX_UPLOAD_MB = 5


def validate_image_upload(upload, max_mb: int = MAX_UPLOAD_MB) -> None:
    """Sprawdza rozmiar pliku i to, czy to naprawdę obraz JPEG/PNG/WebP."""
    if upload.size > max_mb * 1024 * 1024:
        raise ValidationError(f"Plik jest za duży — maksymalnie {max_mb} MB.")
    try:
        upload.seek(0)
        with Image.open(upload) as image:
            image_format = image.format
            image.verify()
    except (UnidentifiedImageError, OSError, SyntaxError):
        raise ValidationError("To nie jest poprawny plik graficzny.")
    finally:
        upload.seek(0)
    if image_format not in ALLOWED_FORMATS:
        raise ValidationError("Dozwolone formaty: JPEG, PNG lub WebP.")


def resize_image_field(field_file, max_size: tuple[int, int]) -> None:
    """Skaluje świeżo przesłany obraz do ``max_size`` (zachowuje proporcje i orientację EXIF)."""
    if not field_file or getattr(field_file, "_committed", True):
        return
    field_file.seek(0)
    with Image.open(field_file) as source:
        image = ImageOps.exif_transpose(source)
        image.thumbnail(max_size)
        has_alpha = image.mode in ("RGBA", "LA", "P")
        buffer = BytesIO()
        if has_alpha:
            image.save(buffer, "PNG", optimize=True)
            extension = ".png"
        else:
            image.convert("RGB").save(buffer, "JPEG", quality=85, optimize=True)
            extension = ".jpg"
    field_file.save(Path(field_file.name).stem + extension, ContentFile(buffer.getvalue()), save=False)
