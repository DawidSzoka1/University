"""Filmy i pliki lekcji: rozpoznawanie linków wideo i walidacja przesyłanych plików."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from django.core.exceptions import ValidationError

VIDEO_MAX_MB = 200
ATTACHMENT_MAX_MB = 25
ATTACHMENT_EXTENSIONS = {
    "pdf", "txt", "md", "csv", "json", "xml", "py", "ipynb", "sql", "js", "html", "css",
    "zip", "docx", "xlsx", "pptx", "odt", "ods", "png", "jpg", "jpeg",
    "yml", "yaml", "toml", "ini", "sh", "java", "c", "cpp", "h",
}
# Pliki konfiguracyjne bez rozszerzenia, typowe w kursach DevOps
ATTACHMENT_EXACT_NAMES = {"Dockerfile", "Makefile", ".gitignore", ".env.example"}

_YT_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")


def video_embed_url(url: str) -> str | None:
    """Adres odtwarzacza dla linku YouTube/Vimeo albo None, gdy link nie jest obsługiwany."""
    if not url:
        return None
    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower().removeprefix("www.").removeprefix("m.")
    video_id = None
    if host in {"youtube.com", "youtube-nocookie.com"}:
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [None])[0]
        elif parsed.path.startswith(("/embed/", "/shorts/", "/live/")):
            video_id = parsed.path.split("/")[2]
        if video_id and _YT_ID.match(video_id):
            # Tryb bez ciasteczek śledzących
            return f"https://www.youtube-nocookie.com/embed/{video_id}"
        return None
    if host == "youtu.be":
        video_id = parsed.path.lstrip("/").split("/")[0]
        return f"https://www.youtube-nocookie.com/embed/{video_id}" if _YT_ID.match(video_id) else None
    if host in {"vimeo.com", "player.vimeo.com"}:
        match = re.search(r"(\d{6,12})", parsed.path)
        return f"https://player.vimeo.com/video/{match.group(1)}" if match else None
    return None


def validate_video_url(url: str) -> None:
    if url and video_embed_url(url) is None:
        raise ValidationError("Obsługujemy linki do YouTube i Vimeo, np. https://www.youtube.com/watch?v=…")


def validate_video_file(upload) -> None:
    """MP4/WebM do 200 MB — sprawdzane rozszerzenie i nagłówek pliku, nie tylko nazwa."""
    if upload.size > VIDEO_MAX_MB * 1024 * 1024:
        raise ValidationError(f"Film jest za duży — maksymalnie {VIDEO_MAX_MB} MB.")
    extension = Path(upload.name).suffix.lower()
    upload.seek(0)
    head = upload.read(16)
    upload.seek(0)
    is_mp4 = extension in {".mp4", ".m4v"} and head[4:8] == b"ftyp"
    is_webm = extension == ".webm" and head[:4] == b"\x1a\x45\xdf\xa3"
    if not (is_mp4 or is_webm):
        raise ValidationError("Dozwolone formaty filmu: MP4 lub WebM.")


def validate_attachment(upload) -> None:
    if upload.size > ATTACHMENT_MAX_MB * 1024 * 1024:
        raise ValidationError(f"Plik „{upload.name}” jest za duży — maksymalnie {ATTACHMENT_MAX_MB} MB.")
    name = Path(upload.name).name
    extension = Path(name).suffix.lower().lstrip(".")
    if name not in ATTACHMENT_EXACT_NAMES and extension not in ATTACHMENT_EXTENSIONS:
        allowed = ", ".join(sorted(ATTACHMENT_EXTENSIONS))
        raise ValidationError(f"Plik „{upload.name}” ma niedozwolony typ. Dozwolone: {allowed}.")
