"""Dyplom ukończenia kursu w PDF (reportlab, A4 poziomo, fonty OFL z polskimi znakami).

Tekst w PDF jest prawdziwym tekstem (da się go zaznaczyć i przeczytać czytnikiem ekranu),
a dokument ma tytuł, autora i język w metadanych. Kolor akcentu pochodzi z domyślnego motywu.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from django.utils import formats, timezone
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import simpleSplit
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

from apps.learning.models import Certificate

FONTS = Path(__file__).resolve().parent / "assets" / "fonts"
BODY, DISPLAY = "Atkinson", "Bricolage"
INK = HexColor("#111827")
MUTED = HexColor("#4B5563")
PAPER = HexColor("#FFFFFF")


def _register_fonts() -> None:
    if BODY not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(BODY, str(FONTS / "AtkinsonHyperlegible-Regular.ttf")))
        pdfmetrics.registerFont(TTFont(DISPLAY, str(FONTS / "BricolageGrotesque.ttf")))


def _accent() -> HexColor:
    from apps.core.models import ColorTheme

    theme = ColorTheme.objects.filter(is_default=True).first()
    return HexColor(theme.primary if theme else "#B45309")


def _split(text: str, font: str, size: float, width: float) -> list[str]:
    """Łamanie po słowach; zbyt długie „słowa” (np. adres URL) dzielone po znakach."""
    lines: list[str] = []
    for line in simpleSplit(text, font, size, width):
        while pdfmetrics.stringWidth(line, font, size) > width and len(line) > 1:
            cut = len(line)
            while cut > 1 and pdfmetrics.stringWidth(line[:cut], font, size) > width:
                cut -= 1
            lines.append(line[:cut])
            line = line[cut:]
        lines.append(line)
    return lines


def _fit(text: str, font: str, max_size: float, min_size: float, width: float, max_lines: int) -> tuple[float, list[str]]:
    """Największy rozmiar fontu, przy którym tekst mieści się w ``max_lines`` liniach."""
    size = max_size
    while size > min_size:
        lines = _split(text, font, size, width)
        if len(lines) <= max_lines:
            return size, lines
        size -= 1
    return min_size, _split(text, font, min_size, width)[:max_lines]


def render_certificate(certificate: Certificate, verify_url: str) -> bytes:
    _register_fonts()
    accent = _accent()
    width, height = landscape(A4)
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=(width, height), lang="pl-PL")
    pdf.setTitle(f"Dyplom: {certificate.course_title} — {certificate.full_name}")
    pdf.setAuthor("E-Learning")
    pdf.setSubject(f"Dyplom ukończenia kursu, nr {certificate.code}")
    pdf.setCreator("Platforma E-Learning")

    pdf.setFillColor(PAPER)
    pdf.rect(0, 0, width, height, stroke=0, fill=1)

    # Ramka w stylu „zeszytu”: gruba linia + cienka wewnętrzna, pasek akcentu u góry
    margin = 30
    pdf.setStrokeColor(INK)
    pdf.setLineWidth(2.5)
    pdf.rect(margin, margin, width - 2 * margin, height - 2 * margin)
    pdf.setLineWidth(0.6)
    pdf.rect(margin + 8, margin + 8, width - 2 * margin - 16, height - 2 * margin - 16)
    pdf.setFillColor(accent)
    pdf.rect(margin, height - margin - 10, width - 2 * margin, 10, stroke=0, fill=1)

    left = margin + 46
    right = width - margin - 46
    content_width = right - left

    # Znak marki i numer
    top = height - margin - 64
    pdf.setFillColor(accent)
    pdf.rect(left, top - 6, 30, 30, stroke=0, fill=1)
    pdf.setFillColor(PAPER)
    pdf.setFont(DISPLAY, 20)
    pdf.drawCentredString(left + 15, top + 2, "e")
    pdf.setFillColor(INK)
    pdf.setFont(DISPLAY, 16)
    pdf.drawString(left + 40, top + 3, "E-Learning")
    pdf.setFont(BODY, 10)
    pdf.setFillColor(MUTED)
    pdf.drawRightString(right, top + 4, f"Nr {certificate.code}")

    # Nagłówek
    y = top - 58
    pdf.setFillColor(accent)
    pdf.setFont(BODY, 12)
    pdf.drawString(left, y, "D Y P L O M   U K O Ń C Z E N I A   K U R S U")
    pdf.setStrokeColor(INK)
    pdf.setLineWidth(1)
    pdf.line(left, y - 12, right, y - 12)

    # Treść: kurs → ukończony przez → imię i nazwisko (forma neutralna płciowo)
    y -= 46
    pdf.setFillColor(MUTED)
    pdf.setFont(BODY, 13)
    pdf.drawString(left, y, "Potwierdzamy, że kurs")
    size, lines = _fit(certificate.course_title, DISPLAY, 30, 18, content_width, 2)
    pdf.setFillColor(INK)
    pdf.setFont(DISPLAY, size)
    for line in lines:
        y -= size + 6
        pdf.drawString(left, y, line)
    y -= 30
    pdf.setFillColor(MUTED)
    pdf.setFont(BODY, 13)
    pdf.drawString(left, y, "został ukończony przez")
    size, lines = _fit(certificate.full_name, DISPLAY, 46, 24, content_width, 1)
    y -= size + 10
    pdf.setFillColor(INK)
    pdf.setFont(DISPLAY, size)
    pdf.drawString(left, y, lines[0])
    pdf.setStrokeColor(accent)
    pdf.setLineWidth(3)
    pdf.line(left, y - 10, left + min(content_width, pdf.stringWidth(lines[0], DISPLAY, size)), y - 10)

    y -= 40
    pdf.setFillColor(MUTED)
    pdf.setFont(BODY, 11)
    pdf.drawString(left, y, "Wszystkie lekcje ukończone, a testy kursu zaliczone.")

    # Stopka: trzy kolumny oddzielone liniami
    base = margin + 52
    pdf.setStrokeColor(INK)
    pdf.setLineWidth(1)
    pdf.line(left, base + 34, right, base + 34)
    column = content_width / 3
    issued = formats.date_format(timezone.localtime(certificate.issued_at), "j E Y")
    entries = [
        ("Data wystawienia", issued),
        ("Prowadzący", certificate.teacher_name or "—"),
        ("Sprawdź autentyczność", verify_url),
    ]
    for index, (label, value) in enumerate(entries):
        x = left + index * column
        if index:
            pdf.line(x - 12, base - 6, x - 12, base + 34)
        pdf.setFillColor(MUTED)
        pdf.setFont(BODY, 9)
        pdf.drawString(x, base + 18, label.upper())
        pdf.setFillColor(INK)
        value_size, value_lines = _fit(value, BODY, 12, 7, column - 24, 2)
        pdf.setFont(BODY, value_size)
        for line_index, line in enumerate(value_lines):
            pdf.drawString(x, base - line_index * (value_size + 2), line)
    pdf.linkURL(verify_url, (left + 2 * column, base - 14, right, base + 30), relative=0)

    pdf.showPage()
    pdf.save()
    return buffer.getvalue()
