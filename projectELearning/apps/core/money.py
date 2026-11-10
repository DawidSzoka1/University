"""Kwoty po polsku: ``1234.5`` → ``1 234,50 zł`` (twarde spacje, kwota nie łamie się między wierszami)."""

from __future__ import annotations

from decimal import Decimal

NBSP = " "


def format_pln(value) -> str:
    whole, _, fraction = f"{Decimal(value or 0):.2f}".partition(".")
    sign = "-" if whole.startswith("-") else ""
    whole = whole.lstrip("-")
    groups = []
    while len(whole) > 3:
        groups.insert(0, whole[-3:])
        whole = whole[:-3]
    groups.insert(0, whole)
    return f"{sign}{NBSP.join(groups)},{fraction}{NBSP}zł"
