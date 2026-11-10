"""Blokada logowania po serii nieudanych prób (ochrona przed zgadywaniem hasła).

Po ``MAX_FAILURES`` nieudanych próbach w ciągu ``WINDOW_SECONDS`` logowanie na to konto
(login albo e-mail — oba wskazują to samo konto) jest blokowane na ``LOCK_SECONDS``.
Udane logowanie zeruje licznik. Stan trzymany w pamięci podręcznej (Redis), nie w bazie.
"""

from __future__ import annotations

import hashlib
import math
import time

from django.core.cache import cache

MAX_FAILURES = 5
WINDOW_SECONDS = 15 * 60
LOCK_SECONDS = 5 * 60


def _key(login: str) -> str:
    from apps.accounts.models import CustomUser

    login = (login or "").strip()
    user = CustomUser.objects.filter(username__iexact=login).first()
    if user is None and "@" in login:
        user = CustomUser.objects.filter(email__iexact=login).first()
    # Nieznany login: skrót zamiast surowego tekstu (bezpieczny klucz pamięci podręcznej)
    ident = f"user-{user.pk}" if user else "name-" + hashlib.sha256(login.lower().encode()).hexdigest()[:32]
    return f"login-lockout:{ident}"


def locked_for(login: str) -> int:
    """Ile sekund jeszcze trwa blokada (0 = brak blokady)."""
    until = cache.get(_key(login) + ":until")
    return max(0, math.ceil(until - time.time())) if until else 0


def register_failure(login: str) -> int:
    """Zapisuje nieudaną próbę. Zwraca liczbę pozostałych prób (0 = właśnie zablokowano)."""
    key = _key(login)
    failures = cache.get(key + ":count", 0) + 1
    if failures >= MAX_FAILURES:
        cache.set(key + ":until", time.time() + LOCK_SECONDS, LOCK_SECONDS)
        cache.delete(key + ":count")
        return 0
    cache.set(key + ":count", failures, WINDOW_SECONDS)
    return MAX_FAILURES - failures


def reset(login: str) -> None:
    key = _key(login)
    cache.delete_many([key + ":count", key + ":until"])


def minutes_text(seconds: int) -> str:
    from apps.core.templatetags.core_tags import pl_plural

    minutes = max(1, math.ceil(seconds / 60))
    return f"{minutes} {pl_plural(minutes, 'minutę,minuty,minut')}"
