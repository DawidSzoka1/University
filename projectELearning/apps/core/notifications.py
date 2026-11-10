"""Tworzenie powiadomień z dowolnego modułu.

    from apps.core.notifications import notify, notify_many
    from apps.core.models import Notification

    notify(uczen, Notification.Kind.COMMENT_APPROVED, "Twój komentarz został zaakceptowany",
           "Recenzja kursu „Python” jest już widoczna.", url=course_url)
    notify_many(nauczyciele, Notification.Kind.QUIZ_TO_GRADE, "Nowy test do sprawdzenia", url=grading_url)

``url`` to adres wewnątrz platformy (np. z ``reverse``) — panel otwiera go po kliknięciu.
"""

from __future__ import annotations

from collections.abc import Iterable

from apps.core.models import Notification

_TITLE_MAX = Notification._meta.get_field("title").max_length
_URL_MAX = Notification._meta.get_field("url").max_length


def notify(user, kind: str, title: str, message: str = "", url: str = "") -> Notification:
    """Tworzy jedno powiadomienie. Nieznany rodzaj to błąd programisty — ValueError."""
    if kind not in Notification.Kind.values:
        raise ValueError(f"Nieznany rodzaj powiadomienia: {kind!r}")
    if url and not url.startswith("/"):
        raise ValueError("Powiadomienie może prowadzić tylko do adresu wewnątrz platformy (zaczynającego się od /).")
    return Notification.objects.create(
        user=user,
        kind=kind,
        title=title[:_TITLE_MAX],
        message=message,
        url=url[:_URL_MAX],
    )


def notify_many(users: Iterable, kind: str, title: str, message: str = "", url: str = "", *, exclude=None) -> int:
    """To samo powiadomienie dla wielu osób (bez duplikatów i bez ``exclude``). Zwraca liczbę."""
    seen: set[int] = set()
    count = 0
    for user in users:
        if user is None or user.pk in seen or (exclude is not None and user.pk == exclude.pk):
            continue
        seen.add(user.pk)
        notify(user, kind, title, message, url)
        count += 1
    return count
