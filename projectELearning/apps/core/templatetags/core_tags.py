from __future__ import annotations

from typing import Any
from django import template
from django.http import QueryDict

register = template.Library()


@register.simple_tag(takes_context=True)
def url_replace(context: dict[str, Any], **kwargs: Any) -> str:
    """Aktualizuje lub usuwa parametry GET zapytania, zachowując pozostałe.

    Przydatne przy paginacji i filtrach, np.:
      <a href="?{% url_replace page=page_obj.next_page_number %}">Następna</a>
      <a href="?{% url_replace sort='popular' page=1 %}">Sortuj</a>
      <a href="?{% url_replace category='' %}">Wyczyść kategorię</a>
    """
    request = context.get("request")
    if not request:
        query = QueryDict(mutable=True)
    else:
        query = request.GET.copy()

    for key, value in kwargs.items():
        if value is None or value == "":
            query.pop(key, None)
        else:
            query[key] = str(value)

    return query.urlencode()


@register.simple_tag(takes_context=True)
def page_query(context: dict[str, Any], param: str, number: Any) -> str:
    """Query string z numerem strony pod dowolnym parametrem (kilka list na jednej stronie)."""
    request = context.get("request")
    query = request.GET.copy() if request else QueryDict(mutable=True)
    query[param or "page"] = str(number)
    return query.urlencode()


@register.filter
def pl_plural(value: Any, forms: str) -> str:
    """Polska odmiana rzeczownika po liczbie: ``{{ n|pl_plural:"kurs,kursy,kursów" }}``.

    1 → forma 1; 2–4 (bez 12–14) → forma 2; pozostałe → forma 3.
    """
    one, few, many = (part.strip() for part in forms.split(","))
    try:
        n = abs(int(value))
    except (TypeError, ValueError):
        return many
    if n == 1:
        return one
    if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
        return few
    return many


@register.filter
def time_ago(value) -> str:
    """„2 godziny temu” — jedna, najważniejsza jednostka; poniżej minuty „przed chwilą”."""
    from django.utils import timezone
    from django.utils.timesince import timesince

    if not value:
        return ""
    if (timezone.now() - value).total_seconds() < 60:
        return "przed chwilą"
    return f"{timesince(value, depth=1)} temu"


@register.filter
def pln(value) -> str:
    """Kwota po polsku: ``{{ order.total|pln }}`` → „1 234,50 zł”."""
    from apps.core.money import format_pln

    return format_pln(value)
