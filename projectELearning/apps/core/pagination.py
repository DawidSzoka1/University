from __future__ import annotations

from typing import Any
from django.core.paginator import EmptyPage, Page, PageNotAnInteger, Paginator
from django.db.models import QuerySet
from django.http import HttpRequest


def paginate_queryset(
    request: HttpRequest,
    queryset: QuerySet[Any] | list[Any],
    per_page: int = 12,
    page_param: str = "page",
) -> tuple[Page, Paginator]:
    """Generyczna pomocnicza funkcja paginująca dowolny QuerySet lub listę.

    - Bezpiecznie obsługuje brak parametru, wartości nieliczbowe i przekroczenie zakresu.
    - Zwraca krotkę (page_obj, paginator).
    """
    paginator = Paginator(queryset, per_page)
    page_number = request.GET.get(page_param, 1)

    try:
        page_obj = paginator.page(page_number)
    except PageNotAnInteger:
        page_obj = paginator.page(1)
    except EmptyPage:
        # Jeśli numer strony jest za duży, zwróć ostatnią stronę; dla pustego QuerySetu stronę 1
        page_obj = paginator.page(paginator.num_pages if paginator.num_pages > 0 else 1)

    return page_obj, paginator
