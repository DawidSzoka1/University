"""Liczba sztuk w koszyku do linku w nawigacji."""

from __future__ import annotations

from django.db.models import Sum
from django.http import HttpRequest

from apps.orders.models import CartItem


def cart_count(request: HttpRequest) -> dict:
    user = getattr(request, "user", None)
    if user is not None and user.is_authenticated:
        items = CartItem.objects.filter(cart__user=user)
    else:
        session = getattr(request, "session", None)
        cart_id = session.get("cart_id") if session is not None else None
        if not cart_id:
            return {"cart_count": 0}
        items = CartItem.objects.filter(cart_id=cart_id, cart__user__isnull=True)
    return {"cart_count": items.aggregate(total=Sum("quantity"))["total"] or 0}
