"""Koszyk: zalogowany ma jeden koszyk w bazie, gość — koszyk wskazany w sesji (``cart_id``).
Po zalogowaniu koszyk gościa jest scalany z koszykiem konta (ilości się sumują)."""

from __future__ import annotations

from dataclasses import dataclass

from django.db import transaction
from django.db.models import F, Value
from django.db.models.functions import Least

from apps.catalog.models import Course
from apps.orders.models import MAX_QUANTITY, Cart, CartItem

SESSION_KEY = "cart_id"


class CartError(ValueError):
    """Kursu nie da się dodać (bezpłatny, nieopublikowany, limit sztuk) — komunikat dla ucznia."""


@dataclass
class AddResult:
    item: CartItem
    created: bool
    capped: bool  # ilość obcięta do MAX_QUANTITY


def get_cart(request, create: bool = False) -> Cart | None:
    user = getattr(request, "user", None)
    if user is not None and user.is_authenticated:
        if create:
            return Cart.objects.get_or_create(user=user)[0]
        return Cart.objects.filter(user=user).first()
    cart_id = request.session.get(SESSION_KEY)
    cart = Cart.objects.filter(pk=cart_id, user__isnull=True).first() if cart_id else None
    if cart is None and create:
        cart = Cart.objects.create()
        request.session[SESSION_KEY] = cart.pk
    return cart


def check_purchasable(course: Course) -> None:
    if not (course.is_published and course.status == Course.Status.APPROVED):
        raise CartError("Ten kurs nie jest jeszcze dostępny w sprzedaży.")
    if course.is_course_free:
        raise CartError("Ten kurs jest bezpłatny — zapisz się na stronie kursu, bez koszyka.")


@transaction.atomic
def add_to_cart(cart: Cart, course: Course, quantity: int = 1) -> AddResult:
    """Dodaje kurs; ten sam kurs dodany ponownie łączy się w jedną pozycję z większą ilością."""
    check_purchasable(course)
    quantity = max(1, int(quantity))
    item, created = CartItem.objects.select_for_update().get_or_create(
        cart=cart, course=course, defaults={"quantity": min(quantity, MAX_QUANTITY)}
    )
    capped = quantity > MAX_QUANTITY if created else item.quantity + quantity > MAX_QUANTITY
    if not created:
        item.quantity = min(item.quantity + quantity, MAX_QUANTITY)
        item.save(update_fields=["quantity"])
    cart.save(update_fields=["updated_at"])
    return AddResult(item=item, created=created, capped=capped)


def set_quantity(item: CartItem, quantity: int) -> CartItem | None:
    """Nowa ilość pozycji; 0 usuwa ją z koszyka."""
    if quantity <= 0:
        item.delete()
        return None
    if quantity > MAX_QUANTITY:
        raise CartError(f"Jednego kursu możesz kupić najwyżej {MAX_QUANTITY} sztuk.")
    item.quantity = quantity
    item.save(update_fields=["quantity"])
    return item


@transaction.atomic
def merge_carts(source: Cart, target: Cart) -> None:
    """Przenosi pozycje koszyka gościa do koszyka konta (z kumulacją) i usuwa koszyk gościa."""
    for item in source.items.select_related("course"):
        existing = target.items.filter(course=item.course).first()
        if existing:
            CartItem.objects.filter(pk=existing.pk).update(
                quantity=Least(F("quantity") + item.quantity, Value(MAX_QUANTITY))
            )
        else:
            CartItem.objects.create(cart=target, course=item.course, quantity=item.quantity)
    source.delete()


def merge_guest_cart(request, user) -> None:
    """Wywoływane po zalogowaniu (sygnał ``user_logged_in``)."""
    cart_id = request.session.pop(SESSION_KEY, None) if hasattr(request, "session") else None
    if not cart_id:
        return
    guest = Cart.objects.filter(pk=cart_id, user__isnull=True).first()
    if guest is None or not guest.items.exists():
        if guest is not None:
            guest.delete()
        return
    merge_carts(guest, Cart.objects.get_or_create(user=user)[0])


def cart_items(cart: Cart | None):
    if cart is None:
        return CartItem.objects.none()
    return cart.items.select_related("course", "course__category")
