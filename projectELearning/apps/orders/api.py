"""API koszyka chronione JWT (``Authorization: Bearer …``).

Te same reguły co w formularzach koszyka (``apps.orders.cart``): kumulacja ilości przy ponownym
dodaniu, limit ``MAX_QUANTITY``, tylko opublikowane płatne kursy. Koszyk należy do konta z tokenu;
koszyk gościa (sesja) obsługują zwykłe formularze. Strona koszyka używa tego API do zmiany ilości
bez przeładowania (``static/js/cart.js``), token pobiera przez ``accounts:api_session_token``.

- ``GET    /orders/api/koszyk/``                 — zawartość koszyka i podsumowanie
- ``POST   /orders/api/koszyk/pozycje/``         — ``{"course": "<slug lub id>", "quantity": 1}``
- ``PATCH  /orders/api/koszyk/pozycje/<id>/``    — ``{"quantity": n}`` (0 usuwa pozycję)
- ``DELETE /orders/api/koszyk/pozycje/<id>/``
"""

from __future__ import annotations

from django.urls import reverse
from rest_framework import status
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework_simplejwt.authentication import JWTAuthentication

from apps.catalog.models import Course, Enrollment
from apps.core.money import format_pln
from apps.core.templatetags.core_tags import pl_plural
from apps.orders.cart import CartError, add_to_cart, cart_items, set_quantity
from apps.orders.models import MAX_QUANTITY, Cart, CartItem

jwt_only = authentication_classes([JWTAuthentication])
authenticated = permission_classes([IsAuthenticated])


def _pieces(count: int) -> str:
    return f"{count} {pl_plural(count, 'sztuka,sztuki,sztuk')}"


def item_note(quantity: int, owned: bool) -> str:
    """Ten sam opis co na stronie koszyka: która sztuka odblokuje kurs, a które są kodami."""
    if owned:
        if quantity == 1:
            return "Masz już ten kurs — ta sztuka to kod dostępu dla innej osoby."
        return f"Masz już ten kurs — wszystkie {_pieces(quantity)} to kody dostępu dla innych osób."
    codes = quantity - 1
    if codes:
        return f"1 sztuka dla Ciebie + {codes} {pl_plural(codes, 'kod,kody,kodów')} dostępu dla innych osób."
    return ""


def serialize_item(item: CartItem, owned: bool) -> dict:
    course = item.course
    return {
        "id": item.pk,
        "course": {
            "id": course.pk,
            "slug": course.slug,
            "title": course.title,
            "url": reverse("catalog:course_detail", kwargs={"slug": course.slug}),
        },
        "quantity": item.quantity,
        "unit_price": str(item.unit_price),
        "unit_price_display": format_pln(item.unit_price),
        "line_total": str(item.line_total),
        "line_total_display": format_pln(item.line_total),
        # Pierwsza sztuka odblokowuje kurs kupującemu, chyba że już go ma — reszta to kody dostępu.
        "owned": owned,
        "code_count": item.quantity if owned else item.quantity - 1,
        "note": item_note(item.quantity, owned),
    }


def serialize_cart(user, cart: Cart | None) -> dict:
    items = list(cart_items(cart))
    owned = set(
        Enrollment.objects.filter(user=user, course__in=[i.course_id for i in items]).values_list("course_id", flat=True)
    )
    total = sum((item.line_total for item in items), 0)
    count = sum(item.quantity for item in items)
    return {
        "items": [serialize_item(item, item.course_id in owned) for item in items],
        "count": count,
        "count_display": _pieces(count),
        "total": str(total),
        "total_display": format_pln(total),
        "max_quantity": MAX_QUANTITY,
    }


def _user_cart(user, create: bool = False) -> Cart | None:
    if create:
        return Cart.objects.get_or_create(user=user)[0]
    return Cart.objects.filter(user=user).first()


def _find_course(value) -> Course | None:
    if value in (None, ""):
        return None
    lookup = {"pk": value} if str(value).isdigit() else {"slug": str(value)}
    return Course.objects.filter(**lookup).first()


def _parse_quantity(value, minimum: int) -> int | None:
    try:
        quantity = int(value)
    except (TypeError, ValueError):
        return None
    return quantity if quantity >= minimum else None


def _error(message: str, code: int = status.HTTP_400_BAD_REQUEST) -> Response:
    return Response({"detail": message}, status=code)


@api_view(["GET"])
@jwt_only
@authenticated
def api_cart(request):
    return Response(serialize_cart(request.user, _user_cart(request.user)))


@api_view(["POST"])
@jwt_only
@authenticated
def api_cart_add(request):
    """Dodaje kurs; ten sam kurs dodany ponownie zwiększa ilość istniejącej pozycji."""
    course = _find_course(request.data.get("course"))
    if course is None:
        return _error("Nie ma takiego kursu.", status.HTTP_404_NOT_FOUND)
    quantity = _parse_quantity(request.data.get("quantity", 1), minimum=1)
    if quantity is None:
        return _error(f"Ilość musi być liczbą od 1 do {MAX_QUANTITY}.")
    cart = _user_cart(request.user, create=True)
    try:
        result = add_to_cart(cart, course, quantity)
    except CartError as error:
        return _error(str(error))
    if result.capped:
        detail = f"W koszyku jest już maksymalnie {MAX_QUANTITY} szt. kursu „{course.title}”."
    elif result.created:
        detail = f"Dodano „{course.title}” do koszyka."
    else:
        detail = f"Kurs „{course.title}” był już w koszyku — teraz {_pieces(result.item.quantity)}."
    return Response(
        {
            "detail": detail,
            "created": result.created,
            "capped": result.capped,
            "item_id": result.item.pk,
            "cart": serialize_cart(request.user, cart),
        },
        status=status.HTTP_201_CREATED if result.created else status.HTTP_200_OK,
    )


@api_view(["PATCH", "DELETE"])
@jwt_only
@authenticated
def api_cart_item(request, pk: int):
    """PATCH zmienia ilość (0 usuwa pozycję), DELETE usuwa pozycję. Tylko pozycje własnego koszyka."""
    cart = _user_cart(request.user)
    item = CartItem.objects.select_related("course").filter(pk=pk, cart=cart).first() if cart else None
    if item is None:
        return _error("Tej pozycji nie ma w Twoim koszyku.", status.HTTP_404_NOT_FOUND)
    title = item.course.title

    if request.method == "DELETE":
        item.delete()
        return Response({"detail": f"Usunięto „{title}” z koszyka.", "item_id": None, "cart": serialize_cart(request.user, cart)})

    quantity = _parse_quantity(request.data.get("quantity"), minimum=0)
    if quantity is None:
        return _error(f"Ilość musi być liczbą od 0 do {MAX_QUANTITY}.")
    try:
        updated = set_quantity(item, quantity)
    except CartError as error:
        return _error(str(error))
    cart.save(update_fields=["updated_at"])
    if updated is None:
        detail = f"Usunięto „{title}” z koszyka."
    else:
        detail = f"„{title}”: {_pieces(updated.quantity)}."
    return Response({"detail": detail, "item_id": updated.pk if updated else None, "cart": serialize_cart(request.user, cart)})
