"""Zamówienia: złożenie z koszyka, zmiany statusu, wydanie dostępu po opłaceniu, kody i e-maile."""

from __future__ import annotations

import logging

from django.conf import settings
from django.core.mail import EmailMultiAlternatives
from django.db import IntegrityError, transaction
from django.template.loader import render_to_string
from django.urls import reverse
from django.utils import timezone

from apps.catalog.models import Enrollment
from apps.catalog.permissions import is_platform_admin
from apps.core.models import Notification
from apps.core.notifications import notify
from apps.orders.models import AccessCode, Cart, Order, OrderItem, OrderStatusChange

logger = logging.getLogger(__name__)

ADDRESS_FIELDS = ("first_name", "last_name", "email", "street_address", "postal_code", "city", "phone_number")
# E-mail konta służy do logowania — z zamówienia zapisujemy do profilu tylko dane adresowe
PROFILE_FIELDS = ("first_name", "last_name", "street_address", "postal_code", "city", "phone_number")


class OrderError(ValueError):
    """Operacji nie da się wykonać — komunikat dla użytkownika."""


def can_manage_orders(user) -> bool:
    """Statusy zamówień zmienia moderator (nauczyciel) albo administrator."""
    return user.is_authenticated and (is_platform_admin(user) or getattr(user, "role", "") == "moderator")


def log_status(order: Order, note: str = "", by=None) -> OrderStatusChange:
    return OrderStatusChange.objects.create(order=order, status=order.status, note=note[:255], changed_by=by)


# ---------------------------------------------------------------------------
# Złożenie zamówienia
# ---------------------------------------------------------------------------

@transaction.atomic
def place_order(user, cart: Cart, address: dict, save_address: bool) -> Order:
    """Tworzy zamówienie z koszyka (ceny z tej chwili), opcjonalnie zapisuje adres w profilu,
    czyści koszyk i wysyła e-mail „oczekuje na płatność”."""
    items = list(cart.items.select_related("course").select_for_update())
    if not items:
        raise OrderError("Koszyk jest pusty.")
    for item in items:
        if item.course.is_course_free or not item.course.is_published:
            raise OrderError(f"Kurs „{item.course.title}” nie jest już dostępny w sprzedaży — usuń go z koszyka.")

    for _ in range(5):  # numer kolejny w roku; przy równoległych zamówieniach ponawiamy
        try:
            with transaction.atomic():
                order = Order.objects.create(
                    number=Order.next_number(),
                    user=user,
                    status=Order.Status.NEW,
                    save_address=save_address,
                    **{field: address.get(field, "") for field in ADDRESS_FIELDS},
                )
            break
        except IntegrityError:
            continue
    else:  # pragma: no cover - praktycznie nieosiągalne
        raise OrderError("Nie udało się nadać numeru zamówienia. Spróbuj ponownie.")

    total = 0
    for item in items:
        line = item.course.price * item.quantity
        OrderItem.objects.create(
            order=order, course=item.course, title=item.course.title,
            quantity=item.quantity, unit_price=item.course.price, line_total=line,
        )
        total += line
    order.total = total
    order.status = Order.Status.PENDING_PAYMENT
    order.save(update_fields=["total", "status", "updated_at"])
    log_status(order, "Zamówienie złożone — oczekuje na płatność.", by=user)

    if save_address:
        for field in PROFILE_FIELDS:
            setattr(user, field, address.get(field, getattr(user, field)))
        user.save(update_fields=list(PROFILE_FIELDS))

    cart.items.all().delete()
    transaction.on_commit(lambda: send_order_email(order, "created"))
    return order


# ---------------------------------------------------------------------------
# Statusy
# ---------------------------------------------------------------------------

STATUS_MESSAGES = {
    Order.Status.NEW: "Zamówienie {number} ma status „Nowe”.",
    Order.Status.PENDING_PAYMENT: "Zamówienie {number} czeka na płatność.",
    Order.Status.PAID: "Płatność za zamówienie {number} została przyjęta. Kursy są już w „Moich kursach”.",
    Order.Status.COMPLETED: "Zamówienie {number} zostało zrealizowane.",
    Order.Status.CANCELLED: "Zamówienie {number} zostało anulowane.",
}


def _notify_status(order: Order) -> None:
    notify(
        order.user,
        Notification.Kind.ORDER_STATUS,
        f"Zamówienie {order.number}: {order.get_status_display().lower()}",
        STATUS_MESSAGES[order.status].format(number=order.number),
        url=order.get_absolute_url(),
    )


@transaction.atomic
def mark_paid(order: Order, *, provider: str, payment_intent: str = "", by=None, note: str = "") -> bool:
    """Oznacza zamówienie jako opłacone i wydaje dostęp. Idempotentne (webhook może przyjść kilka razy).
    Zwraca ``True``, jeśli status faktycznie się zmienił."""
    order = Order.objects.select_for_update().get(pk=order.pk)
    if order.status in (Order.Status.PAID, Order.Status.COMPLETED):
        return False
    order.status = Order.Status.PAID
    order.paid_at = timezone.now()
    order.payment_provider = provider
    if payment_intent:
        order.payment_intent = payment_intent
    order.save(update_fields=["status", "paid_at", "payment_provider", "payment_intent", "updated_at"])
    log_status(order, note or "Płatność przyjęta.", by=by)
    fulfill(order)
    transaction.on_commit(lambda: (_notify_status(order), send_order_email(order, "paid")))
    return True


def fulfill(order: Order) -> None:
    """Pierwsza sztuka kursu → zapis kupującego (jeśli jeszcze nie ma dostępu), pozostałe → kody dostępu."""
    if order.fulfilled_at:
        return
    for item in order.items.select_related("course"):
        codes = item.quantity
        _, enrolled = Enrollment.objects.get_or_create(user=order.user, course=item.course)
        if enrolled:
            codes -= 1
        for _ in range(codes):
            AccessCode.objects.create(order_item=item)
    order.fulfilled_at = timezone.now()
    order.save(update_fields=["fulfilled_at"])


ALLOWED_MANUAL = {
    Order.Status.NEW: {Order.Status.PENDING_PAYMENT, Order.Status.PAID, Order.Status.CANCELLED},
    Order.Status.PENDING_PAYMENT: {Order.Status.NEW, Order.Status.PAID, Order.Status.CANCELLED},
    Order.Status.PAID: {Order.Status.COMPLETED, Order.Status.CANCELLED},
    Order.Status.COMPLETED: {Order.Status.PAID},
    Order.Status.CANCELLED: {Order.Status.NEW, Order.Status.PENDING_PAYMENT},
}


@transaction.atomic
def change_status(order: Order, new_status: str, by, note: str = "") -> None:
    """Ręczna zmiana statusu przez moderatora/administratora (z historią i powiadomieniem)."""
    if new_status == order.status:
        raise OrderError("Zamówienie ma już ten status.")
    if new_status not in ALLOWED_MANUAL.get(order.status, set()):
        raise OrderError(
            f"Nie można zmienić statusu z „{order.get_status_display()}” na „{Order.Status(new_status).label}”."
        )
    if new_status == Order.Status.PAID and order.status != Order.Status.COMPLETED:
        mark_paid(order, provider=order.payment_provider or Order.Provider.MANUAL, by=by,
                  note=note or "Płatność potwierdzona ręcznie.")
        order.refresh_from_db()
        return
    order.status = new_status
    order.save(update_fields=["status", "updated_at"])
    log_status(order, note or "Status zmieniony przez obsługę.", by=by)
    transaction.on_commit(lambda: _notify_status(order))


def cancel_by_user(order: Order, user) -> None:
    if not order.can_cancel:
        raise OrderError("Tego zamówienia nie można już anulować.")
    order.status = Order.Status.CANCELLED
    order.save(update_fields=["status", "updated_at"])
    log_status(order, "Anulowane przez kupującego.", by=user)


# ---------------------------------------------------------------------------
# Kody dostępu
# ---------------------------------------------------------------------------

@transaction.atomic
def redeem_code(user, raw_code: str) -> AccessCode:
    code_text = (raw_code or "").strip().upper().replace(" ", "")
    code = (
        AccessCode.objects.select_for_update()
        .select_related("order_item__course", "order_item__order")
        .filter(code=code_text)
        .first()
    )
    if code is None or code.order_item.order.status not in (Order.Status.PAID, Order.Status.COMPLETED):
        raise OrderError("Nie znaleziono takiego kodu. Sprawdź, czy został przepisany bez błędów.")
    if code.redeemed_by_id:
        raise OrderError("Ten kod został już wykorzystany.")
    course = code.order_item.course
    if Enrollment.objects.filter(user=user, course=course).exists():
        raise OrderError(f"Masz już dostęp do kursu „{course.title}” — przekaż kod innej osobie.")
    Enrollment.objects.create(user=user, course=course)
    code.redeemed_by = user
    code.redeemed_at = timezone.now()
    code.save(update_fields=["redeemed_by", "redeemed_at"])
    return code


# ---------------------------------------------------------------------------
# E-maile
# ---------------------------------------------------------------------------

def email_palette() -> dict:
    """Kolory e-maila z domyślnego motywu (klienci poczty nie obsługują zmiennych CSS)."""
    from apps.core.colors import _active_theme, theme_palette

    theme = _active_theme()
    if theme is None:
        return theme_palette("#B45309", "#0F766E", "#FFFBEB")
    return theme_palette(theme.primary, theme.secondary, theme.surface, theme.accent, theme.surface_variant)


EMAIL_SUBJECTS = {
    "created": "Zamówienie {number} przyjęte — oczekuje na płatność",
    "paid": "Płatność za zamówienie {number} przyjęta",
}


def send_order_email(order: Order, kind: str) -> None:
    order = Order.objects.prefetch_related("items__codes").get(pk=order.pk)
    site = settings.SITE_URL.rstrip("/")
    context = {
        "order": order,
        "items": order.items.all(),
        "order_url": site + order.get_absolute_url(),
        "pay_url": site + reverse("orders:order_pay", kwargs={"number": order.number}),
        "courses_url": site + reverse("learning:my_courses"),
        "redeem_url": site + reverse("orders:redeem"),
        "codes": AccessCode.objects.filter(order_item__order=order).select_related("order_item"),
        "palette": email_palette(),
    }
    subject = EMAIL_SUBJECTS[kind].format(number=order.number)
    text = render_to_string(f"orders/emails/order_{kind}.txt", context)
    html = render_to_string(f"orders/emails/order_{kind}.html", context)
    message = EmailMultiAlternatives(subject, text, settings.DEFAULT_FROM_EMAIL, [order.email or order.user.email])
    message.attach_alternative(html, "text/html")
    try:
        message.send()
    except Exception:  # serwer poczty niedostępny nie może zablokować zamówienia
        logger.exception("Nie udało się wysłać e-maila %s dla %s", kind, order.number)
