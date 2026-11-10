"""Stripe Checkout (tryb testowy): sesja płatności, powrót z bramki i webhook zmieniający status."""

from __future__ import annotations

import logging
from pathlib import Path

import stripe
from django.conf import settings
from django.urls import reverse

from apps.orders.models import Order
from apps.orders.services import log_status, mark_paid

logger = logging.getLogger(__name__)

PAID_EVENTS = {"checkout.session.completed", "checkout.session.async_payment_succeeded"}


class PaymentError(Exception):
    """Bramka niedostępna albo źle skonfigurowana — komunikat dla użytkownika."""


def is_configured() -> bool:
    return bool(settings.STRIPE_SECRET_KEY)


def is_test_mode() -> bool:
    return settings.STRIPE_SECRET_KEY.startswith(("sk_test_", "rk_test_"))


def to_minor_units(amount) -> int:
    """Kwota w groszach (Stripe przyjmuje liczby całkowite)."""
    return int((amount * 100).quantize(1))


def create_checkout_session(order: Order) -> str:
    """Tworzy sesję Stripe Checkout dla zamówienia i zwraca adres strony płatności."""
    if not is_configured():
        raise PaymentError("Płatności nie są skonfigurowane (brak STRIPE_SECRET_KEY w .env).")
    site = settings.SITE_URL.rstrip("/")
    success = site + reverse("orders:payment_return", kwargs={"number": order.number})
    cancel = site + reverse("orders:payment_cancel", kwargs={"number": order.number})
    line_items = [
        {
            "quantity": item.quantity,
            "price_data": {
                "currency": settings.STRIPE_CURRENCY,
                "unit_amount": to_minor_units(item.unit_price),
                "product_data": {"name": item.title, "metadata": {"course_id": str(item.course_id)}},
            },
        }
        for item in order.items.all()
    ]
    try:
        session = stripe.checkout.Session.create(
            api_key=settings.STRIPE_SECRET_KEY,
            mode="payment",
            line_items=line_items,
            customer_email=order.email or None,
            client_reference_id=order.number,
            metadata={"order_number": order.number},
            payment_intent_data={"metadata": {"order_number": order.number}},
            locale="pl",
            success_url=f"{success}?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=cancel,
        )
    except stripe.StripeError as error:
        logger.warning("Stripe: nie udało się utworzyć sesji dla %s: %s", order.number, error)
        raise PaymentError("Bramka płatności jest chwilowo niedostępna. Spróbuj ponownie za chwilę.") from error

    order.payment_provider = Order.Provider.STRIPE
    order.payment_reference = session.id
    fields = ["payment_provider", "payment_reference", "updated_at"]
    if order.status == Order.Status.NEW:
        order.status = Order.Status.PENDING_PAYMENT
        fields.append("status")
    order.save(update_fields=fields)
    if "status" in fields:
        log_status(order, "Przekazano do bramki płatności.")
    return session.url


def _apply_session(order: Order, session) -> bool:
    """Sprawdza sesję Stripe i oznacza zamówienie jako opłacone. Kwota i waluta muszą się zgadzać."""
    if session.get("payment_status") != "paid":
        return False
    if session.get("amount_total") != to_minor_units(order.total) or session.get("currency") != settings.STRIPE_CURRENCY:
        logger.error("Stripe: kwota/waluta sesji %s nie zgadza się z zamówieniem %s", session.get("id"), order.number)
        return False
    return mark_paid(
        order,
        provider=Order.Provider.STRIPE,
        payment_intent=str(session.get("payment_intent") or ""),
        note="Płatność potwierdzona przez Stripe.",
    )


def confirm_from_return(order: Order, session_id: str) -> bool:
    """Powrót ze Stripe (success_url): sprawdza sesję po stronie serwera — działa też bez webhooka."""
    if not is_configured() or not session_id or session_id != order.payment_reference:
        return False
    try:
        session = stripe.checkout.Session.retrieve(session_id, api_key=settings.STRIPE_SECRET_KEY)
    except stripe.StripeError as error:
        logger.warning("Stripe: nie udało się pobrać sesji %s: %s", session_id, error)
        return False
    return _apply_session(order, session)


def sync_order(order: Order) -> bool:
    """Zapasowe sprawdzenie bez webhooka: oczekujące zamówienie z sesją Stripe → zapytaj Stripe,
    czy zostało opłacone (np. kupujący zamknął kartę przed powrotem na platformę)."""
    if order.status != Order.Status.PENDING_PAYMENT or not order.payment_reference.startswith("cs_"):
        return False
    return confirm_from_return(order, order.payment_reference)


def webhook_secret() -> str:
    """``STRIPE_WEBHOOK_SECRET`` z .env albo plik zapisany przez usługę stripe-cli (czytany przy
    każdym zdarzeniu, więc nowy sekret działa bez restartu aplikacji)."""
    if settings.STRIPE_WEBHOOK_SECRET:
        return settings.STRIPE_WEBHOOK_SECRET
    try:
        return Path(settings.STRIPE_WEBHOOK_SECRET_FILE).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def handle_webhook(payload: bytes, signature: str) -> str:
    """Weryfikuje podpis i obsługuje zdarzenie. Zwraca krótki opis (do logu / odpowiedzi)."""
    secret = webhook_secret()
    if not secret:
        raise PaymentError("Brak sekretu webhooka (STRIPE_WEBHOOK_SECRET albo .stripe/webhook_secret).")
    try:
        event = stripe.Webhook.construct_event(payload, signature, secret)
    except (ValueError, stripe.SignatureVerificationError) as error:
        raise PaymentError("Nieprawidłowy podpis webhooka.") from error

    if event["type"] not in PAID_EVENTS:
        return f"pominięto {event['type']}"
    session = event["data"]["object"]
    number = (session.get("metadata") or {}).get("order_number") or session.get("client_reference_id")
    order = Order.objects.filter(number=number).first()
    if order is None:
        return f"brak zamówienia {number}"
    changed = _apply_session(order, session)
    return f"{order.number}: {'opłacone' if changed else 'bez zmian'}"
