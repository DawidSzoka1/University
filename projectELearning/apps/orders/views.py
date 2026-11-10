"""Koszyk, zamówienie z adresem, płatność Stripe (sandbox), webhook, obsługa zamówień i kody dostępu."""

from __future__ import annotations

import logging

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db.models import Count, Q
from django.http import Http404, HttpRequest, HttpResponse, HttpResponseRedirect
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils.http import url_has_allowed_host_and_scheme
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from apps.catalog.models import Course, Enrollment
from apps.core.pagination import paginate_queryset
from apps.core.templatetags.core_tags import pl_plural
from apps.orders import payments
from apps.orders.cart import CartError, add_to_cart, cart_items, get_cart, set_quantity
from apps.orders.forms import CheckoutForm, OrderStatusForm, QuantityForm, RedeemForm
from apps.orders.models import MAX_QUANTITY, CartItem, Order
from apps.orders.services import (
    OrderError,
    can_manage_orders,
    cancel_by_user,
    change_status,
    place_order,
    redeem_code,
)

logger = logging.getLogger(__name__)


def _safe_next(request: HttpRequest, fallback: str) -> str:
    target = request.POST.get("next") or request.GET.get("next")
    if target and url_has_allowed_host_and_scheme(target, allowed_hosts={request.get_host()}):
        return target
    return fallback


# ---------------------------------------------------------------------------
# Koszyk (także dla gościa)
# ---------------------------------------------------------------------------

def cart_detail(request: HttpRequest) -> HttpResponse:
    cart = get_cart(request)
    items = list(cart_items(cart))
    owned = set()
    if request.user.is_authenticated and items:
        owned = set(
            Enrollment.objects.filter(user=request.user, course__in=[i.course for i in items]).values_list("course_id", flat=True)
        )
    for item in items:
        item.owned = item.course_id in owned
        item.code_count = item.quantity if item.owned else item.quantity - 1
    total = sum((item.line_total for item in items), 0)
    return render(
        request,
        "orders/cart.html",
        {"items": items, "total": total, "max_quantity": MAX_QUANTITY, "payments_ready": payments.is_configured()},
    )


@require_POST
def cart_add(request: HttpRequest, slug: str) -> HttpResponse:
    course = get_object_or_404(Course, slug=slug)
    fallback = reverse("catalog:course_detail", kwargs={"slug": course.slug})
    try:
        result = add_to_cart(get_cart(request, create=True), course)
    except CartError as error:
        messages.error(request, str(error))
        return redirect(_safe_next(request, fallback))
    count = result.item.quantity
    if result.capped:
        messages.warning(request, f"W koszyku jest już maksymalnie {MAX_QUANTITY} szt. kursu „{course.title}”.")
    elif result.created:
        messages.success(request, f"Dodano „{course.title}” do koszyka.")
    else:
        messages.success(
            request,
            f"Kurs „{course.title}” był już w koszyku — teraz {count} {pl_plural(count, 'sztuka,sztuki,sztuk')}.",
        )
    return redirect(_safe_next(request, fallback))


def _cart_item(request: HttpRequest, pk: int) -> CartItem:
    cart = get_cart(request)
    if cart is None:
        raise Http404
    return get_object_or_404(CartItem.objects.select_related("course"), pk=pk, cart=cart)


@require_POST
def cart_update(request: HttpRequest, pk: int) -> HttpResponse:
    item = _cart_item(request, pk)
    step = request.POST.get("step")
    if step in ("+1", "-1"):
        quantity = item.quantity + int(step)
    else:
        form = QuantityForm(request.POST)
        if not form.is_valid():
            messages.error(request, f"Ilość musi być liczbą od 0 do {MAX_QUANTITY}.")
            return redirect("orders:cart")
        quantity = form.cleaned_data["quantity"]
    title = item.course.title
    try:
        updated = set_quantity(item, quantity)
    except CartError as error:
        messages.error(request, str(error))
        return redirect("orders:cart")
    if updated is None:
        messages.info(request, f"Usunięto „{title}” z koszyka.")
    else:
        messages.success(request, f"„{title}”: {updated.quantity} {pl_plural(updated.quantity, 'sztuka,sztuki,sztuk')}.")
    return redirect("orders:cart")


@require_POST
def cart_remove(request: HttpRequest, pk: int) -> HttpResponse:
    item = _cart_item(request, pk)
    title = item.course.title
    item.delete()
    messages.info(request, f"Usunięto „{title}” z koszyka.")
    return redirect("orders:cart")


# ---------------------------------------------------------------------------
# Zamówienie i płatność
# ---------------------------------------------------------------------------

@login_required
def checkout(request: HttpRequest) -> HttpResponse:
    cart = get_cart(request)
    items = list(cart_items(cart))
    if not items:
        messages.info(request, "Koszyk jest pusty — dodaj kurs ze strony katalogu.")
        return redirect("orders:cart")
    form = CheckoutForm(request.POST or None, initial=CheckoutForm.initial_from(request.user))
    if request.method == "POST" and form.is_valid():
        data = form.cleaned_data
        try:
            order = place_order(request.user, cart, data, data["save_address"])
        except OrderError as error:
            messages.error(request, str(error))
            return redirect("orders:cart")
        if data["save_address"]:
            messages.info(request, "Zapisaliśmy dane adresowe w Twoim profilu.")
        return _start_payment(request, order, first=True)
    total = sum((item.line_total for item in items), 0)
    return render(
        request,
        "orders/checkout.html",
        {"form": form, "items": items, "total": total, "payments_ready": payments.is_configured()},
    )


def _start_payment(request: HttpRequest, order: Order, first: bool = False) -> HttpResponse:
    if first:
        messages.success(
            request,
            f"Złożono zamówienie {order.number}. Potwierdzenie wysłaliśmy na {order.email}.",
        )
    try:
        url = payments.create_checkout_session(order)
    except payments.PaymentError as error:
        messages.error(request, str(error))
        return redirect(order.get_absolute_url())
    response = HttpResponseRedirect(url)  # strona płatności Stripe Checkout
    response.status_code = 303
    return response


def _own_order(request: HttpRequest, number: str) -> Order:
    order = get_object_or_404(Order.objects.select_related("user"), number=number)
    if order.user_id != request.user.pk and not can_manage_orders(request.user):
        raise Http404
    return order


@login_required
def order_list(request: HttpRequest) -> HttpResponse:
    orders = Order.objects.filter(user=request.user).annotate(item_count=Count("items"))
    page_obj, paginator = paginate_queryset(request, orders, per_page=10)
    return render(request, "orders/order_list.html", {"orders": page_obj, "page_obj": page_obj, "paginator": paginator})


@login_required
def order_detail(request: HttpRequest, number: str) -> HttpResponse:
    order = _own_order(request, number)
    if payments.sync_order(order):
        order.refresh_from_db()
        messages.success(request, f"Stripe potwierdził płatność za {order.number}.")
    manager = can_manage_orders(request.user)
    status_form = None
    if manager:
        status_form = OrderStatusForm(initial={"status": order.status})
    return _render_detail(request, order, manager, status_form)


def _render_detail(request, order, manager, status_form) -> HttpResponse:
    items = list(order.items.select_related("course").prefetch_related("codes__redeemed_by"))
    return render(
        request,
        "orders/order_detail.html",
        {
            "order": order,
            "items": items,
            "history": order.history.select_related("changed_by"),
            "is_owner": order.user_id == request.user.pk,
            "manager": manager,
            "status_form": status_form,
            "payments_ready": payments.is_configured(),
            "test_mode": payments.is_test_mode(),
            "has_codes": any(item.codes.all() for item in items),
        },
    )


@login_required
@require_POST
def order_pay(request: HttpRequest, number: str) -> HttpResponse:
    order = get_object_or_404(Order, number=number, user=request.user)
    if not order.can_pay:
        messages.info(request, f"Zamówienie {order.number} ma status „{order.get_status_display()}” — nie wymaga płatności.")
        return redirect(order.get_absolute_url())
    return _start_payment(request, order)


@login_required
def payment_return(request: HttpRequest, number: str) -> HttpResponse:
    """Powrót ze Stripe po płatności — sprawdzamy sesję po stronie serwera (nie ufamy samemu adresowi)."""
    order = get_object_or_404(Order, number=number, user=request.user)
    if payments.confirm_from_return(order, request.GET.get("session_id", "")):
        messages.success(request, f"Dziękujemy! Płatność za {order.number} przyjęta — kursy są w „Moich kursach”.")
    else:
        order.refresh_from_db()
        if order.status in (Order.Status.PAID, Order.Status.COMPLETED):
            messages.success(request, f"Zamówienie {order.number} jest opłacone.")
        else:
            messages.info(request, "Płatność jest przetwarzana. Status zmieni się automatycznie po potwierdzeniu przez bramkę.")
    return redirect(order.get_absolute_url())


@login_required
def payment_cancel(request: HttpRequest, number: str) -> HttpResponse:
    order = get_object_or_404(Order, number=number, user=request.user)
    messages.warning(request, f"Płatność za {order.number} została przerwana. Możesz spróbować ponownie.")
    return redirect(order.get_absolute_url())


@login_required
@require_POST
def order_cancel(request: HttpRequest, number: str) -> HttpResponse:
    order = get_object_or_404(Order, number=number, user=request.user)
    try:
        cancel_by_user(order, request.user)
    except OrderError as error:
        messages.error(request, str(error))
    else:
        messages.info(request, f"Anulowano zamówienie {order.number}.")
    return redirect(order.get_absolute_url())


@csrf_exempt
@require_POST
def stripe_webhook(request: HttpRequest) -> HttpResponse:
    """Jedyny widok bez CSRF — autentyczność zapewnia podpis Stripe (STRIPE_WEBHOOK_SECRET)."""
    try:
        result = payments.handle_webhook(request.body, request.headers.get("Stripe-Signature", ""))
    except payments.PaymentError as error:
        logger.warning("Stripe webhook odrzucony: %s", error)
        return HttpResponse(str(error), status=400, content_type="text/plain; charset=utf-8")
    logger.info("Stripe webhook: %s", result)
    return HttpResponse(result, content_type="text/plain; charset=utf-8")


# ---------------------------------------------------------------------------
# Obsługa zamówień (moderator / administrator)
# ---------------------------------------------------------------------------

def _require_manager(user) -> None:
    if not can_manage_orders(user):
        raise Http404("Zamówieniami zarządzają moderatorzy i administratorzy.")


@login_required
def manage_orders(request: HttpRequest) -> HttpResponse:
    _require_manager(request.user)
    status = request.GET.get("status", "")
    query = request.GET.get("q", "").strip()
    orders = Order.objects.select_related("user").annotate(item_count=Count("items"))
    if status in Order.Status.values:
        orders = orders.filter(status=status)
    if query:
        orders = orders.filter(
            Q(number__icontains=query) | Q(email__icontains=query) | Q(user__username__icontains=query)
            | Q(last_name__icontains=query)
        )
    counts = dict(Order.objects.values_list("status").annotate(n=Count("pk")))
    page_obj, paginator = paginate_queryset(request, orders, per_page=15)
    from apps.catalog.moderation_views import _queue_counts

    return render(
        request,
        "orders/manage_orders.html",
        {
            "orders": page_obj,
            "page_obj": page_obj,
            "paginator": paginator,
            "status": status,
            "query": query,
            "statuses": [(value, label, counts.get(value, 0)) for value, label in Order.Status.choices],
            "all_count": sum(counts.values()),
            **_queue_counts(request.user),
        },
    )


@login_required
@require_POST
def order_status(request: HttpRequest, number: str) -> HttpResponse:
    _require_manager(request.user)
    order = get_object_or_404(Order.objects.select_related("user"), number=number)
    form = OrderStatusForm(request.POST)
    if form.is_valid():
        try:
            change_status(order, form.cleaned_data["status"], request.user, form.cleaned_data["note"])
        except OrderError as error:
            form.add_error("status", str(error))
        else:
            order.refresh_from_db()
            messages.success(request, f"Zamówienie {order.number}: status „{order.get_status_display()}”.")
            return redirect(order.get_absolute_url())
    messages.error(request, "Nie zmieniono statusu — popraw zaznaczone pole.")
    return _render_detail(request, order, True, form)


# ---------------------------------------------------------------------------
# Kody dostępu
# ---------------------------------------------------------------------------

@login_required
def redeem(request: HttpRequest) -> HttpResponse:
    form = RedeemForm(request.POST or None, initial={"code": request.GET.get("kod", "")})
    if request.method == "POST" and form.is_valid():
        try:
            code = redeem_code(request.user, form.cleaned_data["code"])
        except OrderError as error:
            form.add_error("code", str(error))
        else:
            messages.success(request, f"Kod przyjęty — masz dostęp do kursu „{code.course.title}”.")
            return redirect("catalog:course_detail", slug=code.course.slug)
    return render(request, "orders/redeem.html", {"form": form})
