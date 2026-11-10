"""Koszyk, zamówienia, płatności (Stripe) i kody dostępu do kursów."""

from __future__ import annotations

import secrets
from decimal import Decimal

from django.conf import settings
from django.db import models
from django.db.models import Q
from django.urls import reverse
from django.utils import timezone

from apps.core.money import format_pln

MAX_QUANTITY = 10  # sztuk jednego kursu w koszyku (1 dla siebie + kody dla innych)


class Cart(models.Model):
    """Koszyk zalogowanego (``user``) albo gościa (id w sesji); po zalogowaniu scalany."""

    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name="cart",
        verbose_name="Użytkownik",
    )
    created_at = models.DateTimeField("Utworzono", auto_now_add=True)
    updated_at = models.DateTimeField("Zaktualizowano", auto_now=True)

    class Meta:
        verbose_name = "Koszyk"
        verbose_name_plural = "Koszyki"

    def __str__(self) -> str:
        return f"Koszyk {self.user or 'gościa'} ({self.pk})"

    @property
    def total(self) -> Decimal:
        return sum((item.line_total for item in self.items.all()), Decimal("0.00"))

    @property
    def quantity(self) -> int:
        return sum(item.quantity for item in self.items.all())


class CartItem(models.Model):
    """Pozycja koszyka. Ponowne dodanie tego samego kursu zwiększa ``quantity`` (kumulacja)."""

    cart = models.ForeignKey(Cart, on_delete=models.CASCADE, related_name="items", verbose_name="Koszyk")
    course = models.ForeignKey(
        "catalog.Course", on_delete=models.CASCADE, related_name="cart_items", verbose_name="Kurs"
    )
    quantity = models.PositiveIntegerField("Ilość", default=1)
    added_at = models.DateTimeField("Dodano", auto_now_add=True)

    class Meta:
        ordering = ["added_at", "pk"]
        verbose_name = "Pozycja koszyka"
        verbose_name_plural = "Pozycje koszyka"
        constraints = [models.UniqueConstraint(fields=["cart", "course"], name="unique_cart_course")]

    def __str__(self) -> str:
        return f"{self.course} ×{self.quantity}"

    @property
    def unit_price(self) -> Decimal:
        return self.course.price

    @property
    def line_total(self) -> Decimal:
        return self.course.price * self.quantity


class Order(models.Model):
    """Zamówienie z kopią adresu i cen z chwili złożenia."""

    class Status(models.TextChoices):
        NEW = "new", "Nowe"
        PENDING_PAYMENT = "pending_payment", "Oczekuje na płatność"
        PAID = "paid", "Opłacone"
        COMPLETED = "completed", "Zrealizowane"
        CANCELLED = "cancelled", "Anulowane"

    class Provider(models.TextChoices):
        STRIPE = "stripe", "Stripe (sandbox)"
        MANUAL = "manual", "Ręcznie (moderator/administrator)"

    number = models.CharField("Numer", max_length=32, unique=True)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.PROTECT,
        related_name="orders",
        verbose_name="Użytkownik",
    )
    status = models.CharField("Status", max_length=20, choices=Status.choices, default=Status.NEW)
    total = models.DecimalField("Wartość", max_digits=10, decimal_places=2, default=0)
    # Adres wysyłki faktury / danych — kopia z chwili złożenia zamówienia
    email = models.EmailField("E-mail", default="")
    first_name = models.CharField("Imię", max_length=150, default="")
    last_name = models.CharField("Nazwisko", max_length=150, default="")
    street_address = models.CharField("Ulica i numer", max_length=255, default="")
    postal_code = models.CharField("Kod pocztowy", max_length=20, default="")
    city = models.CharField("Miasto", max_length=100, default="")
    phone_number = models.CharField("Telefon", max_length=20, blank=True, default="")
    save_address = models.BooleanField("Zapisano adres w profilu", default=False)
    # Płatność
    payment_provider = models.CharField(
        "Bramka płatności", max_length=20, choices=Provider.choices, blank=True, default=""
    )
    payment_reference = models.CharField(
        "Identyfikator sesji płatności", max_length=255, blank=True, default="", db_index=True
    )
    payment_intent = models.CharField("Identyfikator płatności", max_length=255, blank=True, default="")
    created_at = models.DateTimeField("Utworzono", auto_now_add=True)
    updated_at = models.DateTimeField("Zaktualizowano", auto_now=True)
    paid_at = models.DateTimeField("Opłacono", null=True, blank=True)
    fulfilled_at = models.DateTimeField(
        "Wydano dostęp", null=True, blank=True, help_text="Zapis na kursy i kody dostępu — tylko raz."
    )

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Zamówienie"
        verbose_name_plural = "Zamówienia"

    def __str__(self) -> str:
        return self.number

    def get_absolute_url(self) -> str:
        return reverse("orders:order_detail", kwargs={"number": self.number})

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}".strip()

    @property
    def can_pay(self) -> bool:
        return self.status in (self.Status.NEW, self.Status.PENDING_PAYMENT)

    @property
    def can_cancel(self) -> bool:
        return self.status in (self.Status.NEW, self.Status.PENDING_PAYMENT)

    @property
    def total_display(self) -> str:
        return format_pln(self.total)

    @classmethod
    def next_number(cls) -> str:
        year = timezone.localdate().year
        prefix = f"ORD-{year}-"
        last = cls.objects.filter(number__startswith=prefix).order_by("-number").values_list("number", flat=True).first()
        sequence = int(last.rsplit("-", 1)[1]) + 1 if last else 1
        return f"{prefix}{sequence:06d}"


class OrderItem(models.Model):
    """Pozycja zamówienia z ceną z chwili zakupu."""

    order = models.ForeignKey(Order, on_delete=models.CASCADE, related_name="items", verbose_name="Zamówienie")
    course = models.ForeignKey(
        "catalog.Course", on_delete=models.PROTECT, related_name="order_items", verbose_name="Kurs"
    )
    title = models.CharField("Tytuł kursu", max_length=200, default="")
    quantity = models.PositiveIntegerField("Ilość", default=1)
    unit_price = models.DecimalField("Cena jednostkowa", max_digits=10, decimal_places=2, default=0)
    line_total = models.DecimalField("Wartość pozycji", max_digits=10, decimal_places=2, default=0)

    class Meta:
        ordering = ["pk"]
        verbose_name = "Pozycja zamówienia"
        verbose_name_plural = "Pozycje zamówienia"
        constraints = [models.UniqueConstraint(fields=["order", "course"], name="unique_order_course")]

    def __str__(self) -> str:
        return f"{self.title or self.course} ×{self.quantity}"


class OrderStatusChange(models.Model):
    """Historia statusów zamówienia (kto, kiedy, dlaczego) — widoczna dla ucznia i obsługi."""

    order = models.ForeignKey(Order, on_delete=models.CASCADE, related_name="history", verbose_name="Zamówienie")
    status = models.CharField("Status", max_length=20, choices=Order.Status.choices)
    note = models.CharField("Opis", max_length=255, blank=True, default="")
    changed_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="order_status_changes",
        verbose_name="Zmienił",
    )
    created_at = models.DateTimeField("Data", auto_now_add=True)

    class Meta:
        ordering = ["created_at", "pk"]
        verbose_name = "Zmiana statusu zamówienia"
        verbose_name_plural = "Historia statusów zamówień"

    def __str__(self) -> str:
        return f"{self.order} → {self.get_status_display()}"


def generate_code() -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # bez 0/O i 1/I
    return "KURS-" + "-".join("".join(secrets.choice(alphabet) for _ in range(4)) for _ in range(2))


class AccessCode(models.Model):
    """Kod dostępu do kursu — każda kupiona sztuka ponad pierwszą dla kupującego."""

    order_item = models.ForeignKey(
        OrderItem, on_delete=models.CASCADE, related_name="codes", verbose_name="Pozycja zamówienia"
    )
    code = models.CharField("Kod", max_length=20, unique=True, default=generate_code)
    redeemed_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="redeemed_codes",
        verbose_name="Wykorzystał",
    )
    redeemed_at = models.DateTimeField("Wykorzystano", null=True, blank=True)
    created_at = models.DateTimeField("Utworzono", auto_now_add=True)

    class Meta:
        ordering = ["pk"]
        verbose_name = "Kod dostępu"
        verbose_name_plural = "Kody dostępu"
        constraints = [
            models.CheckConstraint(
                condition=Q(redeemed_by__isnull=True) | Q(redeemed_at__isnull=False),
                name="access_code_redeemed_has_date",
            )
        ]

    def __str__(self) -> str:
        return self.code

    @property
    def course(self):
        return self.order_item.course
