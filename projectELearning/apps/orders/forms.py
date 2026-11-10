"""Formularze koszyka i zamówień."""

from __future__ import annotations

import re

from django import forms

from apps.core.forms import AccessibleErrorsMixin
from apps.orders.models import MAX_QUANTITY, Order

POSTAL_CODE = re.compile(r"^\d{2}-\d{3}$")
PHONE = re.compile(r"^\+?[\d \-()]{9,20}$")


def _text(autocomplete: str, **attrs) -> forms.TextInput:
    return forms.TextInput(attrs={"class": "form-control", "autocomplete": autocomplete, **attrs})


class CheckoutForm(AccessibleErrorsMixin, forms.Form):
    """Dane do zamówienia — podstawione z profilu, można je zmienić i zapisać w profilu."""

    first_name = forms.CharField(label="Imię", max_length=150, widget=_text("given-name"))
    last_name = forms.CharField(label="Nazwisko", max_length=150, widget=_text("family-name"))
    email = forms.EmailField(
        label="E-mail do potwierdzenia",
        help_text="Na ten adres wyślemy potwierdzenie zamówienia i informację o płatności.",
        widget=forms.EmailInput(attrs={"class": "form-control", "autocomplete": "email"}),
    )
    street_address = forms.CharField(label="Ulica i numer", max_length=255, widget=_text("street-address"))
    postal_code = forms.CharField(
        label="Kod pocztowy", max_length=6, help_text="Format 00-000.",
        widget=_text("postal-code", inputmode="numeric"),
    )
    city = forms.CharField(label="Miasto", max_length=100, widget=_text("address-level2"))
    phone_number = forms.CharField(
        label="Telefon (opcjonalnie)", max_length=20, required=False, widget=_text("tel", inputmode="tel"),
    )
    save_address = forms.BooleanField(
        label="Zapisz te dane w moim profilu",
        required=False,
        help_text="Przy następnym zamówieniu podstawimy je automatycznie.",
        widget=forms.CheckboxInput(attrs={"class": "form-check"}),
    )
    accept_terms = forms.BooleanField(
        label="Akceptuję regulamin sklepu i rozumiem, że płatność jest testowa (sandbox)",
        error_messages={"required": "Zaznacz akceptację regulaminu, aby złożyć zamówienie."},
        widget=forms.CheckboxInput(attrs={"class": "form-check"}),
    )

    def clean_postal_code(self) -> str:
        value = self.cleaned_data["postal_code"].strip()
        if not POSTAL_CODE.match(value):
            raise forms.ValidationError("Wpisz kod pocztowy w formacie 00-000, np. 31-234.")
        return value

    def clean_phone_number(self) -> str:
        value = self.cleaned_data["phone_number"].strip()
        if value and not PHONE.match(value):
            raise forms.ValidationError("Wpisz numer telefonu cyframi, np. +48 600 100 200.")
        return value

    @classmethod
    def initial_from(cls, user) -> dict:
        return {
            "first_name": user.first_name,
            "last_name": user.last_name,
            "email": user.email,
            "street_address": user.street_address,
            "postal_code": user.postal_code,
            "city": user.city,
            "phone_number": user.phone_number,
        }


class QuantityForm(forms.Form):
    quantity = forms.IntegerField(label="Ilość", min_value=0, max_value=MAX_QUANTITY)


class OrderStatusForm(AccessibleErrorsMixin, forms.Form):
    status = forms.ChoiceField(
        label="Nowy status",
        choices=Order.Status.choices,
        widget=forms.Select(attrs={"class": "form-control", "data-select-enhance": ""}),
    )
    note = forms.CharField(
        label="Notatka (widoczna dla kupującego)",
        max_length=255,
        required=False,
        widget=forms.TextInput(attrs={"class": "form-control"}),
    )


class RedeemForm(AccessibleErrorsMixin, forms.Form):
    code = forms.CharField(
        label="Kod dostępu",
        max_length=20,
        help_text="Kod ma postać KURS-XXXX-XXXX. Wielkość liter nie ma znaczenia.",
        widget=forms.TextInput(attrs={"class": "form-control", "autocomplete": "off", "spellcheck": "false"}),
        error_messages={"required": "Wpisz kod dostępu."},
    )
