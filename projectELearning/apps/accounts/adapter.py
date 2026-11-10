"""Logowanie przez portale (allauth): łączenie z istniejącym kontem bez formularza z błędem.

Domyślnie allauth przy adresie e-mail zajętym przez konto lokalne pokazuje formularz
„Dokończ rejestrację” z błędem „konto już istnieje”, a jego automatyczne logowanie po
e-mailu czyści hasło niezweryfikowanego konta. Zamiast tego:

1. Portal zwrócił e-mail istniejącego konta → logowanie portalu czeka w sesji, a użytkownik
   trafia na stronę logowania z wyjaśnieniem i wpisanym adresem.
2. Po zalogowaniu na to konto (hasłem albo innym połączonym portalem) sygnał
   ``user_logged_in`` dołącza oczekujący portal — właściciel potwierdził oba konta.
"""

from __future__ import annotations

import time

from allauth.core.exceptions import ImmediateHttpResponse
from allauth.socialaccount.adapter import DefaultSocialAccountAdapter
from allauth.socialaccount.models import SocialAccount, SocialLogin
from django.contrib import messages
from django.contrib.auth import get_user_model
from django.shortcuts import redirect
from django.urls import reverse

PENDING_SOCIAL_KEY = "pending_social_connect"
PENDING_TTL_SECONDS = 15 * 60


def social_email(sociallogin: SocialLogin) -> str:
    """Adres z portalu: najpierw zweryfikowany, potem dowolny, na końcu pole użytkownika."""
    addresses = sorted(sociallogin.email_addresses, key=lambda a: not a.verified)
    for address in addresses:
        if address.email:
            return address.email.strip()
    return (sociallogin.user.email or "").strip()


def provider_name(sociallogin: SocialLogin) -> str:
    try:
        return sociallogin.account.get_provider().name
    except Exception:  # noqa: BLE001 - nazwa tylko do komunikatu
        return sociallogin.account.provider


def get_pending(request) -> dict | None:
    """Oczekujące łączenie z sesji albo None (wygasa po 15 minutach)."""
    pending = request.session.get(PENDING_SOCIAL_KEY)
    if not pending:
        return None
    if time.time() - pending.get("created", 0) > PENDING_TTL_SECONDS:
        request.session.pop(PENDING_SOCIAL_KEY, None)
        return None
    return pending


def connect_pending(request, user) -> str | None:
    """Dołącza oczekujący portal do właśnie zalogowanego konta. Zwraca nazwę portalu albo None."""
    pending = get_pending(request)
    request.session.pop(PENDING_SOCIAL_KEY, None)
    if not pending:
        return None
    # Tylko konto, którego dotyczyło łączenie (ten sam e-mail) — nie dowolne konto na tym urządzeniu
    if not user.email or user.email.lower() != pending["email"].lower():
        return None
    sociallogin = SocialLogin.deserialize(pending["sociallogin"])
    if SocialAccount.objects.filter(provider=sociallogin.account.provider, uid=sociallogin.account.uid).exists():
        return None
    sociallogin.connect(request, user)
    return pending["provider"]


class SocialAccountAdapter(DefaultSocialAccountAdapter):
    def pre_social_login(self, request, sociallogin: SocialLogin) -> None:
        if sociallogin.is_existing or request.user.is_authenticated:
            return  # konto portalu już połączone albo łączenie z profilu
        email = social_email(sociallogin)
        if not email:
            return
        owner = get_user_model().objects.filter(email__iexact=email).first()
        if owner is None:
            return
        name = provider_name(sociallogin)
        request.session[PENDING_SOCIAL_KEY] = {
            "sociallogin": sociallogin.serialize(),
            "email": email,
            "provider": name,
            "created": time.time(),
        }
        messages.info(
            request,
            f"Konto z adresem {email} już istnieje. Zaloguj się na nie — połączymy je z {name}.",
        )
        raise ImmediateHttpResponse(redirect(reverse("accounts:login") + "?polacz=1"))

    def populate_user(self, request, sociallogin, data):
        """Nazwa użytkownika z portalu bywa zajęta — wtedy dobieramy wolną zamiast pokazywać formularz."""
        user = super().populate_user(request, sociallogin, data)
        username = (user.username or "").strip()
        if username and get_user_model().objects.filter(username__iexact=username).exists():
            from allauth.account.adapter import get_adapter as get_account_adapter

            user.username = get_account_adapter(request).generate_unique_username(
                [username, user.first_name, user.last_name, user.email, "uzytkownik"]
            )
        return user
