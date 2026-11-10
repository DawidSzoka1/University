"""Sygnały aplikacji accounts.

Obsługuje zdarzenia allauth — kopiuje dane z konta społecznościowego
(avatar, imię, nazwisko, email) do CustomUser przy pierwszym logowaniu
lub aktualizacji konta społecznościowego.
"""

import logging

from django.dispatch import receiver

logger = logging.getLogger(__name__)

try:
    from allauth.socialaccount.signals import (
        social_account_added,
        social_account_updated,
    )

    @receiver(social_account_added)
    @receiver(social_account_updated)
    def copy_social_account_data(request, sociallogin, **kwargs) -> None:
        """Kopiuje dane z konta społecznościowego do profilu użytkownika.

        Uzupełnia pola tylko wtedy, gdy są puste — nie nadpisuje danych
        wprowadzonych ręcznie przez użytkownika.
        """
        user = sociallogin.user
        extra_data = sociallogin.account.extra_data

        if not user.first_name:
            user.first_name = extra_data.get('given_name', '') or extra_data.get(
                'first_name', ''
            )
            # Jeśli dane zawierają tylko pełne 'name' lub 'display_name' (np. GitLab, Bitbucket)
            full_name = extra_data.get('name', '') or extra_data.get('display_name', '')
            if not user.first_name and full_name:
                full_name = full_name.strip()
                if ' ' in full_name:
                    user.first_name, user.last_name = full_name.split(' ', 1)
                else:
                    user.first_name = full_name

        if not user.last_name:
            user.last_name = extra_data.get('family_name', '') or extra_data.get(
                'last_name', ''
            )

        if not user.email:
            user.email = extra_data.get('email', '')

        # Avatar — pobierz URL z danych konta społecznościowego
        if not user.avatar_url and not user.avatar:
            avatar_url = ""
            provider = sociallogin.account.provider
            if provider == "discord":
                discord_id = extra_data.get("id")
                avatar_hash = extra_data.get("avatar")
                if discord_id and avatar_hash:
                    avatar_url = f"https://cdn.discordapp.com/avatars/{discord_id}/{avatar_hash}.png"
            elif provider == "gitlab":
                avatar_url = extra_data.get("avatar_url", "")
            elif provider in ("bitbucket", "bitbucket_oauth2"):
                avatar_url = extra_data.get("links", {}).get("avatar", {}).get("href", "")

            if not avatar_url:
                avatar_url = (
                    extra_data.get("avatar_url", "")
                    or extra_data.get("picture", "")
                    or extra_data.get("links", {}).get("avatar", {}).get("href", "")
                )

            if avatar_url:
                user.avatar_url = avatar_url
                logger.info(
                    "Social avatar URL for user %s (%s): %s",
                    user.username,
                    provider,
                    avatar_url,
                )

        user.save()

except ImportError:
    # django-allauth nie jest zainstalowany — pomijamy rejestrację sygnałów.
    pass


from django.contrib import messages as django_messages
from django.contrib.auth.signals import user_logged_in


@receiver(user_logged_in, dispatch_uid="accounts_connect_pending_social")
def connect_pending_social_account(sender, request, user, **kwargs) -> None:
    """Po zalogowaniu (hasłem albo innym portalem) dołącza portal czekający w sesji."""
    if request is None or not hasattr(request, "session"):
        return
    from apps.accounts.adapter import connect_pending

    provider = connect_pending(request, user)
    if provider:
        django_messages.success(
            request,
            f"Połączono konto {provider}. Następnym razem wystarczy przycisk „{provider}” na stronie logowania.",
            fail_silently=True,
        )


@receiver(user_logged_in, dispatch_uid="accounts_unread_notifications_after_login")
def announce_unread_notifications(sender, request, user, **kwargs) -> None:
    """Zaraz po zalogowaniu komunikat (nie alert) o nieprzeczytanych powiadomieniach."""
    if request is None or not hasattr(request, "session"):
        return
    from apps.core.templatetags.core_tags import pl_plural

    unread = user.notifications.filter(is_read=False).count()
    if unread:
        django_messages.info(
            request,
            f"Masz {unread} {pl_plural(unread, 'nieprzeczytane powiadomienie,nieprzeczytane powiadomienia,nieprzeczytanych powiadomień')} "
            "— zobacz je pod dzwonkiem w prawym górnym rogu.",
            fail_silently=True,
        )
