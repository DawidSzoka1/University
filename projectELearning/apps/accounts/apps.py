from django.apps import AppConfig


class AccountsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.accounts'
    verbose_name = 'Użytkownicy i Role'

    def ready(self):
        import apps.accounts.signals  # noqa: F401

        # Ujednolicenie URL powrotnego dla Bitbucketa (bez przyrostka _oauth2)
        try:
            from allauth.socialaccount.providers.bitbucket_oauth2.views import (
                BitbucketOAuth2Adapter,
            )
            from allauth.utils import build_absolute_uri
            from django.urls import reverse

            def get_bitbucket_callback_url(self, request, app):
                try:
                    return build_absolute_uri(request, reverse("bitbucket_callback"))
                except Exception:
                    return build_absolute_uri(request, "/accounts/social/bitbucket/login/callback/")

            BitbucketOAuth2Adapter.get_callback_url = get_bitbucket_callback_url
        except ImportError:
            pass
