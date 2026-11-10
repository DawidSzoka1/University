from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from apps.accounts.models import CustomUser
from apps.core.context_processors import SESSION_THEME_KEY
from apps.core.models import ColorTheme

User = get_user_model()


class ThemeSelectionTests(TestCase):
    """Przełączanie motywów dla zalogowanych i anonimowych."""

    @classmethod
    def setUpTestData(cls):
        cls.light = ColorTheme.objects.create(
            name="Jasny",
            primary="#B45309",
            secondary="#0F766E",
            surface="#FFFBEB",
            is_default=True,
        )
        cls.dark = ColorTheme.objects.create(
            name="Ciemny",
            primary="#38BDF8",
            secondary="#22D3EE",
            surface="#0B1220",
        )
        cls.user = User.objects.create_user(username="tester", password="test123!")

    def test_anonymous_selection_is_kept_in_session(self):
        response = self.client.post(
            reverse("accounts:set-theme"),
            {"theme": self.dark.pk, "next": "/"},
        )
        self.assertRedirects(response, "/", fetch_redirect_response=False)
        self.assertEqual(self.client.session[SESSION_THEME_KEY], self.dark.pk)

        page = self.client.get(reverse("core:home"))
        self.assertEqual(page.context["theme_primary"], "#38BDF8")
        self.assertEqual(page.context["theme_mode"], "dark")

    def test_logged_in_selection_is_saved_on_user(self):
        self.client.force_login(self.user)
        response = self.client.post(
            reverse("accounts:set-theme"),
            {"theme": self.dark.pk, "next": "/"},
        )
        self.assertRedirects(response, "/", fetch_redirect_response=False)

        self.user.refresh_from_db()
        self.assertEqual(self.user.selected_theme, self.dark)
        # Kolejne żądania (także z innego urządzenia) widzą ten sam motyw.
        page = self.client.get(reverse("core:home"))
        self.assertEqual(page.context["theme_primary"], "#38BDF8")
        self.assertEqual(page.context["theme_source"], "user")

    def test_default_option_clears_selection(self):
        self.client.force_login(self.user)
        self.client.post(
            reverse("accounts:set-theme"),
            {"theme": self.dark.pk, "next": "/"},
        )
        self.client.post(
            reverse("accounts:set-theme"),
            {"theme": "default", "next": "/"},
        )
        self.user.refresh_from_db()
        self.assertIsNone(self.user.selected_theme)

        page = self.client.get(reverse("core:home"))
        self.assertEqual(page.context["theme_primary"], self.light.primary)

    def test_invalid_theme_does_not_change_selection(self):
        self.client.force_login(self.user)
        response = self.client.post(
            reverse("accounts:set-theme"),
            {"theme": "999999", "next": "/"},
        )
        self.assertEqual(response.status_code, 302)
        self.user.refresh_from_db()
        self.assertIsNone(self.user.selected_theme)

    def test_set_theme_requires_post(self):
        response = self.client.get(reverse("accounts:set-theme"))
        self.assertEqual(response.status_code, 405)

    def test_external_next_url_is_ignored(self):
        response = self.client.post(
            reverse("accounts:set-theme"),
            {"theme": self.dark.pk, "next": "https://evil.example.com/"},
        )
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, reverse("accounts:select-theme"))

    def test_selection_page_works_for_guests(self):
        response = self.client.get(reverse("accounts:select-theme"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, self.light.name)
        self.assertContains(response, self.dark.name)

    def test_theme_switcher_is_rendered_in_navbar(self):
        response = self.client.get(reverse("core:home"))
        self.assertContains(response, "data-theme-switcher")
        self.assertContains(response, "Wszystkie motywy")


class AuthSessionTests(TestCase):
    """Testy uwierzytelniania sesyjnego, odporności na SQLi oraz zachowania pól."""

    def setUp(self):
        self.user = User.objects.create_user(
            username="student_jan",
            email="jan@example.com",
            password="TajneHaslo123!",
            first_name="Jan",
            last_name="Kowalski",
        )

    def test_login_successful_with_username(self):
        response = self.client.post(
            reverse("accounts:login"),
            {"login": "student_jan", "password": "TajneHaslo123!"},
        )
        self.assertRedirects(response, reverse("accounts:select-theme"))
        self.assertEqual(int(self.client.session["_auth_user_id"]), self.user.pk)

    def test_login_successful_with_email(self):
        response = self.client.post(
            reverse("accounts:login"),
            {"login": "jan@example.com", "password": "TajneHaslo123!"},
        )
        self.assertRedirects(response, reverse("accounts:select-theme"))
        self.assertEqual(int(self.client.session["_auth_user_id"]), self.user.pk)

    def test_login_failure_preserves_login_and_password_input(self):
        """Wymóg Min: zapamiętywanie pól przy przeładowaniu (login i hasło zachowane w polach)."""
        response = self.client.post(
            reverse("accounts:login"),
            {"login": "jan@example.com", "password": "BledneHaslo999!"},
        )
        self.assertEqual(response.status_code, 200)
        content = response.content.decode()
        # Login i hasło są zachowane w wartościach pól formularza
        self.assertIn('value="jan@example.com"', content)
        self.assertIn('value="BledneHaslo999!"', content)
        # Komunikat błędu wyświetlany w ostylowanym alercie
        self.assertIn("auth-alert--error", content)
        self.assertContains(response, "Nieprawidłowa nazwa użytkownika / e-mail lub hasło.")

    def test_login_sql_injection_attempt_is_safe(self):
        """Wymóg Min: odporność na SQL Injection przez ORM."""
        sqli_payload = "' OR 1=1 --"
        response = self.client.post(
            reverse("accounts:login"),
            {"login": sqli_payload, "password": sqli_payload},
        )
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("_auth_user_id", self.client.session)
        self.assertContains(response, "Niepoprawne dane logowania")

    def test_registration_preserves_fields_except_password_on_error(self):
        """Wymóg Min: zachowanie pól w formularzu rejestracji przy błędzie."""
        response = self.client.post(
            reverse("accounts:register"),
            {
                "username": "nowy_student",
                "email": "nowy@example.com",
                "first_name": "Adam",
                "last_name": "Nowak",
                "password1": "DobreHaslo123!",
                "password2": "NiezgodneHaslo456!",
            },
        )
        self.assertEqual(response.status_code, 200)
        content = response.content.decode()
        self.assertIn('value="nowy_student"', content)
        self.assertIn('value="nowy@example.com"', content)
        self.assertIn('value="Adam"', content)
        self.assertIn('value="Nowak"', content)
        # Hasła nie są zwracane
        self.assertNotIn("DobreHaslo123!", content)
        self.assertNotIn("NiezgodneHaslo456!", content)

    def test_registration_success_creates_student_and_logs_in(self):
        """Rejestracja tworzy konto ze statusem student i automatycznie loguje."""
        response = self.client.post(
            reverse("accounts:register"),
            {
                "username": "nowy_uczen",
                "email": "uczen@example.com",
                "first_name": "Piotr",
                "last_name": "Zieliński",
                "password1": "Praktyka2026!",
                "password2": "Praktyka2026!",
            },
        )
        self.assertRedirects(response, reverse("core:home"))
        created = User.objects.get(username="nowy_uczen")
        self.assertEqual(created.role, "student")
        self.assertEqual(created.email, "uczen@example.com")
        self.assertEqual(int(self.client.session["_auth_user_id"]), created.pk)

    def test_logout_clears_session(self):
        self.client.force_login(self.user)
        response = self.client.get(reverse("accounts:logout"))
        self.assertRedirects(response, reverse("core:home"))
        self.assertNotIn("_auth_user_id", self.client.session)


class TokenAuthAPITests(TestCase):
    """Testy uwierzytelniania tokenowego JWT (Wymóg 4.0)."""

    def setUp(self):
        self.user = User.objects.create_user(
            username="api_user",
            password="SilneHaslo123!",
            email="api@example.com",
        )

    def test_obtain_jwt_token_pair(self):
        response = self.client.post(
            reverse("accounts:token-obtain-pair"),
            {"username": "api_user", "password": "SilneHaslo123!"},
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("access", data)
        self.assertIn("refresh", data)
        return data["access"]

    def test_api_me_with_jwt_token(self):
        token_resp = self.client.post(
            reverse("accounts:token-obtain-pair"),
            {"username": "api_user", "password": "SilneHaslo123!"},
            content_type="application/json",
        )
        access_token = token_resp.json()["access"]

        response = self.client.get(
            reverse("accounts:api-me"),
            HTTP_AUTHORIZATION=f"Bearer {access_token}",
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["username"], "api_user")
        self.assertEqual(data["email"], "api@example.com")

    def test_api_me_without_token_is_unauthorized(self):
        response = self.client.get(reverse("accounts:api-me"))
        self.assertEqual(response.status_code, 401)


class SocialAuthTests(TestCase):
    """Testy logowania społecznościowego (Discord, GitLab, Bitbucket - Wymóg 4.5)."""

    def test_social_login_buttons_rendered_on_login_and_register_pages(self):
        for url_name in ("accounts:login", "accounts:register"):
            with self.subTest(url=url_name):
                response = self.client.get(reverse(url_name))
                self.assertEqual(response.status_code, 200)
                content = response.content.decode()
                # Co najmniej 2 portale inne niż Google i GitHub; Facebook wyłączony
                self.assertNotIn("/accounts/social/facebook/", content)
                self.assertIn("/accounts/social/discord/login/", content)
                self.assertIn("/accounts/social/gitlab/login/", content)
                self.assertIn("/accounts/social/bitbucket/login/", content)
                # Formularze POST zamiast samych linków GET dla bezpośredniego przekierowania
                self.assertIn('method="post"', content)
                self.assertIn('class="social-btn-form"', content)
                # Split-screen layout obecny
                self.assertIn("auth-split", content)
                self.assertIn("auth-split__info", content)

    def test_social_login_post_redirects_to_external_provider(self):
        """Wysłanie POST na endpoint logowania dostawcy natychmiast przekierowuje do OAuth."""
        for provider, expected_host in (
            ("discord", "discord.com"),
            ("gitlab", "gitlab.com"),
            ("bitbucket_oauth2", "bitbucket.org"),
            ("bitbucket", "bitbucket.org"),
        ):
            with self.subTest(provider=provider):
                response = self.client.post(
                    f"/accounts/social/{provider}/login/?process=login",
                    HTTP_HOST="localhost",
                )
                self.assertEqual(response.status_code, 302)
                location = response.headers.get("Location")
                self.assertIn(expected_host, location)
                if provider == "bitbucket":
                    from urllib.parse import unquote
                    unquoted = unquote(location)
                    # Wymóg: czysty URL bitbucket bez _oauth2
                    self.assertIn("/accounts/social/bitbucket/login/callback/", unquoted)
                    self.assertNotIn("_oauth2", unquoted)

    def test_social_intermediate_login_page_renders_with_our_design_system(self):
        """Strona pośrednia (jeśli wejdzie się GET) posiada nasze szablony i tokeny CSS."""
        response = self.client.get(
            "/accounts/social/discord/login/?process=login",
            HTTP_HOST="localhost",
        )
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "socialaccount/login.html")
        self.assertTemplateUsed(response, "base.html")
        content = response.content.decode()
        self.assertIn("auth-card", content)
        self.assertIn("Zaloguj się za pomocą Discord", content)

    def test_profile_displays_social_connection_options(self):
        """W profilu użytkownik widzi listę portali z możliwością połączenia (process=connect)."""
        user = CustomUser.objects.create_user(
            username="profile_student",
            email="profile@example.com",
            password="TajneHaslo123!",
        )
        self.client.force_login(user)
        response = self.client.get(reverse("accounts:profile"))
        self.assertEqual(response.status_code, 200)
        content = response.content.decode()
        self.assertIn("Połączone konta społecznościowe", content)
        for provider in ("Discord", "GitLab", "Bitbucket"):
            self.assertIn(provider, content)
        self.assertNotIn("Facebook", content)
        # Przycisk łączenia zawiera process=connect
        self.assertIn("process=connect", content)

    def test_disconnect_social_account_success(self):
        """Użytkownik z hasłem może bezpiecznie odłączyć konto społecznościowe."""
        from allauth.socialaccount.models import SocialAccount
        user = CustomUser.objects.create_user(
            username="disconnect_student",
            email="disc@example.com",
            password="TajneHaslo123!",
        )
        account = SocialAccount.objects.create(
            user=user,
            provider="discord",
            uid="123456789",
            extra_data={"username": "disc_user"},
        )
        self.client.force_login(user)
        response = self.client.post(
            reverse("accounts:disconnect-social-account", kwargs={"account_id": account.pk})
        )
        self.assertRedirects(response, reverse("accounts:profile"))
        self.assertFalse(SocialAccount.objects.filter(pk=account.pk).exists())

    def test_disconnect_social_account_blocked_if_no_password_and_last_account(self):
        """Nie można odłączyć jedynego konta społecznościowego, jeśli użytkownik nie ma hasła."""
        from allauth.socialaccount.models import SocialAccount
        user = CustomUser.objects.create_user(
            username="oauth_only_user",
            email="oauth@example.com",
        )
        user.set_unusable_password()
        user.save()
        account = SocialAccount.objects.create(
            user=user,
            provider="gitlab",
            uid="987654",
            extra_data={"username": "gitlab_user"},
        )
        self.client.force_login(user)
        response = self.client.post(
            reverse("accounts:disconnect-social-account", kwargs={"account_id": account.pk})
        )
        self.assertRedirects(response, reverse("accounts:profile"))
        # Konto nie zostało usunięte
        self.assertTrue(SocialAccount.objects.filter(pk=account.pk).exists())


class RankThresholdTests(TestCase):
    """Progi rang zgodne ze specyfikacją: Nowicjusz, Pomocnik, Ekspert, Mistrz."""

    def test_rank_for_points(self):
        expected = {-5: "Obserwator", 0: "Nowicjusz", 10: "Nowicjusz", 11: "Pomocnik", 50: "Pomocnik",
                    51: "Ekspert", 200: "Ekspert", 201: "Mistrz"}
        for points, rank in expected.items():
            with self.subTest(points=points):
                self.assertEqual(CustomUser.rank_for_points(points), rank)

    def test_new_user_starts_as_novice(self):
        user = CustomUser.objects.create_user(username="nowy_uczen", password="Haslo12345!")
        self.assertEqual(user.rank_title, CustomUser.Rank.NOVICE)


class SocialConnectExistingAccountTests(TestCase):
    """Portal zwrócił e-mail istniejącego konta → logowanie zamiast formularza z błędem, potem połączenie."""

    password = "TajneHaslo123!"

    def setUp(self):
        self.owner = CustomUser.objects.create_user(
            username="oficer", email="oficer@example.com", password=self.password
        )

    def make_sociallogin(self, email="oficer@example.com", uid="777", username="oficer"):
        from allauth.account.models import EmailAddress
        from allauth.socialaccount.models import SocialAccount, SocialLogin

        from allauth.socialaccount.adapter import get_adapter
        from django.test import RequestFactory

        provider = get_adapter().get_provider(RequestFactory().get("/"), "gitlab")
        user = CustomUser(username=username, email=email)
        account = SocialAccount(provider="gitlab", uid=uid, extra_data={"email": email, "username": username})
        return SocialLogin(
            user=user, account=account, provider=provider,
            email_addresses=[EmailAddress(email=email, verified=False, primary=True)],
        )

    def request_with_session(self):
        from django.contrib.auth.models import AnonymousUser
        from django.contrib.messages.middleware import MessageMiddleware
        from django.contrib.sessions.middleware import SessionMiddleware
        from django.test import RequestFactory

        request = RequestFactory().get("/accounts/social/gitlab/login/callback/")
        SessionMiddleware(lambda r: None).process_request(request)
        MessageMiddleware(lambda r: None).process_request(request)
        request.user = AnonymousUser()
        return request

    def store_pending(self, email="oficer@example.com", uid="777"):
        import time

        from apps.accounts.adapter import PENDING_SOCIAL_KEY

        session = self.client.session
        session[PENDING_SOCIAL_KEY] = {
            "sociallogin": self.make_sociallogin(email=email, uid=uid).serialize(),
            "email": email,
            "provider": "GitLab",
            "created": time.time(),
        }
        session.save()

    def test_taken_email_redirects_to_login_instead_of_signup_form(self):
        from allauth.core.exceptions import ImmediateHttpResponse

        from apps.accounts.adapter import PENDING_SOCIAL_KEY, SocialAccountAdapter

        request = self.request_with_session()
        with self.assertRaises(ImmediateHttpResponse) as ctx:
            SocialAccountAdapter(request).pre_social_login(request, self.make_sociallogin())
        self.assertEqual(ctx.exception.response.url, reverse("accounts:login") + "?polacz=1")
        self.assertEqual(request.session[PENDING_SOCIAL_KEY]["email"], "oficer@example.com")

    def test_new_email_goes_through_normal_signup(self):
        from apps.accounts.adapter import PENDING_SOCIAL_KEY, SocialAccountAdapter

        request = self.request_with_session()
        SocialAccountAdapter(request).pre_social_login(request, self.make_sociallogin(email="nowy@example.com"))
        self.assertNotIn(PENDING_SOCIAL_KEY, request.session)

    def test_login_page_explains_and_prefills_email(self):
        self.store_pending()
        response = self.client.get(reverse("accounts:login") + "?polacz=1")
        self.assertContains(response, "Łączenie z GitLab")
        self.assertContains(response, 'value="oficer@example.com"')
        self.assertContains(response, reverse("accounts:social_connect_cancel"))

    def test_password_login_connects_portal_and_keeps_password(self):
        from allauth.socialaccount.models import SocialAccount

        self.store_pending()
        response = self.client.post(reverse("accounts:login"), {"login": "oficer@example.com", "password": self.password})
        self.assertEqual(response.status_code, 302)
        self.assertTrue(SocialAccount.objects.filter(user=self.owner, provider="gitlab", uid="777").exists())
        self.owner.refresh_from_db()
        self.assertTrue(self.owner.check_password(self.password))

    def test_other_account_login_does_not_get_the_portal(self):
        from allauth.socialaccount.models import SocialAccount

        CustomUser.objects.create_user(username="ktos", email="ktos@example.com", password=self.password)
        self.store_pending()
        self.client.post(reverse("accounts:login"), {"login": "ktos", "password": self.password})
        self.assertFalse(SocialAccount.objects.exists())

    def test_cancel_drops_pending_connection(self):
        from apps.accounts.adapter import PENDING_SOCIAL_KEY

        self.store_pending()
        self.assertEqual(self.client.get(reverse("accounts:social_connect_cancel")).status_code, 405)
        self.client.post(reverse("accounts:social_connect_cancel"))
        self.assertNotIn(PENDING_SOCIAL_KEY, self.client.session)

    def test_taken_portal_username_gets_unique_alternative(self):
        from apps.accounts.adapter import SocialAccountAdapter

        request = self.request_with_session()
        sociallogin = self.make_sociallogin(email="inny@example.com")
        user = SocialAccountAdapter(request).populate_user(request, sociallogin, {"username": "oficer", "email": "inny@example.com"})
        self.assertNotEqual(user.username.lower(), "oficer")
        self.assertTrue(user.username)


class AvailabilityCheckTests(TestCase):
    def setUp(self):
        CustomUser.objects.create_user(username="zajety", email="zajety@example.com", password="Haslo12345!")
        self.url = reverse("accounts:check_availability")

    def test_reports_taken_and_free_values(self):
        taken = self.client.get(self.url, {"field": "email", "value": "ZAJETY@example.com"}).json()
        self.assertFalse(taken["available"])
        self.assertIn("już istnieje", taken["message"])
        self.assertTrue(self.client.get(self.url, {"field": "username", "value": "wolny"}).json()["available"])
        self.assertFalse(self.client.get(self.url, {"field": "username", "value": "Zajety"}).json()["available"])

    def test_rejects_other_fields(self):
        self.assertEqual(self.client.get(self.url, {"field": "password", "value": "x"}).status_code, 400)

    def test_register_form_has_live_check(self):
        response = self.client.get(reverse("accounts:register"))
        self.assertContains(response, "data-availability-form")
        self.assertContains(response, "js/availability.js")
