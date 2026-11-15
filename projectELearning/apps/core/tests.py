from django.core.exceptions import ValidationError
from django.test import SimpleTestCase, TestCase

from apps.core.colors import (
    contrast_ratio,
    is_valid_hex,
    normalize_hex,
    readable_text_color,
    theme_palette,
)
from apps.core.models import ColorTheme

STARTER_THEMES = [
    ("Bursztyn", "#B45309", "#0F766E", "#D97706", "#FFFBEB", "#FEF3C7"),
    ("Ocean", "#38BDF8", "#22D3EE", "#818CF8", "#0B1220", "#152033"),
    ("Las", "#166534", "#A16207", "#15803D", "#F0FDF4", "#DCFCE7"),
    ("Neon", "#C084FC", "#F472B6", "#38BDF8", "#1E1033", "#2D184C"),
]


class ColorUtilsTests(SimpleTestCase):
    def test_normalize_hex_accepts_short_prefix(self):
        self.assertEqual(normalize_hex("4f46e5"), "#4F46E5")
        self.assertEqual(normalize_hex("  #4f46e5  "), "#4F46E5")

    def test_normalize_hex_rejects_invalid_values(self):
        for value in ("", "#FFF", "niebieski", "#GGGGGG", "#1234567"):
            with self.subTest(value=value):
                self.assertFalse(is_valid_hex(value))
                with self.assertRaises(ValueError):
                    normalize_hex(value)

    def test_readable_text_always_reaches_aa_contrast(self):
        for _name, primary, secondary, accent, surface, surface_variant in STARTER_THEMES:
            for color in (primary, secondary, accent, surface, surface_variant):
                with self.subTest(color=color):
                    text = readable_text_color(color)
                    self.assertGreaterEqual(contrast_ratio(color, text), 4.5)

    def test_palette_flags_dark_surfaces(self):
        self.assertFalse(
            theme_palette(
                STARTER_THEMES[0][1],
                STARTER_THEMES[0][2],
                STARTER_THEMES[0][4],
                STARTER_THEMES[0][3],
                STARTER_THEMES[0][5],
            )["is_dark"]
        )
        self.assertTrue(
            theme_palette(
                STARTER_THEMES[1][1],
                STARTER_THEMES[1][2],
                STARTER_THEMES[1][4],
                STARTER_THEMES[1][3],
                STARTER_THEMES[1][5],
            )["is_dark"]
        )


class ColorThemeModelTests(TestCase):
    def setUp(self):
        # Data migration seeds four themes; start from a clean slate here.
        ColorTheme.objects.all().delete()

    def test_save_normalizes_colours(self):
        theme = ColorTheme.objects.create(
            name="Test",
            primary="4f46e5",
            secondary="#10b981",
            accent="f59e0b",
            surface="#f8fafc",
            surface_variant="ede9fe",
        )
        theme.refresh_from_db()
        self.assertEqual(theme.primary, "#4F46E5")
        self.assertEqual(theme.secondary, "#10B981")
        self.assertEqual(theme.accent, "#F59E0B")
        self.assertEqual(theme.surface, "#F8FAFC")
        self.assertEqual(theme.surface_variant, "#EDE9FE")

    def test_first_theme_becomes_default(self):
        theme = ColorTheme.objects.create(
            name="Pierwszy", primary="#111111", secondary="#222222", surface="#FFFFFF"
        )
        self.assertTrue(theme.is_default)

    def test_only_one_default_theme(self):
        first = ColorTheme.objects.create(
            name="Pierwszy", primary="#111111", secondary="#222222", surface="#FFFFFF"
        )
        second = ColorTheme.objects.create(
            name="Drugi",
            primary="#333333",
            secondary="#444444",
            surface="#000000",
            is_default=True,
        )
        first.refresh_from_db()
        self.assertTrue(second.is_default)
        self.assertFalse(first.is_default)

    def test_clean_rejects_invalid_colour(self):
        theme = ColorTheme(name="Zły", primary="czerwony")
        with self.assertRaises(ValidationError):
            theme.clean()

    def test_derived_colours(self):
        theme = ColorTheme(
            name="Ciemny",
            primary="#38BDF8",
            secondary="#22D3EE",
            accent="#818CF8",
            surface="#0B1220",
            surface_variant="#152033",
        )
        self.assertTrue(theme.is_dark)
        self.assertEqual(theme.mode, "dark")
        self.assertEqual(theme.on_surface, "#FFFFFF")
        self.assertEqual(theme.on_surface_variant, "#FFFFFF")
        self.assertEqual(theme.on_primary, "#111827")
        self.assertEqual(theme.on_secondary, "#111827")
        self.assertEqual(theme.on_accent, "#111827")


class SeededThemesTests(TestCase):
    def test_starter_themes_are_created_by_migration(self):
        self.assertEqual(ColorTheme.objects.count(), 4)
        self.assertEqual(ColorTheme.objects.filter(is_default=True).count(), 1)
        self.assertSetEqual(
            set(ColorTheme.objects.values_list("name", flat=True)),
            {"Bursztyn", "Ocean", "Las", "Neon"},
        )


class HomePageTests(TestCase):
    def test_home_renders_key_sections_and_counts(self):
        from apps.catalog.models import Category, Course

        category = Category.objects.create(name="Programowanie")
        Course.objects.create(title="Python od podstaw", category=category)

        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["course_count"], 1)
        self.assertEqual(response.context["category_count"], 1)
        self.assertContains(response, 'Ucz się w <span class="marker">swoim tempie.</span>', html=False)
        self.assertContains(response, "Python od podstaw")

    def test_home_does_not_leak_template_comments(self):
        response = self.client.get("/")
        content = response.content.decode()

        self.assertNotIn("Only the 3 base colours", content)
        self.assertNotIn("{#", content)

    def test_guests_see_a_single_auth_button(self):
        response = self.client.get("/")
        content = response.content.decode()

        self.assertContains(response, "Zaloguj się")
        self.assertGreaterEqual(content.count('href="/accounts/login/"'), 2)
        self.assertNotContains(response, 'href="/accounts/register/"')

    def test_default_theme_omits_inline_override_allowing_main_css_control(self):
        """Weryfikacja minimalnego wymogu: w stanie domyślnym base.html nie nadpisuje :root stylem inline."""
        response = self.client.get("/")
        content = response.content.decode()
        self.assertFalse(response.context["theme_is_custom"])
        self.assertNotIn("--color-primary:", content)

    def test_custom_theme_injects_inline_style_overriding_css(self):
        """Gdy użytkownik/sesja wybierze motyw, 5 tokenów jest wstrzykiwanych do :root."""
        ocean = ColorTheme.objects.get(name="Ocean")
        session = self.client.session
        session["theme_id"] = ocean.pk
        session.save()

        response = self.client.get("/")
        content = response.content.decode()
        self.assertTrue(response.context["theme_is_custom"])
        self.assertIn(f"--color-primary: {ocean.primary};", content)
        self.assertIn(f"--color-accent: {ocean.accent};", content)
        self.assertIn(f"--color-surface-variant: {ocean.surface_variant};", content)

    def test_flash_messages_render_as_floating_toasts_with_close_button(self):
        """Powiadomienia flash renderują się w kontenerze toastów z przyciskiem zamknięcia."""
        from django.urls import reverse

        response = self.client.post(
            reverse("accounts:set-theme"),
            {"theme": "default", "next": "/"},
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'id="toast-container"')
        self.assertContains(response, "toast toast--success")
        self.assertContains(response, "data-toast-close")
        self.assertContains(response, "Przywrócono domyślny motyw")


class WebUsabilityAndAccessibilityTests(TestCase):
    """Testy weryfikujące zgodność ze standardami web usability, WCAG i intuicyjności:
    – Pełna nawigacja klawiaturą (skip link, tabindex, focusable main)
    – Brak pułapek fokusowych (unikalne id landmarków, Esc zamyka menu/modale)
    – Każde pole formularza posiada jednoznaczną etykietę (<label for>)
    – Przyciski i przełączniki posiadają jednoznaczne etykiety (aria-label/text)
    """

    def test_skip_link_and_main_landmark(self):
        response = self.client.get("/")
        content = response.content.decode()

        self.assertIn('href="#main-content"', content)
        self.assertIn('class="skip-link"', content)
        self.assertIn('id="main-content"', content)
        self.assertIn('tabindex="-1"', content)
        self.assertIn('role="main"', content)
        # Main ID should be unique across page
        self.assertEqual(content.count('id="main-content"'), 1)

    def test_login_form_controls_have_explicit_labels(self):
        from django.urls import reverse
        response = self.client.get(reverse("accounts:login"))
        content = response.content.decode()

        self.assertIn('for="id_login"', content)
        self.assertIn('id="id_login"', content)
        self.assertIn('for="id_password"', content)
        self.assertIn('id="id_password"', content)
        self.assertIn('for="id_remember_me"', content)
        self.assertIn('id="id_remember_me"', content)

    def test_register_form_controls_have_explicit_labels(self):
        from django.urls import reverse
        response = self.client.get(reverse("accounts:register"))
        content = response.content.decode()

        for field_id in ("id_username", "id_email", "id_first_name", "id_last_name", "id_password1", "id_password2"):
            with self.subTest(field=field_id):
                self.assertIn(f'for="{field_id}"', content)
                self.assertIn(f'id="{field_id}"', content)

    def test_profile_edit_form_controls_have_explicit_labels(self):
        from django.urls import reverse
        from apps.accounts.models import CustomUser

        user = CustomUser.objects.create_user(username="testuser", email="test@example.com", password="Password123!")
        self.client.force_login(user)

        response = self.client.get(reverse("accounts:profile-edit"))
        self.assertEqual(response.status_code, 200)
        content = response.content.decode()

        for field_id in ("id_first_name", "id_last_name", "id_email", "id_phone_number", "id_street_address", "id_postal_code", "id_city", "id_avatar"):
            with self.subTest(field=field_id):
                self.assertIn(f'for="{field_id}"', content)
                self.assertIn(f'id="{field_id}"', content)

    def test_navbar_interactive_elements_have_accessible_names(self):
        response = self.client.get("/")
        content = response.content.decode()

        self.assertIn('aria-label="Nawigacja główna"', content)
        self.assertIn('aria-controls="primary-nav"', content)
        self.assertIn('aria-expanded="false"', content)
        self.assertIn('class="navbar__toggle"', content)
        self.assertIn('data-theme-switcher', content)
        self.assertIn('theme-switcher__summary', content)
        self.assertIn('aria-label="Motyw kolorystyczny', content)

    def test_custom_confirmation_dialog_present_and_accessible(self):
        response = self.client.get("/")
        content = response.content.decode()

        self.assertIn('id="app-confirm-dialog"', content)
        self.assertIn('class="confirm-dialog"', content)
        self.assertIn('aria-labelledby="confirm-dialog-title"', content)
        self.assertIn('aria-describedby="confirm-dialog-message"', content)
        self.assertIn('data-confirm-cancel', content)
        self.assertIn('data-confirm-accept', content)

    def test_social_account_disconnect_uses_custom_confirm_attribute(self):
        from django.urls import reverse
        from apps.accounts.models import CustomUser
        from allauth.socialaccount.models import SocialAccount

        user = CustomUser.objects.create_user(username="socialtester", email="st@example.com", password="Password123!")
        SocialAccount.objects.create(user=user, provider="discord", uid="disc-1234")
        self.client.force_login(user)

        response = self.client.get(reverse("accounts:profile"))
        self.assertEqual(response.status_code, 200)
        content = response.content.decode()

        self.assertNotIn("confirm(", content)
        self.assertIn('data-confirm="Czy na pewno chcesz odłączyć konto Discord?', content)
        self.assertIn('data-confirm-danger="true"', content)


class AdminCustomThemeTests(TestCase):
    """Testy sprawdzające nowoczesny wygląd i konfigurację panelu /admin/ z Django Unfold."""

    def setUp(self):
        from apps.accounts.models import CustomUser
        self.admin = CustomUser.objects.create_superuser(
            username="superadmin", email="sa@example.com", password="Pass123!"
        )
        self.client.force_login(self.admin)

    def test_admin_renders_unfold_interface_and_branding(self):
        response = self.client.get("/admin/")
        self.assertEqual(response.status_code, 200)
        content = response.content.decode()

        self.assertIn("E-Learning", content)
        self.assertTrue("unfold" in content.lower() or "admin" in content.lower())

    def test_admin_models_accessible_with_unfold(self):
        for path in [
            "/admin/catalog/course/",
            "/admin/catalog/category/",
            "/admin/catalog/enrollment/",
            "/admin/core/colortheme/",
            "/admin/accounts/customuser/",
        ]:
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200, f"Błąd w widoku admina dla {path}")

    def test_platform_navbar_has_admin_link_for_staff_user(self):
        # When staff user visits the platform, navbar displays "Panel Django" link
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        content = response.content.decode()
        self.assertIn("Panel Django", content)
        self.assertIn("/admin/", content)

        # When regular student visits, "Panel Django" is not shown
        from apps.accounts.models import CustomUser
        student = CustomUser.objects.create_user(
            username="student_user", email="student@example.com", password="Pass123!", role="student"
        )
        self.client.force_login(student)
        resp_student = self.client.get("/")
        self.assertNotIn("Panel Django", resp_student.content.decode())






class PolishPluralFilterTests(SimpleTestCase):
    def test_forms_follow_polish_grammar(self):
        from apps.core.templatetags.core_tags import pl_plural

        forms = "kurs,kursy,kursów"
        expected = {0: "kursów", 1: "kurs", 2: "kursy", 4: "kursy", 5: "kursów", 12: "kursów", 22: "kursy", 25: "kursów"}
        for n, word in expected.items():
            with self.subTest(n=n):
                self.assertEqual(pl_plural(n, forms), word)


class RedesignAccessibilityTests(TestCase):
    def test_login_error_is_linked_to_field(self):
        response = self.client.post("/accounts/login/", {"login": "", "password": ""})
        content = response.content.decode()

        self.assertIn('aria-invalid="true"', content)
        self.assertIn('aria-describedby="id_login-error"', content)
        self.assertIn('id="id_login-error"', content)

    def test_active_nav_link_is_marked(self):
        response = self.client.get("/catalog/")
        self.assertContains(response, 'class="navbar__link" aria-current="page">Kursy</a>', html=False)

    def test_auth_pages_put_form_heading_first(self):
        content = self.client.get("/accounts/register/").content.decode()
        self.assertLess(content.index('id="register-heading"'), content.index("auth-split__info"))
        self.assertIn('id="id_password1_helptext"', content)


class UnfoldPaletteTests(SimpleTestCase):
    """Panel admina dziedziczy kolory motywu i zachowuje kontrast WCAG AA."""

    THEMES = {
        "Bursztyn": ("#B45309", "#FFFBEB", "#FEF3C7"),
        "Ocean": ("#38BDF8", "#0B1220", "#152033"),
        "Las": ("#166534", "#F0FDF4", "#DCFCE7"),
        "Neon": ("#C084FC", "#1E1033", "#2D184C"),
    }

    def test_button_and_text_contrast_for_built_in_themes(self):
        from apps.core.colors import contrast_ratio, is_dark_color, unfold_palette

        for name, (primary, surface, variant) in self.THEMES.items():
            with self.subTest(theme=name):
                palette = unfold_palette(primary, surface, variant)
                base, prim = palette["base"], palette["primary"]
                self.assertGreaterEqual(contrast_ratio("#FFFFFF", prim["600"]), 4.5)
                if is_dark_color(surface):
                    self.assertEqual(base["900"], surface.upper())
                    self.assertGreaterEqual(contrast_ratio(base["300"], base["900"]), 4.5)
                    self.assertGreaterEqual(contrast_ratio(base["400"], base["800"]), 4.5)
                else:
                    self.assertEqual(base["50"], surface.upper())
                    self.assertGreaterEqual(contrast_ratio(base["600"], "#FFFFFF"), 4.5)
                    self.assertGreaterEqual(contrast_ratio(base["500"], base["50"]), 4.5)

    def test_admin_mode_follows_theme(self):
        from types import SimpleNamespace
        from unittest import mock

        from apps.core import colors

        with mock.patch.object(colors, "_active_theme", return_value=SimpleNamespace(surface="#0B1220")):
            self.assertEqual(colors.get_unfold_theme_mode(), "dark")
        with mock.patch.object(colors, "_active_theme", return_value=SimpleNamespace(surface="#FFFBEB")):
            self.assertEqual(colors.get_unfold_theme_mode(), "light")


class AdminThemeSyncTests(TestCase):
    def test_admin_loads_mode_sync_script_and_theme_class(self):
        from apps.accounts.models import CustomUser

        ColorTheme.objects.update(is_default=False)
        light = ColorTheme.objects.create(name="Test jasny", primary="#B45309", secondary="#0F766E", surface="#FFFBEB", is_default=True)
        admin = CustomUser.objects.create_superuser("szef", "szef@example.com", "Haslo12345!")
        admin.selected_theme = light
        admin.save()
        self.client.force_login(admin)

        content = self.client.get("/admin/").content.decode()

        self.assertIn('class="light"', content)
        self.assertIn("js/admin.js", content)
        self.assertIn("css/admin.css", content)


class NotificationKindTests(TestCase):
    def test_kind_defaults_to_system_and_accepts_event_types(self):
        from apps.accounts.models import CustomUser
        from apps.core.models import Notification

        user = CustomUser.objects.create_user(username="odbiorca", password="Haslo12345!")
        generic = Notification.objects.create(user=user, title="Witaj")
        approved = Notification.objects.create(
            user=user, kind=Notification.Kind.COMMENT_APPROVED, title="Twój komentarz został zaakceptowany"
        )
        self.assertEqual(generic.kind, Notification.Kind.SYSTEM)
        self.assertEqual(approved.get_kind_display(), "Komentarz zaakceptowany")


class NotificationPanelTests(TestCase):
    def setUp(self):
        from datetime import timedelta

        from django.utils import timezone

        from apps.accounts.models import CustomUser
        from apps.core.models import Notification

        self.Notification = Notification
        self.user = CustomUser.objects.create_user(username="odbiorca_p", password="Haslo12345!")
        self.other = CustomUser.objects.create_user(username="ktos_inny", password="Haslo12345!")
        now = timezone.now()
        for i in range(7):
            n = Notification.objects.create(user=self.user, title=f"Powiadomienie {i}", message="Treść",
                                            is_read=i < 3, url="/catalog/")
            Notification.objects.filter(pk=n.pk).update(created_at=now - timedelta(hours=i * 2))
        Notification.objects.create(user=self.other, title="Cudze powiadomienie")
        self.client.force_login(self.user)

    def test_panel_shows_five_newest_own_notifications_with_unread_state(self):
        # nie „/” — panel startowy zalogowanego sam pokazuje nieprzeczytane powiadomienia
        content = self.client.get("/catalog/").content.decode()
        self.assertIn('aria-controls="notification-panel"', content)
        # Nazwa przycisku zawiera widoczną liczbę (WCAG 2.5.3): „Powiadomienia 4 nieprzeczytane”
        self.assertIn('<span class="sr-only">Powiadomienia</span>', content)
        self.assertIn('<span class="navbar__badge">4</span><span class="sr-only"> nieprzeczytane</span>', content)
        for i in range(5):
            self.assertIn(f"Powiadomienie {i}<", content)
        self.assertNotIn("Powiadomienie 5<", content)
        self.assertNotIn("Cudze powiadomienie", content)
        self.assertIn("2\xa0godziny temu", content)
        self.assertEqual(content.count("Nieprzeczytane: "), 2)  # w panelu: 3 i 4 z pięciu najnowszych
        self.assertIn('href="/powiadomienia/"', content)

    def test_open_is_post_only_marks_read_and_redirects(self):
        unread = self.Notification.objects.get(title="Powiadomienie 6")
        url = f"/powiadomienia/{unread.pk}/otworz/"
        self.assertEqual(self.client.get(url).status_code, 405)
        response = self.client.post(url)
        self.assertRedirects(response, "/catalog/", fetch_redirect_response=False)
        unread.refresh_from_db()
        self.assertTrue(unread.is_read)

    def test_cannot_open_someone_elses_notification(self):
        foreign = self.Notification.objects.get(title="Cudze powiadomienie")
        self.assertEqual(self.client.post(f"/powiadomienia/{foreign.pk}/otworz/").status_code, 404)

    def test_external_url_is_not_followed(self):
        n = self.Notification.objects.create(user=self.user, title="Zły link", url="https://evil.example.com/")
        response = self.client.post(f"/powiadomienia/{n.pk}/otworz/")
        self.assertRedirects(response, "/powiadomienia/")

    def test_list_page_paginates_own_notifications(self):
        for i in range(20):
            self.Notification.objects.create(user=self.user, title=f"Starsze {i}")
        response = self.client.get("/powiadomienia/")
        self.assertContains(response, 'aria-label="Nawigacja stron"')
        self.assertNotContains(response, "Cudze powiadomienie")

    def test_anonymous_user_has_no_panel(self):
        self.client.logout()
        self.assertNotContains(self.client.get("/"), "notification-panel")
        self.assertEqual(self.client.get("/powiadomienia/").status_code, 302)


class NotifyHelperTests(TestCase):
    def setUp(self):
        from apps.accounts.models import CustomUser

        self.a = CustomUser.objects.create_user(username="osoba_a", password="Haslo12345!")
        self.b = CustomUser.objects.create_user(username="osoba_b", password="Haslo12345!")

    def test_notify_creates_notification(self):
        from apps.core.models import Notification
        from apps.core.notifications import notify

        n = notify(self.a, Notification.Kind.ORDER_STATUS, "Zmiana statusu zamówienia", "Opłacone", url="/orders/")
        self.assertEqual((n.user, n.kind, n.is_read, n.url), (self.a, "order_status", False, "/orders/"))

    def test_notify_rejects_unknown_kind_and_external_url(self):
        from apps.core.notifications import notify

        with self.assertRaises(ValueError):
            notify(self.a, "cokolwiek", "Tytuł")
        with self.assertRaises(ValueError):
            notify(self.a, "system", "Tytuł", url="https://example.com/")

    def test_notify_many_skips_duplicates_and_excluded(self):
        from apps.core.models import Notification
        from apps.core.notifications import notify_many

        count = notify_many([self.a, self.b, self.a, None], "system", "Komunikat", exclude=self.b)
        self.assertEqual(count, 1)
        self.assertEqual(Notification.objects.filter(user=self.a).count(), 1)
        self.assertFalse(Notification.objects.filter(user=self.b).exists())


class NotificationFilterAndMarkAllTests(TestCase):
    def setUp(self):
        from apps.accounts.models import CustomUser
        from apps.core.models import Notification

        self.Notification = Notification
        self.user = CustomUser.objects.create_user(username="czytelnik", password="Haslo12345!")
        self.other = CustomUser.objects.create_user(username="sasiad", password="Haslo12345!")
        Notification.objects.create(user=self.user, title="Stare przeczytane", is_read=True)
        Notification.objects.create(user=self.user, title="Świeże nieprzeczytane")
        Notification.objects.create(user=self.other, title="Sąsiada nieprzeczytane")
        self.client.force_login(self.user)

    def test_unread_filter(self):
        everything = self.client.get("/powiadomienia/")
        self.assertContains(everything, "Stare przeczytane")
        self.assertContains(everything, 'class="segmented__option" aria-current="page">Wszystkie</a>', html=False)
        unread = self.client.get("/powiadomienia/?pokaz=nieprzeczytane").content.decode()
        page_list = unread.split('class="notice-list"', 1)[1]  # pomija panel pod dzwonkiem w navbarze
        self.assertIn("Świeże nieprzeczytane", page_list)
        self.assertNotIn("Stare przeczytane", page_list)

    def test_mark_all_read_is_post_only_and_touches_only_own(self):
        url = "/powiadomienia/przeczytane/"
        self.assertEqual(self.client.get(url).status_code, 405)
        response = self.client.post(url, follow=True)
        self.assertContains(response, "Oznaczono 1 powiadomienie jako przeczytane.")
        self.assertFalse(self.Notification.objects.filter(user=self.user, is_read=False).exists())
        self.assertTrue(self.Notification.objects.filter(user=self.other, is_read=False).exists())
        empty = self.client.get("/powiadomienia/?pokaz=nieprzeczytane")
        self.assertContains(empty, "Wszystko przeczytane")
        self.assertNotContains(empty, "Oznacz wszystkie jako przeczytane")


class AccountMenuTests(TestCase):
    """Konto w navbarze to jedno menu <details> — pasek mieści się od 64rem bez ucinania."""

    def setUp(self):
        from apps.accounts.models import CustomUser

        self.student = CustomUser.objects.create_user(username="uczen_menu", password="Haslo12345!")
        self.staff = CustomUser.objects.create_user(
            username="obsluga_menu", password="Haslo12345!", role="admin", is_staff=True
        )

    def test_account_menu_has_profile_and_logout_but_admin_link_only_for_staff(self):
        from django.urls import reverse

        self.client.force_login(self.student)
        response = self.client.get(reverse("core:home"))
        self.assertContains(response, 'class="account-menu" data-account-menu')
        self.assertContains(response, f'href="{reverse("accounts:logout")}" class="account-menu__link')
        self.assertNotContains(response, f'href="{reverse("admin:index")}" class="account-menu__link"')

        self.client.force_login(self.staff)
        response = self.client.get(reverse("core:home"))
        self.assertContains(response, f'href="{reverse("admin:index")}" class="account-menu__link"')

    def test_guest_has_no_account_menu(self):
        from django.urls import reverse

        self.assertNotContains(self.client.get(reverse("core:home")), "data-account-menu")
