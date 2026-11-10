"""Blokada logowania po 5 nieudanych próbach i zmiana / ustawienie hasła."""

from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.test import TestCase
from django.urls import reverse

from apps.accounts import lockout

User = get_user_model()


class LoginLockoutTests(TestCase):
    url = reverse("accounts:login")

    def setUp(self):
        cache.clear()
        self.user = User.objects.create_user(username="ola", email="ola@example.com", password="Dobre-haslo-123")

    def fail(self, login="ola"):
        return self.client.post(self.url, {"login": login, "password": "zle-haslo"})

    def test_remaining_attempts_shown(self):
        response = self.fail()
        self.assertContains(response, "Pozostało prób przed czasową blokadą: 4")

    def test_locks_after_five_failures_even_with_correct_password(self):
        for _ in range(4):
            self.fail()
        response = self.fail(login="ola@example.com")  # e-mail to to samo konto
        self.assertContains(response, "zablokowane na 5 minut")
        response = self.client.post(self.url, {"login": "ola", "password": "Dobre-haslo-123"})
        self.assertContains(response, "Za dużo nieudanych prób")
        self.assertNotIn("_auth_user_id", self.client.session)
        self.assertGreater(lockout.locked_for("ola"), 0)

    def test_success_resets_counter(self):
        for _ in range(3):
            self.fail()
        self.client.post(self.url, {"login": "ola", "password": "Dobre-haslo-123"})
        self.assertIn("_auth_user_id", self.client.session)
        self.client.logout()
        self.assertContains(self.fail(), "Pozostało prób przed czasową blokadą: 4")

    def test_lock_expires(self):
        for _ in range(5):
            self.fail()
        lockout.reset("ola")
        self.client.post(self.url, {"login": "ola", "password": "Dobre-haslo-123"})
        self.assertIn("_auth_user_id", self.client.session)


class PasswordChangeTests(TestCase):
    url = reverse("accounts:password_change")

    def setUp(self):
        cache.clear()
        self.user = User.objects.create_user(username="ola", password="Stare-haslo-123")

    def test_requires_login(self):
        self.assertEqual(self.client.get(self.url).status_code, 302)

    def test_change_keeps_session(self):
        self.client.force_login(self.user)
        response = self.client.get(self.url)
        self.assertContains(response, 'for="id_old_password"')
        response = self.client.post(self.url, {
            "old_password": "Stare-haslo-123", "new_password1": "Nowe-haslo-456", "new_password2": "Nowe-haslo-456",
        })
        self.assertRedirects(response, reverse("accounts:profile"), fetch_redirect_response=False)
        self.user.refresh_from_db()
        self.assertTrue(self.user.check_password("Nowe-haslo-456"))
        self.assertEqual(self.client.get(reverse("accounts:profile")).status_code, 200)  # nadal zalogowany

    def test_wrong_old_password_and_mismatch(self):
        self.client.force_login(self.user)
        response = self.client.post(self.url, {
            "old_password": "zle", "new_password1": "Nowe-haslo-456", "new_password2": "Inne-haslo-789",
        })
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'aria-invalid="true"')
        self.user.refresh_from_db()
        self.assertTrue(self.user.check_password("Stare-haslo-123"))

    def test_social_only_account_sets_password(self):
        self.user.set_unusable_password()
        self.user.save()
        self.client.force_login(self.user)
        response = self.client.get(self.url)
        self.assertContains(response, "Ustaw hasło")
        self.assertNotContains(response, "id_old_password")
        self.client.post(self.url, {"new_password1": "Nowe-haslo-456", "new_password2": "Nowe-haslo-456"})
        self.user.refresh_from_db()
        self.assertTrue(self.user.check_password("Nowe-haslo-456"))

    def test_profile_links_to_password_change(self):
        self.client.force_login(self.user)
        self.assertContains(self.client.get(reverse("accounts:profile")), self.url)
