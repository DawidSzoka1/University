"""Koszyk z kumulacją, zamówienie z adresem z profilu, e-mail, Stripe (sesja, powrót, webhook), statusy, kody."""

import hashlib
import hmac
import json
import time
from decimal import Decimal
from unittest import mock

from django.contrib.auth import get_user_model
from django.core import mail
from django.test import TestCase, override_settings
from django.urls import reverse
from rest_framework_simplejwt.tokens import AccessToken

from apps.catalog.models import Category, Course, Enrollment
from apps.core.models import Notification
from apps.orders.models import AccessCode, Cart, CartItem, Order

User = get_user_model()

ADDRESS = {
    "first_name": "Anna", "last_name": "Nowak", "email": "anna@example.com",
    "street_address": "ul. Długa 5/2", "postal_code": "31-234", "city": "Kraków",
    "phone_number": "+48 600 100 200", "accept_terms": "on",
}
STRIPE = {"STRIPE_SECRET_KEY": "sk_test_x", "STRIPE_WEBHOOK_SECRET": "whsec_test", "SITE_URL": "http://testserver"}


class ShopFixture(TestCase):
    def setUp(self):
        self.teacher = User.objects.create_user(username="nauczyciel", password="Haslo12345!", role="moderator")
        self.student = User.objects.create_user(
            username="uczen", password="Haslo12345!", email="uczen@example.com", first_name="Ala", last_name="Kot",
            street_address="ul. Krótka 1", postal_code="00-001", city="Warszawa",
        )
        self.other = User.objects.create_user(username="inny", password="Haslo12345!", email="inny@example.com")
        category = Category.objects.create(name="Python")
        common = {"category": category, "author": self.teacher, "status": Course.Status.APPROVED, "is_published": True}
        self.course = Course.objects.create(title="Django", price=Decimal("199.00"), is_free=False, **common)
        self.course2 = Course.objects.create(title="Pandas", price=Decimal("49.50"), is_free=False, **common)
        self.free = Course.objects.create(title="Start", price=Decimal("0"), is_free=True, **common)

    def add(self, course=None, client=None):
        return (client or self.client).post(reverse("orders:cart_add", args=[(course or self.course).slug]))

    def place(self, data=None, quantity=1, **extra):
        self.client.force_login(self.student)
        for _ in range(quantity):
            self.add()
        return self.post(reverse("orders:checkout"), data or ADDRESS, **extra)

    def post(self, *args, **kwargs):
        """POST z wykonaniem callbacków on_commit (e-maile, powiadomienia)."""
        with self.captureOnCommitCallbacks(execute=True):
            return self.client.post(*args, **kwargs)


class CartTests(ShopFixture):
    def test_adding_twice_accumulates_quantity_for_guest(self):
        self.add()
        response = self.add()
        self.assertRedirects(response, reverse("catalog:course_detail", args=[self.course.slug]))
        item = CartItem.objects.get()
        self.assertEqual(item.quantity, 2)
        self.assertEqual(item.line_total, Decimal("398.00"))
        page = self.client.get(reverse("orders:cart"))
        self.assertContains(page, "398,00")
        self.assertContains(page, 'aria-label="Koszyk, 2 sztuki"')

    def test_free_and_unpublished_courses_cannot_be_added(self):
        self.add(self.free)
        self.course2.is_published = False
        self.course2.save()
        self.add(self.course2)
        self.assertFalse(CartItem.objects.exists())

    def test_quantity_is_capped(self):
        for _ in range(12):
            self.add()
        self.assertEqual(CartItem.objects.get().quantity, 10)

    def test_guest_cart_merges_into_account_on_login(self):
        self.client.force_login(self.student)
        self.add()
        self.client.logout()
        self.add()
        self.add(self.course2)
        self.client.post(reverse("accounts:login"), {"login": "uczen", "password": "Haslo12345!"})
        cart = Cart.objects.get(user=self.student)
        self.assertEqual({i.course.title: i.quantity for i in cart.items.all()}, {"Django": 2, "Pandas": 1})
        self.assertEqual(Cart.objects.count(), 1)

    def test_update_and_remove_are_post_only_and_scoped_to_own_cart(self):
        self.client.force_login(self.student)
        self.add()
        item = CartItem.objects.get()
        url = reverse("orders:cart_update", args=[item.pk])
        self.assertEqual(self.client.get(url).status_code, 405)
        self.client.post(url, {"step": "+1"})
        self.client.post(url, {"quantity": "5"})
        item.refresh_from_db()
        self.assertEqual(item.quantity, 5)
        self.client.force_login(self.other)
        self.assertEqual(self.client.post(reverse("orders:cart_remove", args=[item.pk])).status_code, 404)
        self.client.force_login(self.student)
        self.client.post(url, {"quantity": "0"})
        self.assertFalse(CartItem.objects.exists())

    def test_course_page_shows_add_button_and_cart_state(self):
        url = reverse("catalog:course_detail", args=[self.course.slug])
        self.assertContains(self.client.get(url), "Dodaj do koszyka")
        self.add()
        self.assertContains(self.client.get(url), "W koszyku: <strong>1 sztuka</strong>")


class CheckoutTests(ShopFixture):
    def test_form_is_prefilled_from_profile(self):
        self.client.force_login(self.student)
        self.add()
        page = self.client.get(reverse("orders:checkout"))
        self.assertContains(page, 'value="ul. Krótka 1"')
        self.assertContains(page, 'value="uczen@example.com"')
        self.assertContains(page, '<label for="id_postal_code"')

    def test_checkout_requires_login_and_items(self):
        self.assertEqual(self.client.get(reverse("orders:checkout")).status_code, 302)
        self.client.force_login(self.student)
        self.assertRedirects(self.client.get(reverse("orders:checkout")), reverse("orders:cart"))

    def test_invalid_data_is_kept_and_described(self):
        response = self.place({**ADDRESS, "postal_code": "31234", "accept_terms": ""})
        self.assertContains(response, "formacie 00-000")
        self.assertContains(response, 'value="ul. Długa 5/2"')
        self.assertContains(response, 'aria-invalid="true"')
        self.assertFalse(Order.objects.exists())

    @override_settings(STRIPE_SECRET_KEY="")
    def test_order_is_created_pending_with_email_and_profile_update(self):
        response = self.place({**ADDRESS, "save_address": "on"}, quantity=2)
        order = Order.objects.get()
        self.assertRedirects(response, order.get_absolute_url(), fetch_redirect_response=False)
        self.assertEqual(order.status, Order.Status.PENDING_PAYMENT)
        self.assertEqual(order.total, Decimal("398.00"))
        self.assertTrue(order.number.startswith("ORD-"))
        self.assertEqual(order.items.get().quantity, 2)
        self.assertFalse(CartItem.objects.exists())
        self.student.refresh_from_db()
        self.assertEqual((self.student.city, self.student.postal_code), ("Kraków", "31-234"))
        self.assertEqual(self.student.email, "uczen@example.com")  # e-mail konta nie jest nadpisywany
        self.assertEqual(len(mail.outbox), 1)
        email = mail.outbox[0]
        self.assertIn("oczekuje na płatność", email.subject)
        self.assertEqual(email.to, ["anna@example.com"])
        self.assertIn("OCZEKUJE NA PŁATNOŚĆ", email.body)
        self.assertIn("398,00", email.alternatives[0][0])
        self.assertEqual(order.history.count(), 1)

    @override_settings(STRIPE_SECRET_KEY="")
    def test_without_save_address_profile_stays(self):
        self.place()
        self.student.refresh_from_db()
        self.assertEqual(self.student.city, "Warszawa")


def fake_session(order, **overrides):
    data = {
        "id": "cs_test_123", "url": "https://checkout.stripe.com/c/pay/cs_test_123",
        "payment_status": "paid", "amount_total": int(order.total * 100), "currency": "pln",
        "metadata": {"order_number": order.number}, "client_reference_id": order.number,
        "payment_intent": "pi_test_1",
    }
    data.update(overrides)
    return data


class FakeSession(dict):
    def __getattr__(self, name):
        return self[name]


@override_settings(**STRIPE)
class StripeTests(ShopFixture):
    def checkout(self, quantity=1):
        with mock.patch("stripe.checkout.Session.create") as create:
            create.side_effect = lambda **kwargs: FakeSession(id="cs_test_123", url="https://checkout.stripe.com/c/pay/cs_test_123")
            response = self.place(quantity=quantity)
        self.create_kwargs = create.call_args.kwargs
        return response, Order.objects.get()

    def test_checkout_redirects_to_stripe_with_order_data(self):
        response, order = self.checkout(quantity=2)
        self.assertEqual(response.status_code, 303)
        self.assertEqual(response["Location"], "https://checkout.stripe.com/c/pay/cs_test_123")
        self.assertEqual(order.payment_reference, "cs_test_123")
        self.assertEqual(self.create_kwargs["line_items"][0]["quantity"], 2)
        self.assertEqual(self.create_kwargs["line_items"][0]["price_data"]["unit_amount"], 19900)
        self.assertEqual(self.create_kwargs["line_items"][0]["price_data"]["currency"], "pln")
        self.assertEqual(self.create_kwargs["metadata"], {"order_number": order.number})
        self.assertIn(f"/orders/{order.number}/platnosc/powrot/?session_id={{CHECKOUT_SESSION_ID}}", self.create_kwargs["success_url"])

    def test_return_from_stripe_marks_paid_and_grants_access_and_codes(self):
        _, order = self.checkout(quantity=3)
        mail.outbox.clear()
        with mock.patch("stripe.checkout.Session.retrieve", return_value=FakeSession(fake_session(order))):
            with self.captureOnCommitCallbacks(execute=True):
                response = self.client.get(reverse("orders:payment_return", args=[order.number]), {"session_id": "cs_test_123"})
        self.assertRedirects(response, order.get_absolute_url(), fetch_redirect_response=False)
        order.refresh_from_db()
        self.assertEqual(order.status, Order.Status.PAID)
        self.assertEqual(order.payment_intent, "pi_test_1")
        self.assertTrue(Enrollment.objects.filter(user=self.student, course=self.course).exists())
        self.assertEqual(AccessCode.objects.count(), 2)  # 3 sztuki: 1 dla kupującego + 2 kody
        self.assertTrue(Notification.objects.filter(user=self.student, kind="order_status").exists())
        self.assertIn("Płatność za zamówienie", mail.outbox[-1].subject)
        self.assertIn(AccessCode.objects.first().code, mail.outbox[-1].body)

    def test_amount_mismatch_or_foreign_session_does_not_mark_paid(self):
        _, order = self.checkout()
        with mock.patch("stripe.checkout.Session.retrieve", return_value=FakeSession(fake_session(order, amount_total=100))):
            self.client.get(reverse("orders:payment_return", args=[order.number]), {"session_id": "cs_test_123"})
        with mock.patch("stripe.checkout.Session.retrieve") as retrieve:
            self.client.get(reverse("orders:payment_return", args=[order.number]), {"session_id": "cs_other"})
            retrieve.assert_not_called()
        order.refresh_from_db()
        self.assertEqual(order.status, Order.Status.PENDING_PAYMENT)

    def webhook(self, payload, secret="whsec_test"):
        body = json.dumps(payload)
        timestamp = int(time.time())
        signature = hmac.new(secret.encode(), f"{timestamp}.{body}".encode(), hashlib.sha256).hexdigest()
        return self.client.post(
            reverse("orders:stripe_webhook"), data=body, content_type="application/json",
            HTTP_STRIPE_SIGNATURE=f"t={timestamp},v1={signature}",
        )

    def event(self, order, type_="checkout.session.completed", **overrides):
        return {"id": "evt_1", "object": "event", "type": type_,
                "data": {"object": {"object": "checkout.session", **fake_session(order, **overrides)}}}

    def test_signed_webhook_marks_paid_once(self):
        _, order = self.checkout()
        response = self.webhook(self.event(order))
        self.assertEqual(response.status_code, 200)
        order.refresh_from_db()
        self.assertEqual(order.status, Order.Status.PAID)
        self.assertEqual(self.webhook(self.event(order)).status_code, 200)  # ponowione zdarzenie
        self.assertEqual(order.history.filter(status=Order.Status.PAID).count(), 1)
        self.assertEqual(Enrollment.objects.filter(user=self.student, course=self.course).count(), 1)

    def test_webhook_rejects_bad_signature_and_ignores_unpaid(self):
        _, order = self.checkout()
        self.assertEqual(self.webhook(self.event(order), secret="whsec_zly").status_code, 400)
        self.assertEqual(self.webhook(self.event(order, payment_status="unpaid")).status_code, 200)
        self.assertEqual(self.webhook(self.event(order, type_="checkout.session.expired")).status_code, 200)
        order.refresh_from_db()
        self.assertEqual(order.status, Order.Status.PENDING_PAYMENT)
        self.assertEqual(self.client.get(reverse("orders:stripe_webhook")).status_code, 405)

    def test_webhook_secret_can_come_from_stripe_cli_file(self):
        import tempfile
        from pathlib import Path

        _, order = self.checkout()
        with tempfile.TemporaryDirectory() as folder:
            secret_file = Path(folder) / "webhook_secret"
            secret_file.write_text("whsec_z_pliku\n")
            with override_settings(STRIPE_WEBHOOK_SECRET="", STRIPE_WEBHOOK_SECRET_FILE=secret_file):
                self.assertEqual(self.webhook(self.event(order), secret="whsec_test").status_code, 400)
                self.assertEqual(self.webhook(self.event(order), secret="whsec_z_pliku").status_code, 200)
        order.refresh_from_db()
        self.assertEqual(order.status, Order.Status.PAID)

    def test_order_page_checks_stripe_when_webhook_missed(self):
        _, order = self.checkout()
        with mock.patch("stripe.checkout.Session.retrieve", return_value=FakeSession(fake_session(order))) as retrieve:
            with self.captureOnCommitCallbacks(execute=True):
                page = self.client.get(order.get_absolute_url())
            retrieve.assert_called_once()
        self.assertContains(page, "Stripe potwierdził płatność")
        order.refresh_from_db()
        self.assertEqual(order.status, Order.Status.PAID)
        with mock.patch("stripe.checkout.Session.retrieve") as retrieve:
            self.client.get(order.get_absolute_url())
            retrieve.assert_not_called()  # opłacone — nie pytamy Stripe ponownie

    def test_pay_again_and_cancel(self):
        _, order = self.checkout()
        with mock.patch("stripe.checkout.Session.create", return_value=FakeSession(id="cs_test_2", url="https://checkout.stripe.com/x")):
            response = self.client.post(reverse("orders:order_pay", args=[order.number]))
        self.assertEqual(response.status_code, 303)
        self.client.post(reverse("orders:order_cancel", args=[order.number]))
        order.refresh_from_db()
        self.assertEqual(order.status, Order.Status.CANCELLED)


@override_settings(STRIPE_SECRET_KEY="")
class ManagementTests(ShopFixture):
    def setUp(self):
        super().setUp()
        self.place(quantity=2)
        self.order = Order.objects.get()

    def test_owner_and_manager_see_order_others_do_not(self):
        url = self.order.get_absolute_url()
        self.assertContains(self.client.get(url), "Oczekuje na płatność")
        self.client.force_login(self.other)
        self.assertEqual(self.client.get(url).status_code, 404)
        self.assertEqual(self.client.get(reverse("orders:manage")).status_code, 404)
        self.client.force_login(self.teacher)
        self.assertContains(self.client.get(reverse("orders:manage")), self.order.number)
        self.assertContains(self.client.get(url), "Zmień status")

    def test_manager_changes_status_with_history_and_access(self):
        self.client.force_login(self.teacher)
        url = reverse("orders:order_status", args=[self.order.number])
        self.assertEqual(self.client.get(url).status_code, 405)
        self.post(url, {"status": "paid", "note": "Przelew tradycyjny"})
        self.order.refresh_from_db()
        self.assertEqual(self.order.status, Order.Status.PAID)
        self.assertEqual(self.order.payment_provider, "manual")
        self.assertTrue(Enrollment.objects.filter(user=self.student, course=self.course).exists())
        self.assertEqual(AccessCode.objects.count(), 1)
        self.post(url, {"status": "completed"})
        self.order.refresh_from_db()
        self.assertEqual(self.order.status, Order.Status.COMPLETED)
        self.assertEqual(
            list(self.order.history.values_list("status", flat=True)), ["pending_payment", "paid", "completed"]
        )
        self.assertEqual(Notification.objects.filter(user=self.student, kind="order_status").count(), 2)

    def test_invalid_transition_is_reported(self):
        self.client.force_login(self.teacher)
        response = self.client.post(reverse("orders:order_status", args=[self.order.number]), {"status": "completed"})
        self.assertContains(response, "Nie można zmienić statusu")
        self.order.refresh_from_db()
        self.assertEqual(self.order.status, Order.Status.PENDING_PAYMENT)

    def test_access_code_redeem_once(self):
        self.client.force_login(self.teacher)
        self.client.post(reverse("orders:order_status", args=[self.order.number]), {"status": "paid"})
        code = AccessCode.objects.get().code
        self.client.force_login(self.other)
        response = self.client.post(reverse("orders:redeem"), {"code": code.lower()})
        self.assertRedirects(response, reverse("catalog:course_detail", args=[self.course.slug]))
        self.assertTrue(Enrollment.objects.filter(user=self.other, course=self.course).exists())
        self.client.force_login(self.teacher)
        self.assertContains(self.client.post(reverse("orders:redeem"), {"code": code}), "już wykorzystany")
        self.assertContains(self.client.post(reverse("orders:redeem"), {"code": "KURS-ZLY0-KOD0"}), "Nie znaleziono")

    def test_order_lists_and_moderation_tab(self):
        self.assertContains(self.client.get(reverse("orders:order_list")), self.order.number)
        self.client.force_login(self.teacher)
        page = self.client.get(reverse("orders:manage"), {"status": "paid"})
        self.assertNotContains(page, self.order.number)
        self.teacher.is_staff = True  # administrator widzi wszystkie kolejki moderacji
        self.teacher.save()
        self.assertContains(self.client.get(reverse("catalog:moderation_comments")), reverse("orders:manage"))


class MoneyFormatTests(TestCase):
    def test_polish_amounts(self):
        from apps.core.money import format_pln

        self.assertEqual(format_pln(Decimal("1234.5")), "1 234,50 zł")
        self.assertEqual(format_pln(Decimal("0")), "0,00 zł")


class CartApiTests(ShopFixture):
    """API koszyka: tylko JWT, te same reguły co formularze (kumulacja, limit, płatne kursy)."""

    def auth(self, user=None):
        token = AccessToken.for_user(user or self.student)
        return {"HTTP_AUTHORIZATION": f"Bearer {token}"}

    def api_add(self, course, quantity=1, user=None):
        return self.client.post(
            reverse("orders:api_cart_add"), {"course": course, "quantity": quantity},
            content_type="application/json", **self.auth(user),
        )

    def api_item(self, method, pk, data=None, user=None):
        call = getattr(self.client, method)
        return call(
            reverse("orders:api_cart_item", args=[pk]), data or {}, content_type="application/json", **self.auth(user)
        )

    def test_requires_jwt_not_session(self):
        url = reverse("orders:api_cart")
        self.assertEqual(self.client.get(url).status_code, 401)
        self.client.force_login(self.student)
        self.assertEqual(self.client.get(url).status_code, 401)
        self.assertEqual(self.client.get(url, **self.auth()).status_code, 200)

    def test_empty_cart(self):
        data = self.client.get(reverse("orders:api_cart"), **self.auth()).json()
        self.assertEqual(data["items"], [])
        self.assertEqual(data["count"], 0)
        self.assertEqual(data["total_display"], "0,00\u00a0zł")

    def test_adding_twice_accumulates(self):
        first = self.api_add(self.course.slug)
        self.assertEqual(first.status_code, 201)
        self.assertTrue(first.json()["created"])
        second = self.api_add(self.course.pk, quantity=2)
        self.assertEqual(second.status_code, 200)
        data = second.json()
        self.assertFalse(data["created"])
        self.assertEqual(CartItem.objects.get(cart__user=self.student).quantity, 3)
        self.assertEqual(data["cart"]["count"], 3)
        self.assertEqual(data["cart"]["total"], "597.00")
        self.assertEqual(data["cart"]["items"][0]["note"], "1 sztuka dla Ciebie + 2 kody dostępu dla innych osób.")
        self.assertIn("teraz 3 sztuki", data["detail"])

    def test_add_is_capped_at_max_quantity(self):
        self.api_add(self.course.slug, quantity=8)
        data = self.api_add(self.course.slug, quantity=5).json()
        self.assertTrue(data["capped"])
        self.assertEqual(data["cart"]["items"][0]["quantity"], 10)

    def test_add_rejects_free_unknown_and_bad_quantity(self):
        self.assertEqual(self.api_add(self.free.slug).status_code, 400)
        self.assertEqual(self.api_add("nie-ma-takiego").status_code, 404)
        self.assertEqual(self.api_add(self.course.slug, quantity="dużo").status_code, 400)
        self.assertEqual(self.api_add(self.course.slug, quantity=0).status_code, 400)
        self.assertFalse(CartItem.objects.exists())

    def test_patch_changes_quantity_and_zero_removes(self):
        item_id = self.api_add(self.course.slug).json()["item_id"]
        response = self.api_item("patch", item_id, {"quantity": 4})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(CartItem.objects.get(pk=item_id).quantity, 4)
        self.assertEqual(response.json()["cart"]["total_display"], "796,00\u00a0zł")

        self.assertEqual(self.api_item("patch", item_id, {"quantity": 11}).status_code, 400)
        self.assertEqual(self.api_item("patch", item_id, {"quantity": -1}).status_code, 400)
        self.assertEqual(CartItem.objects.get(pk=item_id).quantity, 4)

        removed = self.api_item("patch", item_id, {"quantity": 0}).json()
        self.assertIsNone(removed["item_id"])
        self.assertFalse(CartItem.objects.filter(pk=item_id).exists())

    def test_delete_removes_item(self):
        item_id = self.api_add(self.course.slug).json()["item_id"]
        self.api_add(self.course2.slug)
        data = self.api_item("delete", item_id).json()
        self.assertEqual([i["course"]["slug"] for i in data["cart"]["items"]], [self.course2.slug])

    def test_cannot_touch_someone_elses_item(self):
        item_id = self.api_add(self.course.slug).json()["item_id"]
        self.assertEqual(self.api_item("patch", item_id, {"quantity": 2}, user=self.other).status_code, 404)
        self.assertEqual(self.api_item("delete", item_id, user=self.other).status_code, 404)
        self.assertEqual(CartItem.objects.get(pk=item_id).quantity, 1)

    def test_owned_course_note(self):
        Enrollment.objects.create(user=self.student, course=self.course)
        item = self.api_add(self.course.slug, quantity=2).json()["cart"]["items"][0]
        self.assertTrue(item["owned"])
        self.assertEqual(item["code_count"], 2)
        self.assertEqual(item["note"], "Masz już ten kurs — wszystkie 2 sztuki to kody dostępu dla innych osób.")

    def test_session_token_endpoint(self):
        url = reverse("accounts:api_session_token")
        self.assertEqual(self.client.post(url).status_code, 302)  # niezalogowany → logowanie
        self.client.force_login(self.student)
        self.assertEqual(self.client.get(url).status_code, 405)
        token = self.client.post(url).json()["access"]
        self.client.logout()
        response = self.client.get(reverse("orders:api_cart"), HTTP_AUTHORIZATION=f"Bearer {token}")
        self.assertEqual(response.status_code, 200)

    def test_cart_page_enables_api_only_for_logged_in(self):
        self.add()
        page = self.client.get(reverse("orders:cart"))
        self.assertNotContains(page, "data-cart\n")
        self.assertNotContains(page, "js/cart.js")
        self.client.force_login(self.student)
        self.add()
        page = self.client.get(reverse("orders:cart"))
        self.assertContains(page, reverse("orders:api_cart_item", args=[0]))
        self.assertContains(page, "js/cart.js")
