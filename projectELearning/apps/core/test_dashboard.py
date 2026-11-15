"""Panele startowe po zalogowaniu: gość — prezentacja, uczeń / nauczyciel / admin — własny panel."""

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from apps.catalog.models import Category, Comment, Course, Enrollment

User = get_user_model()
HOME = reverse("core:home")


class DashboardTests(TestCase):
    def setUp(self):
        self.admin = User.objects.create_user(username="szef", password="Haslo12345!", role="admin", is_staff=True)
        self.teacher = User.objects.create_user(username="nauczyciel", password="Haslo12345!", role="moderator")
        self.other_teacher = User.objects.create_user(username="inny", password="Haslo12345!", role="moderator")
        self.student = User.objects.create_user(username="uczen", password="Haslo12345!", first_name="Ola")

        self.it = Category.objects.create(name="IT")
        self.it.moderators.add(self.teacher)
        self.art = Category.objects.create(name="Sztuka")
        self.art.moderators.add(self.other_teacher)

        self.course = Course.objects.create(title="Kurs Pythona", category=self.it, author=self.teacher,
                                            status="approved", is_published=True)
        Course.objects.create(title="Propozycja IT", category=self.it, author=self.student, status="pending")
        Course.objects.create(title="Propozycja Sztuka", category=self.art, author=self.student, status="pending")
        Enrollment.objects.create(user=self.student, course=self.course)
        Comment.objects.create(course=self.course, author=self.student, content="Świetny kurs", is_approved=False)

    def test_guest_sees_landing(self):
        response = self.client.get(HOME)
        self.assertTemplateUsed(response, "core/home.html")
        self.assertContains(response, 'id="hero-title"')

    def test_landing_still_available_for_logged_in(self):
        self.client.force_login(self.student)
        response = self.client.get(reverse("core:about"))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "core/home.html")
        self.assertContains(self.client.get(HOME), reverse("core:about"))

    def test_student_panel(self):
        self.client.force_login(self.student)
        response = self.client.get(HOME)
        self.assertTemplateUsed(response, "core/dashboard.html")
        self.assertEqual(response.context["dashboard_role"], "student")
        self.assertNotContains(response, 'id="hero-title"')
        self.assertContains(response, "Cześć, Ola.")
        self.assertContains(response, "Kontynuuj naukę")
        self.assertContains(response, "Kurs Pythona")
        self.assertEqual(response.context["proposed_count"], 2)
        self.assertNotContains(response, "Do zrobienia")
        self.assertNotContains(response, reverse("catalog:category_manage"))

    def test_teacher_panel_shows_only_own_categories(self):
        self.client.force_login(self.teacher)
        response = self.client.get(HOME)
        self.assertEqual(response.context["dashboard_role"], "teacher")
        self.assertEqual(response.context["submission_count"], 1)
        self.assertEqual([c.title for c in response.context["submissions"]], ["Propozycja IT"])
        self.assertEqual(response.context["comment_count"], 1)
        self.assertContains(response, "Propozycja IT")
        self.assertNotContains(response, "Propozycja Sztuka")
        self.assertEqual([row["category"] for row in response.context["category_rows"]], [self.it])
        self.assertEqual(response.context["student_total"], 1)
        self.assertContains(response, reverse("catalog:category_manage"))
        self.assertContains(response, "Panel nauczyciela")

    def test_admin_panel_has_category_management(self):
        self.client.force_login(self.admin)
        response = self.client.get(HOME)
        self.assertEqual(response.context["dashboard_role"], "admin")
        self.assertEqual(response.context["submission_count"], 2)
        self.assertEqual(response.context["category_total"], 2)
        self.assertEqual(response.context["uncovered_count"], 0)
        self.assertContains(response, reverse("catalog:category_create"))
        self.assertContains(response, f'{reverse("catalog:category_create")}?rodzic={self.it.pk}')
        self.assertContains(response, reverse("catalog:category_manage"))
        self.assertContains(response, "Panel administratora")

    def test_admin_panel_flags_category_without_teacher(self):
        Category.objects.create(name="Języki")
        self.client.force_login(self.admin)
        response = self.client.get(HOME)
        self.assertEqual(response.context["uncovered_count"], 1)
        self.assertContains(response, "bez nauczyciela")

    def test_category_create_reachable_from_admin_panel(self):
        self.client.force_login(self.admin)
        self.assertEqual(self.client.get(reverse("catalog:category_create")).status_code, 200)
        response = self.client.post(reverse("catalog:category_create"), {"name": "Języki", "description": "", "is_active": "on"})
        self.assertEqual(response.status_code, 302)
        self.assertTrue(Category.objects.filter(name="Języki").exists())
