"""Statystyki platformy (tylko administrator)."""

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from apps.catalog.models import Category, Course, Enrollment

User = get_user_model()


class PlatformStatsTests(TestCase):
    url = reverse("core:stats")

    def setUp(self):
        self.admin = User.objects.create_user(username="szef", password="Haslo12345!", role="admin", is_staff=True)
        self.teacher = User.objects.create_user(username="nauczyciel", password="Haslo12345!", role="moderator")
        self.students = [User.objects.create_user(username=f"u{i}", password="Haslo12345!") for i in range(4)]
        course = Course.objects.create(title="Kurs A", category=Category.objects.create(name="IT"), author=self.teacher,
                                       status="approved", is_published=True)
        for student in self.students:
            Enrollment.objects.create(user=student, course=course)
        Enrollment.objects.get(user=self.students[0]).mark_completed()

    def test_only_admin(self):
        self.client.force_login(self.teacher)
        self.assertEqual(self.client.get(self.url).status_code, 404)
        self.client.force_login(self.students[0])
        self.assertEqual(self.client.get(self.url).status_code, 404)

    def test_counts_and_completion_rate(self):
        self.client.force_login(self.admin)
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        courses = response.context["courses"]
        self.assertEqual((courses["started"], courses["completed"], courses["rate"]), (4, 1, 25))
        row = response.context["course_page"].object_list[0]
        self.assertEqual((row.started, row.done, row.rate), (4, 1, 25))
        self.assertIn(("Student", 4), response.context["users"]["roles"])
        self.assertEqual(len(response.context["styles"]["rows"]), 4)
        self.assertContains(response, "Statystyki platformy")
        self.assertContains(response, 'aria-current="page">Statystyki')


class ThemeAccessibilityValidationTests(TestCase):
    """Panel admina nie zapisze motywu, w którym tekst nie osiąga kontrastu WCAG AA."""

    def form(self, **colors):
        from apps.core.forms import ColorThemeForm

        data = {"name": "Test", "primary": "#B45309", "secondary": "#0F766E", "accent": "#D97706",
                "surface": "#FFFBEB", "surface_variant": "#FEF3C7", "is_default": False}
        data.update(colors)
        return ColorThemeForm(data=data)

    def test_builtin_themes_pass(self):
        from apps.core.colors import theme_accessibility_problems
        from apps.core.models import ColorTheme

        for theme in ColorTheme.objects.all():
            with self.subTest(theme=theme.name):
                self.assertEqual(theme_accessibility_problems(theme.primary, theme.secondary, theme.accent, theme.surface, theme.surface_variant), [])
        self.assertTrue(self.form().is_valid(), self.form().errors)

    def test_mid_gray_background_rejected(self):
        form = self.form(surface="#77767B", surface_variant="#729FCF")
        self.assertFalse(form.is_valid())
        text = " ".join(form.non_field_errors())
        self.assertIn("Tekst pomocniczy", text)
        self.assertIn("Oba tła muszą być jasne albo oba ciemne", text)

    def test_low_contrast_brand_color_rejected(self):
        form = self.form(primary="#FFF4CC")
        self.assertFalse(form.is_valid())
        self.assertIn("Kolor główny użyty jako tekst", " ".join(form.non_field_errors()))
