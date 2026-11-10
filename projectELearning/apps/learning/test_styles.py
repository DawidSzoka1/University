"""Style uczenia się: słownik stylów, wybór przez ucznia, wersje lekcji, analiza i rekomendacja."""

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from apps.catalog.models import Category, Course, Enrollment
from apps.learning.models import (
    Attempt,
    Choice,
    CourseSection,
    LearningStyle,
    Lesson,
    LessonProgress,
    LessonVariant,
    Question,
    Quiz,
)
from apps.learning.styles import analyse, resolve_version, set_user_style, style_for_attempt

User = get_user_model()


class StyleFixture(TestCase):
    def setUp(self):
        self.teacher = User.objects.create_user(username="nauczyciel", password="Haslo12345!", role="moderator")
        self.student = User.objects.create_user(username="uczen", password="Haslo12345!")
        self.other = User.objects.create_user(username="inny", password="Haslo12345!")
        category = Category.objects.create(name="Python")
        category.moderators.add(self.teacher)
        self.course = Course.objects.create(title="Python od zera", category=category, author=self.teacher)
        section = CourseSection.objects.create(course=self.course, title="Start", order=1)
        self.lesson = Lesson.objects.create(section=section, title="Zmienne", order=1, content_html="<p>Treść podstawowa</p>")
        Enrollment.objects.create(user=self.student, course=self.course)
        self.visual = LearningStyle.objects.get(slug="wzrokowiec")
        self.practical = LearningStyle.objects.get(slug="praktyk")
        self.auditory = LearningStyle.objects.get(slug="sluchowiec")

    def add_variant(self, style, html="<p>Wersja</p>", lesson=None):
        lesson = lesson or self.lesson
        lesson.same_for_all_styles = False
        lesson.save()
        return LessonVariant.objects.create(lesson=lesson, style=style, content_html=html)

    def lesson_url(self, **query):
        url = reverse("learning:lesson_detail", kwargs={"slug": self.course.slug, "pk": self.lesson.pk})
        return url + ("?" + "&".join(f"{k}={v}" for k, v in query.items()) if query else "")

    def graded_attempt(self, style, percent, passed=None):
        quiz = Quiz.objects.create(course=self.course, title=f"Test {Attempt.objects.count()}")
        return Attempt.objects.create(
            quiz=quiz, user=self.student, status=Attempt.Status.GRADED, score=percent, max_score=100,
            passed=percent >= 60 if passed is None else passed, learning_style=style,
        )


class StyleDictionaryTests(StyleFixture):
    def test_four_default_styles_from_migration(self):
        self.assertEqual(
            list(LearningStyle.objects.filter(is_active=True).values_list("slug", flat=True)),
            ["wzrokowiec", "sluchowiec", "czytanie-pisanie", "praktyk"],
        )

    def test_admin_can_add_style_with_polish_name(self):
        style = LearningStyle.objects.create(name="Społeczny", description="Uczysz się w grupie.", order=5)
        self.assertEqual(style.slug, "spoleczny")
        self.assertIn(style, LearningStyle.objects.filter(is_active=True))

    def test_admin_pages_for_styles(self):
        admin = User.objects.create_superuser(username="szef", password="Haslo12345!", email="a@example.com", role="admin")
        self.client.force_login(admin)
        self.assertEqual(self.client.get(reverse("admin:learning_learningstyle_changelist")).status_code, 200)
        self.assertEqual(self.client.get(reverse("admin:learning_learningstyle_add")).status_code, 200)


class ChooseStyleTests(StyleFixture):
    url = reverse("learning:learning_style")

    def test_requires_login(self):
        self.assertEqual(self.client.get(self.url).status_code, 302)

    def test_page_lists_styles_with_labels_in_fieldset(self):
        self.client.force_login(self.student)
        response = self.client.get(self.url)
        self.assertContains(response, "<fieldset class=\"style-options\">", html=False)
        for style in LearningStyle.objects.filter(is_active=True):
            self.assertContains(response, style.description)
        self.assertContains(response, 'for="id_style_1"')

    def test_student_picks_and_changes_style_any_time(self):
        self.client.force_login(self.student)
        self.client.post(self.url, {"style": self.visual.pk})
        self.student.refresh_from_db()
        self.assertEqual(self.student.learning_style, self.visual)
        self.assertIsNotNone(self.student.learning_style_updated_at)
        self.client.post(self.url, {"style": self.practical.pk})
        self.student.refresh_from_db()
        self.assertEqual(self.student.learning_style, self.practical)
        self.client.post(self.url, {"style": ""})
        self.student.refresh_from_db()
        self.assertIsNone(self.student.learning_style)

    def test_inactive_style_cannot_be_chosen(self):
        self.visual.is_active = False
        self.visual.save()
        self.client.force_login(self.student)
        response = self.client.post(self.url, {"style": self.visual.pk})
        self.assertEqual(response.status_code, 200)
        self.student.refresh_from_db()
        self.assertIsNone(self.student.learning_style)

    def test_next_redirect_only_internal(self):
        self.client.force_login(self.student)
        response = self.client.post(self.url, {"style": self.visual.pk, "next": "/catalog/"})
        self.assertRedirects(response, "/catalog/", fetch_redirect_response=False)
        response = self.client.post(self.url, {"style": self.visual.pk, "next": "https://zly.example.com/"})
        self.assertRedirects(response, self.url, fetch_redirect_response=False)

    def test_profile_and_menu_link_to_style_page(self):
        self.client.force_login(self.student)
        response = self.client.get(reverse("accounts:profile"))
        self.assertContains(response, self.url)
        self.assertContains(response, "Styl uczenia się")


class LessonVersionTests(StyleFixture):
    def test_same_content_for_everyone_ignores_variants(self):
        LessonVariant.objects.create(lesson=self.lesson, style=self.visual, content_html="<p>Wizualna</p>")
        set_user_style(self.student, self.visual)
        version = resolve_version(self.lesson, self.student)
        self.assertIsNone(version.variant)
        self.assertFalse(version.has_variants)

    def test_student_sees_variant_for_own_style(self):
        self.add_variant(self.visual, "<p>Wizualna</p>")
        set_user_style(self.student, self.visual)
        self.client.force_login(self.student)
        response = self.client.get(self.lesson_url())
        self.assertContains(response, "Wizualna")
        self.assertNotContains(response, "Treść podstawowa")
        self.assertContains(response, 'aria-label="Wersja lekcji dla stylu uczenia się"')
        self.assertContains(response, "(Twój styl)")

    def test_style_without_variant_gets_base_content(self):
        self.add_variant(self.visual)
        set_user_style(self.student, self.practical)
        self.client.force_login(self.student)
        response = self.client.get(self.lesson_url())
        self.assertContains(response, "Treść podstawowa")
        self.assertContains(response, "nie ma dla niego osobnej wersji")

    def test_switch_to_other_version_and_back_to_base(self):
        self.add_variant(self.visual, "<p>Wizualna</p>")
        self.add_variant(self.practical, "<p>Praktyczna</p>")
        set_user_style(self.student, self.visual)
        self.client.force_login(self.student)
        self.assertContains(self.client.get(self.lesson_url(wersja="praktyk")), "Praktyczna")
        self.assertContains(self.client.get(self.lesson_url(wersja="podstawowa")), "Treść podstawowa")
        self.assertContains(self.client.get(self.lesson_url(wersja="nieznany")), "Wizualna")

    def test_inactive_style_variant_hidden(self):
        self.add_variant(self.visual, "<p>Wizualna</p>")
        self.visual.is_active = False
        self.visual.save()
        set_user_style(self.student, self.visual)
        self.assertIsNone(resolve_version(self.lesson, self.student).variant)

    def test_variant_video_replaces_lesson_video(self):
        self.lesson.video_url = "https://www.youtube.com/watch?v=aaaaaaaaaaa"
        variant = self.add_variant(self.visual)
        variant.video_url = "https://www.youtube.com/watch?v=bbbbbbbbbbb"
        variant.save()
        set_user_style(self.student, self.visual)
        self.client.force_login(self.student)
        response = self.client.get(self.lesson_url())
        self.assertContains(response, "bbbbbbbbbbb")
        self.assertNotContains(response, "aaaaaaaaaaa")

    def test_variant_html_is_sanitized(self):
        variant = self.add_variant(self.visual, '<p>ok</p><script>alert(1)</script><img src=x onerror="alert(2)">')
        self.assertNotIn("<script", variant.content_html)
        self.assertNotIn("onerror", variant.content_html)

    def test_completion_records_displayed_version(self):
        self.add_variant(self.visual)
        self.add_variant(self.practical)
        set_user_style(self.student, self.visual)
        self.client.force_login(self.student)
        url = reverse("learning:lesson_complete", kwargs={"slug": self.course.slug, "pk": self.lesson.pk})
        self.client.post(url)
        self.assertEqual(LessonProgress.objects.get(user=self.student).style, self.visual)
        self.client.post(url)  # cofnięcie
        response = self.client.post(url, {"wersja": "praktyk"})
        self.assertEqual(LessonProgress.objects.get(user=self.student).style, self.practical)
        self.assertIn("wersja=praktyk", response["Location"])

    def test_completion_of_base_content_has_no_style(self):
        set_user_style(self.student, self.visual)
        self.client.force_login(self.student)
        self.client.post(reverse("learning:lesson_complete", kwargs={"slug": self.course.slug, "pk": self.lesson.pk}))
        self.assertIsNone(LessonProgress.objects.get(user=self.student).style)


class AuthorVariantTests(StyleFixture):
    def variants_url(self):
        return reverse("learning:lesson_variants", kwargs={"slug": self.course.slug, "pk": self.lesson.pk})

    def edit_url(self, style):
        return reverse("learning:lesson_variant_edit", kwargs={"slug": self.course.slug, "pk": self.lesson.pk, "style_slug": style.slug})

    def test_only_course_editors_manage_variants(self):
        self.client.force_login(self.other)
        self.assertEqual(self.client.get(self.variants_url()).status_code, 404)
        self.assertEqual(self.client.get(self.edit_url(self.visual)).status_code, 404)
        self.client.force_login(self.teacher)
        response = self.client.get(self.variants_url())
        self.assertContains(response, "Dodaj wersję")
        self.assertContains(response, self.visual.author_hint)

    def test_new_variant_starts_from_base_content_and_enables_variants(self):
        self.client.force_login(self.teacher)
        response = self.client.get(self.edit_url(self.visual))
        self.assertContains(response, "Treść podstawowa")
        self.assertContains(response, f'for="id_content_html"')
        response = self.client.post(self.edit_url(self.visual), {"content_html": "<p>Schemat</p>", "video_url": ""})
        self.assertRedirects(response, self.variants_url(), fetch_redirect_response=False)
        self.lesson.refresh_from_db()
        self.assertFalse(self.lesson.same_for_all_styles)
        self.assertEqual(LessonVariant.objects.get(lesson=self.lesson, style=self.visual).content_html, "<p>Schemat</p>")

    def test_variant_requires_content_and_valid_video(self):
        self.client.force_login(self.teacher)
        response = self.client.post(self.edit_url(self.visual), {"content_html": "<p> </p>", "video_url": "https://example.com/film"})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Wpisz treść tej wersji lekcji.")
        self.assertContains(response, "YouTube")
        self.assertFalse(LessonVariant.objects.exists())

    def test_toggle_and_delete_are_post_only(self):
        variant = self.add_variant(self.visual)
        self.client.force_login(self.teacher)
        toggle = reverse("learning:lesson_styles_toggle", kwargs={"slug": self.course.slug, "pk": self.lesson.pk})
        delete = reverse("learning:lesson_variant_delete", kwargs={"slug": self.course.slug, "pk": self.lesson.pk, "style_slug": "wzrokowiec"})
        self.assertEqual(self.client.get(toggle).status_code, 405)
        self.assertEqual(self.client.get(delete).status_code, 405)
        self.client.post(toggle, {"same": "1"})
        self.lesson.refresh_from_db()
        self.assertTrue(self.lesson.same_for_all_styles)
        self.assertTrue(LessonVariant.objects.filter(pk=variant.pk).exists())  # wersja zachowana
        self.client.post(delete)
        self.assertFalse(LessonVariant.objects.filter(pk=variant.pk).exists())

    def test_lesson_form_and_curriculum_offer_styles(self):
        self.add_variant(self.visual)
        self.client.force_login(self.teacher)
        form = self.client.get(reverse("learning:lesson_edit", kwargs={"slug": self.course.slug, "pk": self.lesson.pk}))
        self.assertContains(form, 'for="id_same_for_all_styles"')
        self.assertContains(form, self.variants_url())
        program = self.client.get(reverse("learning:curriculum_edit", kwargs={"slug": self.course.slug}))
        self.assertContains(program, "1 wersja dla stylów")
        self.assertContains(program, self.variants_url())


class AnalysisTests(StyleFixture):
    def test_attempt_gets_style_of_lessons_done_in_course(self):
        LessonProgress.objects.create(user=self.student, lesson=self.lesson, style=self.practical)
        set_user_style(self.student, self.visual)
        quiz = Quiz.objects.create(course=self.course, title="Sprawdzian")
        self.assertEqual(style_for_attempt(self.student, quiz), self.practical)

    def test_attempt_without_styled_lessons_uses_current_style(self):
        set_user_style(self.student, self.visual)
        self.assertEqual(style_for_attempt(self.student, Quiz.objects.create(course=self.course, title="T")), self.visual)

    def test_submitted_attempt_stores_style(self):
        set_user_style(self.student, self.auditory)
        quiz = Quiz.objects.create(course=self.course, title="Sprawdzian")
        question = Question.objects.create(quiz=quiz, question_type="closed_single", text="2+2?", points=1, order=1)
        right = Choice.objects.create(question=question, text="4", is_correct=True, order=1)
        Choice.objects.create(question=question, text="5", is_correct=False, order=2)
        self.client.force_login(self.student)
        self.client.post(reverse("learning:quiz_detail", kwargs={"slug": self.course.slug, "pk": quiz.pk}), {f"q{question.pk}": right.pk})
        self.assertEqual(Attempt.objects.get(user=self.student).learning_style, self.auditory)

    def test_no_data(self):
        result = analyse(self.student)
        self.assertEqual(result.kind, "no_data")
        self.assertIsNone(result.suggested)

    def test_single_attempt_is_not_enough(self):
        set_user_style(self.student, self.visual)
        self.graded_attempt(self.practical, 100)
        result = analyse(self.student)
        self.assertEqual(result.kind, "no_data")
        self.assertIsNotNone(result.explore)

    def test_recommends_better_style(self):
        set_user_style(self.student, self.auditory)
        self.graded_attempt(self.practical, 100)
        self.graded_attempt(self.practical, 90)
        self.graded_attempt(self.auditory, 30)
        self.graded_attempt(self.auditory, 40)
        result = analyse(self.student)
        self.assertEqual(result.kind, "switch")
        self.assertEqual(result.suggested, self.practical)
        self.assertTrue(result.differs)

    def test_keeps_style_when_difference_is_small(self):
        set_user_style(self.student, self.auditory)
        self.graded_attempt(self.practical, 82)
        self.graded_attempt(self.practical, 82)
        self.graded_attempt(self.auditory, 80)
        self.graded_attempt(self.auditory, 80)
        result = analyse(self.student)
        self.assertEqual(result.kind, "keep")
        self.assertFalse(result.differs)

    def test_suggests_choosing_when_no_style_selected(self):
        self.graded_attempt(self.visual, 90)
        self.graded_attempt(self.visual, 95)
        result = analyse(self.student)
        self.assertEqual(result.kind, "choose")
        self.assertEqual(result.suggested, self.visual)

    def test_pending_attempts_are_ignored(self):
        attempt = self.graded_attempt(self.practical, 100)
        attempt.status = Attempt.Status.PENDING_REVIEW
        attempt.save()
        self.assertEqual(analyse(self.student).graded_attempts, 0)

    def test_style_page_and_lesson_show_recommendation(self):
        set_user_style(self.student, self.auditory)
        for percent in (100, 95):
            self.graded_attempt(self.practical, percent)
        for percent in (20, 30):
            self.graded_attempt(self.auditory, percent)
        self.client.force_login(self.student)
        page = self.client.get(reverse("learning:learning_style"))
        self.assertContains(page, "Polecamy styl „Praktyk”")
        self.assertContains(page, "polecany dla Ciebie")
        lesson = self.client.get(self.lesson_url())
        self.assertContains(lesson, "polecamy styl")


class AssistantStyleTests(StyleFixture):
    def test_prompt_mentions_student_style(self):
        from apps.chatbot import llm
        from apps.chatbot.services import ChatContext

        context = ChatContext(course=self.course, lesson=self.lesson, learning_style=self.visual)
        prompt = llm.system_prompt(context)
        self.assertIn("Styl uczenia się ucznia: Wzrokowiec", prompt)
        self.assertNotIn("Styl uczenia się ucznia", llm.system_prompt(ChatContext(course=self.course)))
