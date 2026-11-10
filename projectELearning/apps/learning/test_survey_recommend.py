"""Ankieta stylu, polecanie kursów, panel uczniów kursu i liczby ukończeń."""

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from apps.catalog.models import Category, Course, Enrollment
from apps.learning.models import (
    Attempt,
    CourseSection,
    LearningStyle,
    Lesson,
    LessonProgress,
    LessonVariant,
    Quiz,
    SurveyQuestion,
    SurveyResponse,
)
from apps.learning.styles import learner_profile, rank_courses, recommend_courses, set_user_style

User = get_user_model()


def answer_all(questions, slug):
    """POST z odpowiedzią wskazującą ``slug`` w każdym pytaniu."""
    data = {}
    for question in questions:
        option = next(o for o in question.options.all() if o.style.slug == slug)
        data[f"q{question.pk}"] = option.pk
    return data


class Fixture(TestCase):
    def setUp(self):
        self.teacher = User.objects.create_user(username="nauczyciel", password="Haslo12345!", role="moderator")
        self.student = User.objects.create_user(username="uczen", password="Haslo12345!", first_name="Ala")
        category = Category.objects.create(name="IT")
        category.moderators.add(self.teacher)
        self.visual = LearningStyle.objects.get(slug="wzrokowiec")
        self.practical = LearningStyle.objects.get(slug="praktyk")
        self.courses = {}
        for title in ("Kurs praktyczny", "Kurs wizualny", "Kurs zwykły"):
            course = Course.objects.create(title=title, category=category, author=self.teacher)
            section = CourseSection.objects.create(course=course, title="S", order=1)
            course.lessons_list = [
                Lesson.objects.create(section=section, title=f"{title} {i}", order=i, same_for_all_styles=False) for i in (1, 2)
            ]
            self.courses[title] = course
        for lesson in self.courses["Kurs praktyczny"].lessons_list:
            LessonVariant.objects.create(lesson=lesson, style=self.practical, content_html="<p>p</p>")
        self.courses["Kurs wizualny"].recommended_styles.add(self.visual)
        self.questions = list(SurveyQuestion.objects.prefetch_related("options__style"))


class SurveyTests(Fixture):
    url = reverse("learning:style_survey")

    def test_ten_questions_with_option_per_style(self):
        self.assertEqual(len(self.questions), 10)
        for question in self.questions:
            self.assertEqual({o.style.slug for o in question.options.all()}, {"wzrokowiec", "sluchowiec", "czytanie-pisanie", "praktyk"})

    def test_requires_login_and_renders_fieldsets(self):
        self.assertEqual(self.client.get(self.url).status_code, 302)
        self.client.force_login(self.student)
        response = self.client.get(self.url)
        self.assertContains(response, '<fieldset class="survey-q', count=10)
        self.assertContains(response, "<legend")

    def test_missing_answers_keep_selected_and_show_errors(self):
        self.client.force_login(self.student)
        data = answer_all(self.questions, "praktyk")
        missing = self.questions[3]
        data.pop(f"q{missing.pk}")
        response = self.client.post(self.url, data)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, f'id="q{missing.pk}-error"')
        self.assertContains(response, "checked", count=9)
        self.assertFalse(SurveyResponse.objects.exists())

    def test_result_sets_style_when_none_chosen(self):
        self.client.force_login(self.student)
        response = self.client.post(self.url, answer_all(self.questions, "praktyk"))
        self.assertRedirects(response, reverse("learning:learning_style") + "#ankieta", fetch_redirect_response=False)
        result = SurveyResponse.objects.get(user=self.student)
        self.assertEqual(result.top_style, self.practical)
        self.assertEqual(result.answers.count(), 10)
        self.student.refresh_from_db()
        self.assertEqual(self.student.learning_style, self.practical)
        page = self.client.get(reverse("learning:learning_style"))
        self.assertContains(page, "Praktyk: 100% odpowiedzi")

    def test_existing_style_is_not_overwritten_but_can_be_switched(self):
        set_user_style(self.student, self.visual)
        self.client.force_login(self.student)
        self.client.post(self.url, answer_all(self.questions, "praktyk"))
        self.student.refresh_from_db()
        self.assertEqual(self.student.learning_style, self.visual)
        page = self.client.get(reverse("learning:learning_style"))
        self.assertContains(page, "Ustaw „Praktyk” jako mój styl")

    def test_foreign_option_is_rejected(self):
        self.client.force_login(self.student)
        data = answer_all(self.questions, "praktyk")
        other_option = self.questions[1].options.first()
        data[f"q{self.questions[0].pk}"] = other_option.pk
        self.client.post(self.url, data)
        self.assertFalse(SurveyResponse.objects.exists())


class RecommendationTests(Fixture):
    def test_profile_combines_chosen_style_survey_and_results(self):
        self.assertTrue(learner_profile(self.student).is_empty)
        set_user_style(self.student, self.visual)
        profile = learner_profile(self.student)
        self.assertEqual(profile.weights, {self.visual.pk: 1.0})
        self.client.force_login(self.student)
        self.client.post(reverse("learning:style_survey"), answer_all(self.questions, "praktyk"))
        profile = learner_profile(self.student)
        self.assertAlmostEqual(profile.weights[self.visual.pk], 0.5)
        self.assertAlmostEqual(profile.weights[self.practical.pk], 0.5)
        self.assertIn("wynik ankiety", profile.sources)

    def test_ranking_uses_variants_and_author_choice(self):
        set_user_style(self.student, self.practical)
        ranked = rank_courses(self.student, self.courses.values())
        self.assertEqual([f.course.title for f in ranked], ["Kurs praktyczny"])
        self.assertEqual(ranked[0].reason, "2 z 2 lekcji ma wersję dla stylu „Praktyk”")
        set_user_style(self.student, self.visual)
        ranked = rank_courses(self.student, self.courses.values())
        self.assertEqual([f.course.title for f in ranked], ["Kurs wizualny"])
        self.assertIn("Autor poleca", ranked[0].reason)

    def test_enrolled_and_unpublished_courses_not_recommended(self):
        set_user_style(self.student, self.practical)
        for course in self.courses.values():
            course.status, course.is_published = "approved", True
            course.save()
        self.assertEqual([f.course.title for f in recommend_courses(self.student)], ["Kurs praktyczny"])
        Enrollment.objects.create(user=self.student, course=self.courses["Kurs praktyczny"])
        self.assertEqual(recommend_courses(self.student), [])
        Course.objects.filter(title="Kurs wizualny").update(status="pending")
        set_user_style(self.student, self.visual)
        self.assertEqual(recommend_courses(self.student), [])

    def test_catalog_section_and_sort(self):
        for course in self.courses.values():
            course.status, course.is_published = "approved", True
            course.save()
        set_user_style(self.student, self.practical)
        self.client.force_login(self.student)
        catalog = self.client.get(reverse("catalog:course_list"))
        self.assertContains(catalog, "Polecane dla Ciebie")
        self.assertContains(catalog, "Dopasowane do mnie")
        self.assertContains(catalog, "Dla Ciebie:")
        sorted_page = self.client.get(reverse("catalog:course_list") + "?sort=dopasowane")
        titles = [c.title for c in sorted_page.context["courses"]]
        self.assertEqual(titles[0], "Kurs praktyczny")
        self.assertEqual(len(titles), 3)  # pozostałe kursy dalej na liście
        self.client.logout()
        self.assertNotContains(self.client.get(reverse("catalog:course_list")), "Polecane dla Ciebie")

    def test_style_page_and_my_courses_show_recommendations(self):
        for course in self.courses.values():
            course.status, course.is_published = "approved", True
            course.save()
        set_user_style(self.student, self.practical)
        self.client.force_login(self.student)
        self.assertContains(self.client.get(reverse("learning:learning_style")), "Kurs praktyczny")
        self.assertContains(self.client.get(reverse("learning:my_courses")), "Kurs praktyczny")

    def test_author_marks_course_styles_in_form(self):
        self.client.force_login(self.teacher)
        course = self.courses["Kurs zwykły"]
        url = reverse("catalog:course_edit", kwargs={"slug": course.slug})
        self.assertContains(self.client.get(url), "Kurs szczególnie dla stylów")


class CourseStudentsTests(Fixture):
    def setUp(self):
        super().setUp()
        self.course = self.courses["Kurs praktyczny"]
        self.url = reverse("learning:course_students", kwargs={"slug": self.course.slug})
        self.other = User.objects.create_user(username="drugi", password="Haslo12345!")
        Enrollment.objects.create(user=self.student, course=self.course)
        done = Enrollment.objects.create(user=self.other, course=self.course)
        done.mark_completed()
        LessonProgress.objects.create(user=self.student, lesson=self.course.lessons_list[0], style=self.practical)
        quiz = Quiz.objects.create(course=self.course, title="T")
        Attempt.objects.create(quiz=quiz, user=self.student, status="graded", score=8, max_score=10, passed=True)
        set_user_style(self.student, self.practical)

    def test_only_course_editors(self):
        self.client.force_login(self.student)
        self.assertEqual(self.client.get(self.url).status_code, 404)

    def test_summary_and_rows(self):
        self.client.force_login(self.teacher)
        response = self.client.get(self.url)
        self.assertEqual(response.context["started"], 2)
        self.assertEqual(response.context["completed"], 1)
        self.assertEqual(response.context["completion_rate"], 50)
        self.assertEqual(response.context["average_score"], 80)
        row = next(r for r in response.context["rows"] if r.user == self.student)
        self.assertEqual((row.lessons_done, row.quizzes_passed, row.average_score), (1, 1, 80))
        self.assertContains(response, "Praktyk: 1")
        self.assertContains(response, "<caption")

    def test_filter_completed(self):
        self.client.force_login(self.teacher)
        response = self.client.get(self.url + "?pokaz=ukonczyli")
        self.assertEqual([r.user for r in response.context["rows"]], [self.other])

    def test_course_detail_shows_started_completed_and_versions(self):
        Course.objects.filter(pk=self.course.pk).update(status="approved", is_published=True)
        response = self.client.get(reverse("catalog:course_detail", kwargs={"slug": self.course.slug}))
        self.assertContains(response, "2 osoby")
        self.assertContains(response, "1 (50%)")
        self.assertContains(response, "Praktyk: 2 z 2 lekcji")

    def test_uncompleting_updates_course_count(self):
        from apps.learning.curriculum import sync_enrollment_progress

        self.course.refresh_from_db()
        self.assertEqual(self.course.completed_count, 1)
        sync_enrollment_progress(self.other, self.course)  # brak ukończonych lekcji → nieukończony
        self.course.refresh_from_db()
        self.assertEqual(self.course.completed_count, 0)
