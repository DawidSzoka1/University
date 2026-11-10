"""Testy w kursie: kreator, blokada za testem obowiązkowym, ocena ABC i ręczna, pliki, dyplomy."""

import io
import shutil
import tempfile

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse
from PIL import Image

from apps.catalog.models import Category, Course, Enrollment
from apps.core.models import Notification
from apps.learning.models import Attempt, Certificate, Choice, CourseSection, Lesson, LessonProgress, Question, Quiz

User = get_user_model()
MEDIA = tempfile.mkdtemp()


def tearDownModule():
    shutil.rmtree(MEDIA, ignore_errors=True)


def png_upload(name="szkic.png"):
    buffer = io.BytesIO()
    Image.new("RGB", (40, 30), (200, 100, 50)).save(buffer, "PNG")
    return SimpleUploadedFile(name, buffer.getvalue(), content_type="image/png")


@override_settings(MEDIA_ROOT=MEDIA)
class QuizFixture(TestCase):
    def setUp(self):
        self.teacher = User.objects.create_user(username="nauczyciel", password="Haslo12345!", role="moderator", first_name="Jan", last_name="Nowak")
        self.student = User.objects.create_user(username="uczen", password="Haslo12345!", first_name="Ala", last_name="Kot")
        self.other = User.objects.create_user(username="inny", password="Haslo12345!")
        category = Category.objects.create(name="Python")
        category.moderators.add(self.teacher)
        self.course = Course.objects.create(title="Python od zera", category=category, author=self.teacher)
        section = CourseSection.objects.create(course=self.course, title="Start", order=1)
        self.l1 = Lesson.objects.create(section=section, title="Lekcja 1", order=1)
        self.l2 = Lesson.objects.create(section=section, title="Lekcja 2", order=2)
        Enrollment.objects.create(user=self.student, course=self.course)

    def make_quiz(self, *, after=None, required=False, open_format=None, title="Sprawdzian", attempts=0):
        quiz = Quiz.objects.create(course=self.course, title=title, after_lesson=after, is_required=required, max_attempts=attempts)
        q = Question.objects.create(quiz=quiz, question_type="closed_single", text="2+2?", points=1, order=1)
        self.right = Choice.objects.create(question=q, text="4", is_correct=True, order=1)
        self.wrong = Choice.objects.create(question=q, text="5", is_correct=False, order=2)
        self.q_single = q
        if open_format:
            self.q_open = Question.objects.create(
                quiz=quiz, question_type="open", answer_format=open_format, text="Wyjaśnij", points=3, order=2
            )
        return quiz

    def quiz_url(self, quiz):
        return reverse("learning:quiz_detail", kwargs={"slug": self.course.slug, "pk": quiz.pk})


class QuizBuilderTests(QuizFixture):
    def test_author_creates_quiz_after_lesson(self):
        self.client.force_login(self.teacher)
        url = reverse("learning:quiz_create", kwargs={"slug": self.course.slug})
        self.assertContains(self.client.get(url + f"?po_lekcji={self.l1.pk}"), "Po lekcji 1.1 Lekcja 1")
        self.client.post(url, {"title": "Test 1", "after_lesson": self.l1.pk, "is_required": "on", "pass_percent": 70, "max_attempts": 2})
        quiz = Quiz.objects.get(title="Test 1")
        self.assertEqual(quiz.after_lesson, self.l1)
        self.assertTrue(quiz.is_required)

    def test_students_cannot_build(self):
        self.client.force_login(self.student)
        self.assertEqual(self.client.get(reverse("learning:quiz_create", kwargs={"slug": self.course.slug})).status_code, 404)

    def question_payload(self, **extra):
        data = {
            "question_type": "closed_single", "answer_format": "text", "text": "Stolica Polski?", "points": 2,
            "choices-TOTAL_FORMS": 3, "choices-INITIAL_FORMS": 0, "choices-MIN_NUM_FORMS": 0, "choices-MAX_NUM_FORMS": 8,
            "choices-0-text": "Warszawa", "choices-0-is_correct": "on",
            "choices-1-text": "Kraków",
            "choices-2-text": "",
        }
        data.update(extra)
        return data

    def test_question_with_choices_and_validation(self):
        quiz = Quiz.objects.create(course=self.course, title="T")
        self.client.force_login(self.teacher)
        url = reverse("learning:question_create", kwargs={"slug": self.course.slug, "quiz_pk": quiz.pk})
        response = self.client.post(url, self.question_payload(**{"choices-0-is_correct": ""}))
        self.assertContains(response, "Zaznacz, która odpowiedź jest poprawna.")
        response = self.client.post(url, self.question_payload(**{"choices-1-is_correct": "on"}))
        self.assertContains(response, "tylko jedna odpowiedź")
        self.client.post(url, self.question_payload())
        question = quiz.questions.get()
        self.assertEqual([c.text for c in question.choices.all()], ["Warszawa", "Kraków"])
        self.assertEqual(question.correct_choice_ids, {question.choices.get(text="Warszawa").pk})

    def test_open_question_with_image_requires_alt(self):
        quiz = Quiz.objects.create(course=self.course, title="T")
        self.client.force_login(self.teacher)
        url = reverse("learning:question_create", kwargs={"slug": self.course.slug, "quiz_pk": quiz.pk})
        data = self.question_payload(question_type="open", answer_format="code", code="print(1)", image=png_upload())
        response = self.client.post(url, data)
        self.assertContains(response, "Opisz zdjęcie")
        data = self.question_payload(question_type="open", answer_format="code", code="print(1)", image=png_upload(), image_alt="Wykres")
        self.client.post(url, data)
        question = quiz.questions.get()
        self.assertTrue(question.image)
        self.assertEqual(question.code, "print(1)")
        self.assertFalse(question.choices.exists())

    def test_quiz_delete_is_post_only(self):
        quiz = self.make_quiz()
        self.client.force_login(self.teacher)
        url = reverse("learning:quiz_delete", kwargs={"slug": self.course.slug, "pk": quiz.pk})
        self.assertEqual(self.client.get(url).status_code, 405)
        self.client.post(url)
        self.assertFalse(Quiz.objects.filter(pk=quiz.pk).exists())

    def test_curriculum_shows_quiz_after_lesson(self):
        self.make_quiz(after=self.l1, required=True)
        self.client.force_login(self.teacher)
        response = self.client.get(reverse("learning:curriculum_edit", kwargs={"slug": self.course.slug}))
        self.assertContains(response, "Test: </span>Sprawdzian")
        self.assertContains(response, "obowiązkowy")


class GatingTests(QuizFixture):
    def test_required_quiz_locks_following_lessons(self):
        quiz = self.make_quiz(after=self.l1, required=True)
        self.client.force_login(self.student)
        response = self.client.get(reverse("learning:lesson_detail", kwargs={"slug": self.course.slug, "pk": self.l2.pk}))
        self.assertRedirects(response, self.quiz_url(quiz))
        complete = reverse("learning:lesson_complete", kwargs={"slug": self.course.slug, "pk": self.l2.pk})
        self.assertEqual(self.client.post(complete).status_code, 404)

        self.client.post(self.quiz_url(quiz), {f"q{self.q_single.pk}": self.right.pk})
        response = self.client.get(reverse("learning:lesson_detail", kwargs={"slug": self.course.slug, "pk": self.l2.pk}))
        self.assertEqual(response.status_code, 200)

    def test_optional_quiz_does_not_lock(self):
        self.make_quiz(after=self.l1, required=False)
        self.client.force_login(self.student)
        response = self.client.get(reverse("learning:lesson_detail", kwargs={"slug": self.course.slug, "pk": self.l2.pk}))
        self.assertEqual(response.status_code, 200)

    def test_author_is_never_locked(self):
        self.make_quiz(after=self.l1, required=True)
        self.client.force_login(self.teacher)
        response = self.client.get(reverse("learning:lesson_detail", kwargs={"slug": self.course.slug, "pk": self.l2.pk}))
        self.assertEqual(response.status_code, 200)

    def test_lesson_next_button_leads_to_quiz(self):
        quiz = self.make_quiz(after=self.l1)
        self.client.force_login(self.student)
        response = self.client.get(reverse("learning:lesson_detail", kwargs={"slug": self.course.slug, "pk": self.l1.pk}))
        self.assertContains(response, self.quiz_url(quiz))


class AutoGradingTests(QuizFixture):
    def test_correct_and_wrong_answers(self):
        quiz = self.make_quiz()
        self.client.force_login(self.student)
        self.client.post(self.quiz_url(quiz), {f"q{self.q_single.pk}": self.wrong.pk})
        failed = Attempt.objects.get()
        self.assertEqual((failed.status, failed.passed, failed.score), ("graded", False, 0))
        self.client.post(self.quiz_url(quiz), {f"q{self.q_single.pk}": self.right.pk})
        passed = Attempt.objects.latest("pk")
        self.assertTrue(passed.passed)
        self.assertEqual(passed.score, 1)

    def test_multiple_choice_requires_exact_set(self):
        quiz = Quiz.objects.create(course=self.course, title="Wiele")
        q = Question.objects.create(quiz=quiz, question_type="closed_multiple", text="Parzyste?", points=2)
        a = Choice.objects.create(question=q, text="2", is_correct=True)
        b = Choice.objects.create(question=q, text="4", is_correct=True)
        c = Choice.objects.create(question=q, text="3", is_correct=False)
        self.client.force_login(self.student)
        self.client.post(self.quiz_url(quiz), {f"q{q.pk}": [a.pk]})
        self.assertEqual(Attempt.objects.latest("pk").score, 0)
        self.client.post(self.quiz_url(quiz), {f"q{q.pk}": [a.pk, b.pk, c.pk]})
        self.assertEqual(Attempt.objects.latest("pk").score, 0)
        self.client.post(self.quiz_url(quiz), {f"q{q.pk}": [a.pk, b.pk]})
        self.assertEqual(Attempt.objects.latest("pk").score, 2)

    def test_missing_answer_shows_error_and_keeps_choices(self):
        quiz = self.make_quiz(open_format="text")
        self.client.force_login(self.student)
        response = self.client.post(self.quiz_url(quiz), {f"q{self.q_single.pk}": self.right.pk})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Wpisz odpowiedź.")
        self.assertContains(response, f'value="{self.right.pk}" checked')
        self.assertFalse(Attempt.objects.exists())

    def test_attempt_limit_and_editor_preview(self):
        quiz = self.make_quiz(attempts=1)
        self.client.force_login(self.student)
        self.client.post(self.quiz_url(quiz), {f"q{self.q_single.pk}": self.wrong.pk})
        self.client.post(self.quiz_url(quiz), {f"q{self.q_single.pk}": self.right.pk})
        self.assertEqual(Attempt.objects.count(), 1)
        self.assertContains(self.client.get(self.quiz_url(quiz)), "Wykorzystano wszystkie podejścia")

        self.client.force_login(self.teacher)
        response = self.client.get(self.quiz_url(quiz))
        self.assertContains(response, "Podgląd autora")
        self.client.post(self.quiz_url(quiz), {f"q{self.q_single.pk}": self.right.pk})
        self.assertFalse(Attempt.objects.filter(user=self.teacher).exists())

    def test_not_enrolled_is_redirected(self):
        quiz = self.make_quiz()
        self.client.force_login(self.other)
        self.assertRedirects(self.client.get(self.quiz_url(quiz)), reverse("catalog:course_detail", kwargs={"slug": self.course.slug}))


class ManualGradingTests(QuizFixture):
    def submit_open(self, quiz, value):
        self.client.force_login(self.student)
        return self.client.post(self.quiz_url(quiz), {f"q{self.q_single.pk}": self.right.pk, f"q{self.q_open.pk}": value})

    def test_open_answer_waits_and_teacher_grades(self):
        quiz = self.make_quiz(open_format="text")
        self.submit_open(quiz, "Bo tak działa dodawanie.")
        attempt = Attempt.objects.get()
        self.assertEqual(attempt.status, "pending_review")
        self.assertEqual(attempt.score, 1)
        self.assertTrue(Notification.objects.filter(user=self.teacher, kind="quiz_to_grade").exists())

        grade_url = reverse("learning:attempt_grade", kwargs={"pk": attempt.pk})
        self.client.force_login(self.other)
        self.assertEqual(self.client.get(grade_url).status_code, 404)

        self.client.force_login(self.teacher)
        self.assertContains(self.client.get(reverse("learning:grading_queue")), grade_url)
        answer = attempt.answers.get(question=self.q_open)
        response = self.client.post(grade_url, {f"a{answer.pk}-points": 9, f"a{answer.pk}-feedback": ""})
        self.assertEqual(response.status_code, 200)  # więcej niż maksimum
        self.client.post(grade_url, {f"a{answer.pk}-points": 2, f"a{answer.pk}-feedback": "Dobrze"})
        attempt.refresh_from_db()
        self.assertEqual((attempt.status, attempt.score, attempt.passed), ("graded", 3, True))
        self.assertTrue(Notification.objects.filter(user=self.student, kind="quiz_graded").exists())
        self.assertNotContains(self.client.get(reverse("learning:grading_queue")), grade_url)

    def test_pending_required_quiz_keeps_lock(self):
        quiz = self.make_quiz(after=self.l1, required=True, open_format="text")
        self.submit_open(quiz, "Odpowiedź")
        response = self.client.get(reverse("learning:lesson_detail", kwargs={"slug": self.course.slug, "pk": self.l2.pk}))
        self.assertRedirects(response, self.quiz_url(quiz), fetch_redirect_response=False)

    def test_image_answer_is_stored_and_protected(self):
        quiz = self.make_quiz(open_format="image")
        self.submit_open(quiz, png_upload())
        answer = Attempt.objects.get().answers.get(question=self.q_open)
        self.assertTrue(answer.file)
        url = reverse("learning:answer_file", kwargs={"pk": answer.pk})
        self.assertEqual(self.client.get(url).status_code, 200)
        self.client.force_login(self.other)
        self.assertEqual(self.client.get(url).status_code, 404)
        self.client.force_login(self.teacher)
        self.assertEqual(self.client.get(url).status_code, 200)

    def test_wrong_file_type_is_rejected(self):
        quiz = self.make_quiz(open_format="file")
        response = self.submit_open(quiz, SimpleUploadedFile("wirus.exe", b"MZ"))
        self.assertContains(response, "niedozwolony typ")
        self.assertFalse(Attempt.objects.exists())


class CompletionAndCertificateTests(QuizFixture):
    def finish_course(self, quiz):
        self.client.force_login(self.student)
        for lesson in (self.l1, self.l2):
            self.client.post(reverse("learning:lesson_complete", kwargs={"slug": self.course.slug, "pk": lesson.pk}))
        self.client.post(self.quiz_url(quiz), {f"q{self.q_single.pk}": self.right.pk})

    def test_completion_requires_lessons_and_passed_quizzes(self):
        quiz = self.make_quiz()
        self.client.force_login(self.student)
        for lesson in (self.l1, self.l2):
            self.client.post(reverse("learning:lesson_complete", kwargs={"slug": self.course.slug, "pk": lesson.pk}))
        enrollment = Enrollment.objects.get(user=self.student)
        self.assertFalse(enrollment.is_completed)
        self.assertEqual(enrollment.progress_percent, 67)
        self.client.post(self.quiz_url(quiz), {f"q{self.q_single.pk}": self.right.pk})
        enrollment.refresh_from_db()
        self.assertTrue(enrollment.is_completed)
        certificate = Certificate.objects.get(user=self.student, course=self.course)
        self.assertEqual(certificate.full_name, "Ala Kot")
        self.assertEqual(certificate.teacher_name, "Jan Nowak")
        self.assertTrue(Notification.objects.filter(user=self.student, kind="course_completed").exists())

    def test_course_without_certificate(self):
        self.course.has_certificate = False
        self.course.save()
        self.finish_course(self.make_quiz())
        self.assertTrue(Enrollment.objects.get(user=self.student).is_completed)
        self.assertFalse(Certificate.objects.exists())

    def test_pdf_download_and_public_verification(self):
        self.finish_course(self.make_quiz())
        certificate = Certificate.objects.get()
        pdf_url = reverse("learning:certificate_download", kwargs={"code": certificate.code})
        response = self.client.get(pdf_url)
        self.assertEqual(response["Content-Type"], "application/pdf")
        self.assertTrue(response.content.startswith(b"%PDF"))
        self.client.force_login(self.other)
        self.assertEqual(self.client.get(pdf_url).status_code, 404)
        self.client.logout()
        verify = self.client.get(reverse("learning:certificate_verify", kwargs={"code": certificate.code}))
        self.assertContains(verify, "Dyplom jest ważny")
        self.assertContains(verify, "Ala Kot")
        self.assertEqual(self.client.get(reverse("learning:certificate_verify", kwargs={"code": "EL-0000-NIEMA"})).status_code, 404)

    def test_certificate_links_in_profile_and_course_page(self):
        self.finish_course(self.make_quiz())
        certificate = Certificate.objects.get()
        pdf_url = reverse("learning:certificate_download", kwargs={"code": certificate.code})
        self.assertContains(self.client.get(reverse("accounts:profile")), pdf_url)
        self.assertContains(self.client.get(reverse("catalog:course_detail", kwargs={"slug": self.course.slug})), pdf_url)

    def test_lesson_progress_reset_does_not_remove_certificate(self):
        self.finish_course(self.make_quiz())
        LessonProgress.objects.filter(lesson=self.l2).delete()
        self.assertTrue(Certificate.objects.exists())


class LearningModeTests(QuizFixture):
    """Autor/moderator/admin zapisany na kurs uczy się jak uczeń; podgląd autora to osobny tryb."""

    def setUp(self):
        super().setUp()
        self.admin = User.objects.create_user(username="admin2", password="Haslo12345!", role="admin", is_staff=True)
        Enrollment.objects.create(user=self.admin, course=self.course)  # np. kupił kurs
        self.l2_url = reverse("learning:lesson_detail", kwargs={"slug": self.course.slug, "pk": self.l2.pk})
        self.mode_url = reverse("learning:learning_mode", kwargs={"slug": self.course.slug})

    def test_enrolled_admin_takes_quiz_and_is_locked_like_student(self):
        quiz = self.make_quiz(after=self.l1, required=True)
        self.client.force_login(self.admin)
        page = self.client.get(self.quiz_url(quiz))
        self.assertContains(page, "Tryb nauki.")
        self.assertContains(page, "Przełącz na podgląd autora")
        self.assertRedirects(self.client.get(self.l2_url), self.quiz_url(quiz))  # blokada jak u ucznia
        response = self.client.post(self.quiz_url(quiz), {f"q{self.q_single.pk}": self.right.pk})
        attempt = Attempt.objects.get(user=self.admin)
        self.assertRedirects(response, reverse("learning:attempt_detail", kwargs={"pk": attempt.pk}))
        self.assertTrue(attempt.passed)
        self.assertEqual(self.client.get(self.l2_url).status_code, 200)

    def test_preview_mode_is_separate_and_does_not_submit(self):
        quiz = self.make_quiz(after=self.l1, required=True)
        self.client.force_login(self.admin)
        self.assertEqual(self.client.get(self.mode_url).status_code, 405)
        self.client.post(self.mode_url, {"mode": "preview", "next": self.l2_url})
        self.assertContains(self.client.get(self.l2_url), "Podgląd autora.")  # bez blokady
        self.client.post(self.quiz_url(quiz), {f"q{self.q_single.pk}": self.right.pk})
        self.assertFalse(Attempt.objects.exists())
        self.client.post(self.mode_url, {"mode": "learn"})
        self.assertContains(self.client.get(self.quiz_url(quiz)), "Tryb nauki.")

    def test_author_not_enrolled_only_previews_and_cannot_switch(self):
        quiz = self.make_quiz()
        self.client.force_login(self.teacher)
        page = self.client.get(self.quiz_url(quiz))
        self.assertContains(page, "zapisz się na kurs")
        self.assertNotContains(page, "Przełącz na podgląd autora")
        self.assertEqual(self.client.post(self.mode_url, {"mode": "learn"}).status_code, 404)
        self.client.force_login(self.student)
        self.assertEqual(self.client.post(self.mode_url, {"mode": "preview"}).status_code, 404)

    def test_nobody_grades_own_attempt(self):
        quiz = self.make_quiz(open_format="text")
        self.client.force_login(self.admin)
        self.client.post(self.quiz_url(quiz), {f"q{self.q_single.pk}": self.right.pk, f"q{self.q_open.pk}": "Moja odpowiedź"})
        attempt = Attempt.objects.get(user=self.admin)
        self.assertEqual(attempt.status, Attempt.Status.PENDING_REVIEW)
        grade_url = reverse("learning:attempt_grade", kwargs={"pk": attempt.pk})
        self.assertNotContains(self.client.get(reverse("learning:grading_queue")), grade_url)
        self.assertEqual(self.client.get(grade_url).status_code, 404)
        self.client.force_login(self.teacher)  # autor kursu ocenia
        self.assertContains(self.client.get(reverse("learning:grading_queue")), grade_url)
        self.assertEqual(self.client.get(grade_url).status_code, 200)
