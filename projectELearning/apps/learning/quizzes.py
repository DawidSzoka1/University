"""Testy: przyjmowanie odpowiedzi, sprawdzanie ABC, ocena ręczna, ukończenie kursu i dyplomy.

Pytania zamknięte (jedno- i wielokrotnego wyboru) liczą się automatycznie: punkty tylko za
dokładnie ten zestaw odpowiedzi, który nauczyciel oznaczył jako poprawny. Pytania otwarte
(tekst, kod, zdjęcie, plik) czekają na nauczyciela — podejście ma wtedy status „czeka na ocenę”
i nie zalicza testu, dopóki nie zostanie ocenione.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from datetime import timedelta

from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import transaction
from django.db.models import Q
from django.urls import reverse
from django.utils import timezone

from apps.catalog.models import Course
from apps.catalog.permissions import can_edit_course, course_reviewers, is_platform_admin
from apps.core.images import resize_image_field, validate_image_upload
from apps.core.models import Notification
from apps.core.notifications import notify, notify_many
from apps.learning.media import validate_attachment
from apps.learning.models import Answer, Attempt, Certificate, Question, Quiz, QuizSitting

ANSWER_IMAGE_SIZE = (1600, 1600)


# --------------------------------------------------------------------- uprawnienia
def can_grade(user, course: Course) -> bool:
    """Oceniają: autor kursu i nauczyciele jego kategorii (oraz admin)."""
    return can_edit_course(user, course)


def gradable_attempts(user):
    """Podejścia czekające na ocenę w kursach, które użytkownik może oceniać."""
    qs = Attempt.objects.filter(status=Attempt.Status.PENDING_REVIEW).select_related(
        "quiz", "quiz__course", "quiz__course__category", "user"
    )
    qs = qs.exclude(user=user)  # własnych podejść nie oceniamy
    if is_platform_admin(user):
        return qs
    from apps.catalog.comments import moderated_category_ids

    ids = moderated_category_ids(user) or []
    return qs.filter(Q(quiz__course__author=user) | Q(quiz__course__category_id__in=ids))


def attempts_left(quiz: Quiz, user) -> int | None:
    """Ile podejść zostało (None — bez limitu)."""
    if not quiz.max_attempts:
        return None
    return max(0, quiz.max_attempts - quiz.attempts.filter(user=user).count())


# ---------------------------------------------------------------- przyjmowanie
@dataclass
class Submission:
    """Odpowiedzi z formularza testu po walidacji. ``errors``: {question_id: komunikat}."""

    values: dict[int, object] = field(default_factory=dict)
    errors: dict[int, str] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return not self.errors


def read_submission(quiz: Quiz, data, files) -> Submission:
    """Czyta odpowiedzi ``q<id>`` z POST/FILES. Każde pytanie wymaga odpowiedzi."""
    result = Submission()
    for question in quiz.questions.prefetch_related("choices"):
        key = f"q{question.pk}"
        choice_ids = {c.pk for c in question.choices.all()}
        if question.question_type == Question.QuestionType.CLOSED_SINGLE:
            raw = data.get(key, "")
            if not raw.isdigit() or int(raw) not in choice_ids:
                result.errors[question.pk] = "Wybierz jedną odpowiedź."
            else:
                result.values[question.pk] = {int(raw)}
        elif question.question_type == Question.QuestionType.CLOSED_MULTIPLE:
            picked = {int(v) for v in data.getlist(key) if v.isdigit() and int(v) in choice_ids}
            if not picked:
                result.errors[question.pk] = "Zaznacz co najmniej jedną odpowiedź."
            else:
                result.values[question.pk] = picked
        elif question.answer_format in (Question.AnswerFormat.TEXT, Question.AnswerFormat.CODE):
            text = data.get(key, "").strip()
            if not text:
                result.errors[question.pk] = "Wpisz odpowiedź." if question.answer_format == "text" else "Wklej lub wpisz kod."
            elif len(text) > 20000:
                result.errors[question.pk] = "Odpowiedź może mieć najwyżej 20 000 znaków."
            else:
                result.values[question.pk] = text
        else:
            upload = files.get(key)
            if upload is None:
                result.errors[question.pk] = "Dodaj zdjęcie." if question.answer_format == "image" else "Dodaj plik."
                continue
            try:
                if question.answer_format == Question.AnswerFormat.IMAGE:
                    validate_image_upload(upload)
                else:
                    validate_attachment(upload)
            except ValidationError as error:
                result.errors[question.pk] = error.messages[0]
                continue
            result.values[question.pk] = upload
    return result


@transaction.atomic
def submit_attempt(quiz: Quiz, user, submission: Submission) -> Attempt:
    """Zapisuje podejście. Pytania zamknięte oceniane od razu, otwarte czekają na nauczyciela."""
    from apps.learning.styles import style_for_attempt

    attempt = Attempt.objects.create(
        quiz=quiz, user=user, max_score=quiz.max_score, learning_style=style_for_attempt(user, quiz)
    )
    for question in quiz.questions.prefetch_related("choices"):
        value = submission.values.get(question.pk)
        answer = Answer(attempt=attempt, question=question)
        if not question.is_open:
            answer.points = question.points if value == question.correct_choice_ids else 0
            answer.save()
            answer.selected_choices.set(value or [])
            continue
        if question.answer_format in (Question.AnswerFormat.TEXT, Question.AnswerFormat.CODE):
            answer.text_answer = value or ""
        elif value is not None:
            answer.original_name = value.name[:255]
            answer.file = value
            if question.answer_format == Question.AnswerFormat.IMAGE:
                resize_image_field(answer.file, ANSWER_IMAGE_SIZE)
        answer.save()

    QuizSitting.objects.filter(user=user, quiz=quiz).delete()
    if quiz.has_open_questions:
        _recount(attempt)
        attempt.save()
        _notify_graders(attempt)
    else:
        finalize_attempt(attempt, grader=None)
    return attempt


def start_sitting(quiz: Quiz, user) -> QuizSitting:
    """Uczeń otworzył test do rozwiązania — od teraz asystent AI odmawia pomocy."""
    sitting, _ = QuizSitting.objects.update_or_create(user=user, quiz=quiz, defaults={"started_at": timezone.now()})
    return sitting


def active_sitting(user) -> QuizSitting | None:
    """Najnowszy niewysłany test ucznia, otwarty nie dawniej niż ``CHATBOT_QUIZ_LOCK_MINUTES`` temu."""
    if not user.is_authenticated:
        return None
    since = timezone.now() - timedelta(minutes=settings.CHATBOT_QUIZ_LOCK_MINUTES)
    return (
        QuizSitting.objects.filter(user=user, started_at__gte=since)
        .select_related("quiz__course")
        .first()
    )


def _recount(attempt: Attempt) -> None:
    attempt.score = sum(a.points or 0 for a in attempt.answers.all())
    attempt.max_score = attempt.quiz.max_score


def finalize_attempt(attempt: Attempt, grader) -> None:
    """Wszystkie odpowiedzi ocenione → wynik, zaliczenie, powiadomienie i postęp kursu."""
    from apps.learning.curriculum import sync_enrollment_progress

    _recount(attempt)
    attempt.passed = attempt.percent >= attempt.quiz.pass_percent
    attempt.status = Attempt.Status.GRADED
    attempt.graded_at = timezone.now()
    attempt.graded_by = grader
    attempt.save()
    if grader is not None:
        result = "zaliczony" if attempt.passed else "niezaliczony"
        notify(
            attempt.user,
            Notification.Kind.QUIZ_GRADED,
            f"Test oceniony: {attempt.quiz.title}",
            f"Wynik {attempt.score}/{attempt.max_score} pkt ({attempt.percent}%) — {result}.",
            url=reverse("learning:attempt_detail", kwargs={"pk": attempt.pk}),
        )
    sync_enrollment_progress(attempt.user, attempt.quiz.course)


def _notify_graders(attempt: Attempt) -> None:
    course = attempt.quiz.course
    graders = list(course_reviewers(course))
    if course.author_id:
        graders.append(course.author)
    name = attempt.user.get_full_name() or attempt.user.username
    notify_many(
        graders,
        Notification.Kind.QUIZ_TO_GRADE,
        "Nowy test do sprawdzenia",
        f"{name}: „{attempt.quiz.title}” w kursie „{course.title}”.",
        url=reverse("learning:attempt_grade", kwargs={"pk": attempt.pk}),
        exclude=attempt.user,
    )


# --------------------------------------------------------------------- dyplomy
def _certificate_code() -> str:
    year = timezone.localdate().year
    while True:
        code = f"EL-{year}-{secrets.token_hex(4).upper()}"
        if not Certificate.objects.filter(code=code).exists():
            return code


def issue_certificate(user, course: Course) -> Certificate | None:
    """Wystawia dyplom (raz na kurs), jeśli kurs go przewiduje."""
    if not course.has_certificate:
        return None
    existing = Certificate.objects.filter(user=user, course=course).first()
    if existing:
        return existing
    teacher = course.author.get_full_name() or course.author.username if course.author_id else ""
    return Certificate.objects.create(
        user=user,
        course=course,
        code=_certificate_code(),
        full_name=user.get_full_name() or user.username,
        course_title=course.title,
        teacher_name=teacher,
    )


def on_course_completed(user, course: Course) -> None:
    certificate = issue_certificate(user, course)
    if certificate:
        notify(
            user,
            Notification.Kind.COURSE_COMPLETED,
            f"Ukończono kurs „{course.title}”",
            "Gratulacje! Dyplom w PDF czeka do pobrania w profilu i na stronie kursu.",
            url=reverse("learning:certificate_verify", kwargs={"code": certificate.code}),
        )
    else:
        notify(
            user,
            Notification.Kind.COURSE_COMPLETED,
            f"Ukończono kurs „{course.title}”",
            "Gratulacje! Wszystkie lekcje i testy zaliczone.",
            url=reverse("catalog:course_detail", kwargs={"slug": course.slug}),
        )
