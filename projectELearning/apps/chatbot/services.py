"""Logika asystenta niezależna od transportu (API JWT i formularz HTML korzystają z tego samego)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from django.conf import settings
from django.db.models import Q
from django.urls import reverse
from django.utils import timezone

from apps.catalog.models import Course
from apps.catalog.permissions import can_edit_course
from apps.chatbot.models import ChatMessage, ChatSession
from apps.learning.curriculum import build_outline, can_access_lesson
from apps.learning.models import Attempt, Lesson, QuizSitting
from apps.learning.quizzes import active_sitting

MAX_MESSAGE_LENGTH = 2000


class ContextError(Exception):
    """Wskazany kurs/lekcja/wynik nie istnieje albo uczeń nie ma do niego dostępu."""


@dataclass
class ChatContext:
    course: Course | None = None
    lesson: Lesson | None = None
    attempt: Attempt | None = None
    learning_style: object | None = None  # LearningStyle ucznia — prompt dopasowuje formę wyjaśnień

    @property
    def label(self) -> str:
        if self.attempt:
            return f"Wynik testu „{self.attempt.quiz.title}” · {self.course.title}"
        if self.lesson:
            return f"Lekcja „{self.lesson.title}” · {self.course.title}"
        if self.course:
            return f"Kurs „{self.course.title}”"
        return "Rozmowa ogólna"


def _int(value) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def can_discuss_course(user, course: Course) -> bool:
    return (course.is_published and course.status == Course.Status.APPROVED) or can_edit_course(user, course)


def resolve_context(user, course_id=None, lesson_id=None, attempt_id=None) -> ChatContext:
    """Sprawdza uprawnienia do materiału, o który pyta uczeń. Lekcja/wynik wyznaczają kurs."""
    attempt_id, lesson_id, course_id = _int(attempt_id), _int(lesson_id), _int(course_id)
    if attempt_id:
        attempt = (
            Attempt.objects.select_related("quiz__course", "quiz__after_lesson")
            .filter(pk=attempt_id, user=user)
            .first()
        )
        if attempt is None:
            raise ContextError("Nie znaleziono tego wyniku testu.")
        return ChatContext(course=attempt.quiz.course, attempt=attempt)
    if lesson_id:
        lesson = Lesson.objects.select_related("section__course__category").filter(pk=lesson_id).first()
        if lesson is None or not can_access_lesson(user, lesson):
            raise ContextError("Nie masz dostępu do tej lekcji.")
        course = lesson.section.course
        entry = build_outline(course, user).find(lesson)
        if entry is not None and not entry.accessible:
            raise ContextError("Ta lekcja jest jeszcze zablokowana.")
        return ChatContext(course=course, lesson=lesson)
    if course_id:
        course = Course.objects.select_related("category").filter(pk=course_id).first()
        if course is None or not can_discuss_course(user, course):
            raise ContextError("Nie znaleziono tego kursu.")
        return ChatContext(course=course)
    return ChatContext()


def get_session(user, course: Course | None) -> ChatSession:
    session, _ = ChatSession.objects.get_or_create(student=user, course=course)
    return session


def find_session(user, course: Course | None) -> ChatSession | None:
    return ChatSession.objects.filter(student=user, course=course).first()


# ---------------------------------------------------------------------------
# Blokada w trakcie testu
# ---------------------------------------------------------------------------

def lock_info(sitting: QuizSitting | None) -> dict | None:
    if sitting is None:
        return None
    quiz = sitting.quiz
    until = timezone.localtime(sitting.started_at + timedelta(minutes=settings.CHATBOT_QUIZ_LOCK_MINUTES))
    return {
        "quiz_title": quiz.title,
        "course_title": quiz.course.title,
        "quiz_url": reverse("learning:quiz_detail", kwargs={"slug": quiz.course.slug, "pk": quiz.pk}),
        "until": until.strftime("%H:%M"),
        "text": refusal_text(sitting),
        "notice": (
            f"W trakcie testu „{quiz.title}” asystent nie pomaga. Wyślij odpowiedzi — potem wyjaśni pytania. "
            f"Blokada wygaśnie najpóźniej o {until:%H:%M}."
        ),
    }


def refusal_text(sitting: QuizSitting) -> str:
    return (
        f"Trwa test „{sitting.quiz.title}”. Podczas testu nie mogę udzielać pomocy — "
        "rozwiąż go samodzielnie. Gdy wyślesz odpowiedzi, chętnie wyjaśnię pytania, "
        "z którymi był problem: otwórz wtedy wynik testu i zapytaj mnie o nie."
    )


def record_refusal(session: ChatSession, text: str, lesson: Lesson | None, sitting: QuizSitting) -> list[ChatMessage]:
    question = ChatMessage.objects.create(
        session=session, sender=ChatMessage.Sender.USER, message_text=text, lesson=lesson, is_refusal=True
    )
    answer = ChatMessage.objects.create(
        session=session, sender=ChatMessage.Sender.BOT, message_text=refusal_text(sitting), lesson=lesson, is_refusal=True
    )
    session.save(update_fields=["updated_at"])
    return [question, answer]


# ---------------------------------------------------------------------------
# Wiadomości
# ---------------------------------------------------------------------------

def is_busy(user) -> bool:
    """Uczeń czeka na odpowiedź na poprzednie pytanie (ostatnia wiadomość jest jego i jest świeża)."""
    last = (
        ChatMessage.objects.filter(session__student=user)
        .order_by("-timestamp", "-pk")
        .only("sender", "timestamp")
        .first()
    )
    if last is None or last.sender != ChatMessage.Sender.USER:
        return False
    return last.timestamp > timezone.now() - timedelta(seconds=settings.OLLAMA_TIMEOUT + 10)


def model_history(session: ChatSession, before: ChatMessage | None = None) -> list[ChatMessage]:
    """Ostatnie wiadomości rozmowy (bez odmów w trakcie testu), od najstarszej."""
    messages = session.messages.filter(is_refusal=False)
    if before is not None:
        messages = messages.filter(Q(timestamp__lt=before.timestamp) | Q(timestamp=before.timestamp, pk__lt=before.pk))
    latest = list(messages.order_by("-timestamp", "-pk")[: settings.CHATBOT_HISTORY_LIMIT])
    return latest[::-1]


def serialize_message(message: ChatMessage) -> dict:
    return {
        "id": message.pk,
        "sender": message.sender,
        "text": message.message_text,
        "html": str(message.html) if message.is_bot else "",
        "timestamp": timezone.localtime(message.timestamp).isoformat(),
        "time": timezone.localtime(message.timestamp).strftime("%H:%M"),
        "lesson": message.lesson.title if message.lesson_id else "",
        "is_refusal": message.is_refusal,
    }


def validate_text(text) -> str:
    text = (text or "").strip() if isinstance(text, str) else ""
    if not text:
        raise ValueError("Wpisz pytanie.")
    if len(text) > MAX_MESSAGE_LENGTH:
        raise ValueError(f"Pytanie może mieć najwyżej {MAX_MESSAGE_LENGTH} znaków (teraz {len(text)}).")
    return text
