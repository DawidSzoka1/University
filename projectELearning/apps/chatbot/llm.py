"""Rozmowa z Bielikiem przez Ollamę: prompt systemowy z materiałami kursu i strumień odpowiedzi."""

from __future__ import annotations

import html
import re
from collections.abc import Iterator

import httpx
import ollama
from django.conf import settings
from django.utils.html import strip_tags

from apps.chatbot.models import ChatMessage
from apps.chatbot.services import ChatContext
from apps.learning.models import Attempt
from apps.learning.quizzes import attempts_left

LESSON_CHARS = 6000
DESCRIPTION_CHARS = 1500

BASE_PROMPT = """Jesteś Bielikiem — asystentem nauki na polskiej platformie kursów online.
Zasady:
- Odpowiadasz wyłącznie po polsku, poprawną polszczyzną, zwięźle i rzeczowo (zwykle do 200 słów).
- Pomagasz zrozumieć materiał kursu: tłumaczysz innymi słowami, podajesz krótkie przykłady, zadajesz pytania sprawdzające.
- Opierasz się na materiałach kursu podanych niżej. Jeśli czegoś w nich nie ma, powiedz to wprost i nie zmyślaj.
- Nie rozwiązujesz za ucznia testów ani zadań z kursu. Gdy prosi o gotową odpowiedź do testu, wyjaśnij zagadnienie i naprowadź go, ale nie wskazuj poprawnej odpowiedzi.
- Kod zapisuj w blokach ``` z nazwą języka. Nie używaj HTML.
- Pytania niezwiązane z nauką zbywaj krótko i uprzejmie, wracając do kursu."""


def model_label(name: str | None = None) -> str:
    """Czytelna nazwa modelu dla ucznia, np. ``hf.co/speakleash/Bielik-Minitron-7B-v3.0-Instruct-GGUF:Q4_K_M``
    → ``Bielik-Minitron-7B-v3.0-Instruct (Q4_K_M)``. ``OLLAMA_MODEL_LABEL`` nadpisuje wynik."""
    if name is None:
        override = getattr(settings, "OLLAMA_MODEL_LABEL", "")
        if override:
            return override
        name = settings.OLLAMA_MODEL
    base, _, tag = name.rpartition("/")[2].partition(":")
    base = re.sub(r"[-_.]gguf$", "", base, flags=re.I)
    return f"{base} ({tag})" if tag and tag != "latest" else base


class AssistantUnavailable(Exception):
    """Ollama nie działa, nie ma modelu albo przekroczono czas — komunikat dla ucznia po polsku."""


def plain_text(value: str, limit: int) -> str:
    text = html.unescape(strip_tags(value or ""))
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " […]"


def _course_block(context: ChatContext) -> str:
    course = context.course
    lines = [f"KURS: {course.title}"]
    if course.category_id:
        lines.append(f"Kategoria: {course.category.name}")
    about = plain_text(course.description_html or course.description, DESCRIPTION_CHARS)
    if about:
        lines.append(f"O kursie: {about}")
    program = []
    for section in course.sections.prefetch_related("lessons").order_by("order"):
        titles = ", ".join(lesson.title for lesson in section.lessons.all())
        program.append(f"- {section.title}: {titles}" if titles else f"- {section.title}")
    if program:
        lines.append("Program:\n" + "\n".join(program))
    return "\n".join(lines)


def _lesson_block(context: ChatContext) -> str:
    lesson = context.lesson
    content = plain_text(lesson.content_html, LESSON_CHARS) or "(lekcja bez treści tekstowej — np. sam film)"
    return f"UCZEŃ CZYTA TERAZ LEKCJĘ: {lesson.title} (sekcja „{lesson.section.title}”)\nTreść lekcji:\n{content}"


def _attempt_block(attempt: Attempt) -> str:
    quiz = attempt.quiz
    reveal = attempt.passed or attempts_left(quiz, attempt.user) == 0
    if attempt.status == Attempt.Status.PENDING_REVIEW:
        status = "czeka na ocenę nauczyciela (pytania otwarte nie są jeszcze ocenione)"
    else:
        status = f"{'zaliczony' if attempt.passed else 'niezaliczony'}, {attempt.score}/{attempt.max_score} pkt ({attempt.percent}%)"
    lines = [f"UCZEŃ WYSŁAŁ JUŻ TEST „{quiz.title}” i ogląda swój wynik: {status}."]
    if reveal:
        lines.append("Test jest zamknięty — możesz omawiać poprawne odpowiedzi i tłumaczyć błędy.")
    else:
        lines.append(
            "Uczeń może jeszcze poprawić test. NIE podawaj poprawnych odpowiedzi ani nie wskazuj właściwej opcji — "
            "wyjaśnij zagadnienie i błąd w rozumowaniu, żeby sam doszedł do rozwiązania."
        )
    answers = attempt.answers.select_related("question").prefetch_related("selected_choices", "question__choices")
    for number, answer in enumerate(answers.order_by("question__order"), start=1):
        question = answer.question
        lines.append(f"\nPytanie {number}: {plain_text(question.text, 600)}")
        if question.code:
            lines.append(f"Kod w pytaniu:\n{question.code[:1200]}")
        if question.is_open:
            given = answer.text_answer[:1200] if answer.text_answer else (f"plik {answer.original_name}" if answer.file else "brak")
            lines.append(f"Odpowiedź ucznia (otwarta): {given}")
        else:
            selected = {choice.pk for choice in answer.selected_choices.all()}
            for choice in question.choices.all():
                marks = []
                if choice.pk in selected:
                    marks.append("zaznaczona przez ucznia")
                if reveal and choice.is_correct:
                    marks.append("poprawna")
                lines.append(f"  • {choice.text}" + (f" [{', '.join(marks)}]" if marks else ""))
        if answer.points is not None:
            lines.append(f"Punkty: {answer.points}/{question.points}")
        if answer.feedback:
            lines.append(f"Komentarz nauczyciela: {answer.feedback[:600]}")
    if attempt.feedback:
        lines.append(f"\nKomentarz nauczyciela do testu: {attempt.feedback[:800]}")
    return "\n".join(lines)


def system_prompt(context: ChatContext) -> str:
    parts = [BASE_PROMPT]
    if context.course:
        parts.append(_course_block(context))
    if context.lesson:
        parts.append(_lesson_block(context))
    if context.attempt:
        parts.append(_attempt_block(context.attempt))
    if not context.course:
        parts.append("Uczeń nie ma teraz otwartego żadnego kursu — to rozmowa ogólna o nauce na platformie.")
    style = getattr(context, "learning_style", None)
    if style is not None:
        parts.append(
            f"Styl uczenia się ucznia: {style.name}. {style.description} "
            "Dopasuj do niego formę wyjaśnień, ale nie zmieniaj treści merytorycznej."
        )
    return "\n\n".join(parts)


def build_messages(context: ChatContext, history: list[ChatMessage], question: str) -> list[dict]:
    messages = [{"role": "system", "content": system_prompt(context)}]
    for message in history:
        role = "assistant" if message.is_bot else "user"
        messages.append({"role": role, "content": message.message_text})
    messages.append({"role": "user", "content": question})
    return messages


def _client() -> ollama.Client:
    return ollama.Client(host=settings.OLLAMA_BASE_URL, timeout=settings.OLLAMA_TIMEOUT)


def stream_reply(messages: list[dict]) -> Iterator[str]:
    """Kolejne fragmenty odpowiedzi. Błędy Ollamy zamienia na ``AssistantUnavailable``."""
    try:
        stream = _client().chat(
            model=settings.OLLAMA_MODEL,
            messages=messages,
            stream=True,
            options={"temperature": 0.4, "num_ctx": 8192},
            keep_alive="30m",
        )
        for part in stream:
            piece = part["message"]["content"]
            if piece:
                yield piece
    except ollama.ResponseError as error:
        if error.status_code == 404:
            raise AssistantUnavailable(
                "Model asystenta nie jest jeszcze pobrany na serwerze. Administrator: "
                f"ollama pull {settings.OLLAMA_MODEL}"
            ) from error
        raise AssistantUnavailable("Serwer modelu zwrócił błąd. Spróbuj ponownie za chwilę.") from error
    except httpx.TimeoutException as error:
        raise AssistantUnavailable("Asystent odpowiadał zbyt długo. Spróbuj zadać krótsze pytanie.") from error
    except (ConnectionError, httpx.HTTPError) as error:
        raise AssistantUnavailable(
            "Asystent jest chwilowo niedostępny — serwer modelu (Ollama) nie odpowiada."
        ) from error


def complete_reply(messages: list[dict]) -> str:
    reply = "".join(stream_reply(messages)).strip()
    if not reply:
        raise AssistantUnavailable("Asystent nie zwrócił odpowiedzi. Spróbuj zadać pytanie inaczej.")
    return reply
