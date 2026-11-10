"""Jedna wymiana pytanie → odpowiedź: zapis pytania, odmowa w trakcie testu, zapis odpowiedzi."""

from __future__ import annotations

from dataclasses import dataclass, field

from apps.chatbot import llm
from apps.chatbot.models import ChatMessage, ChatSession
from apps.chatbot.services import ChatContext, get_session, lock_info, model_history, record_refusal
from apps.learning.quizzes import active_sitting


@dataclass
class Turn:
    session: ChatSession
    question: ChatMessage | None = None
    prompt: list[dict] = field(default_factory=list)
    refusal: list[ChatMessage] = field(default_factory=list)  # [pytanie, odmowa] gdy trwa test
    lock: dict | None = None

    @property
    def refused(self) -> bool:
        return bool(self.refusal)


def start_turn(user, context: ChatContext, text: str) -> Turn:
    """Zapisuje pytanie. W trakcie testu od razu zapisuje odmowę i nie pyta modelu."""
    session = get_session(user, context.course)
    sitting = active_sitting(user)
    if sitting is not None:
        return Turn(session=session, refusal=record_refusal(session, text, context.lesson, sitting), lock=lock_info(sitting))
    question = ChatMessage.objects.create(
        session=session, sender=ChatMessage.Sender.USER, message_text=text, lesson=context.lesson
    )
    from apps.learning.styles import user_style

    context.learning_style = user_style(user)
    history = model_history(session, before=question)
    return Turn(session=session, question=question, prompt=llm.build_messages(context, history, text))


def finish_turn(turn: Turn, reply: str) -> ChatMessage:
    answer = ChatMessage.objects.create(
        session=turn.session, sender=ChatMessage.Sender.BOT, message_text=reply, lesson=turn.question.lesson
    )
    turn.session.save(update_fields=["updated_at"])
    return answer


def cancel_turn(turn: Turn) -> None:
    """Model nie odpowiedział — pytanie znika z historii, uczeń może je wysłać ponownie."""
    if turn.question is not None and turn.question.pk:
        turn.question.delete()
