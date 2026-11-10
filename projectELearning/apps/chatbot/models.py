from __future__ import annotations

from django.conf import settings
from django.db import models
from django.db.models import Q


class ChatSession(models.Model):
    """Rozmowa ucznia z asystentem AI: jedna na kurs i jedna ogólna (``course`` puste)."""

    student = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='chat_sessions',
        verbose_name='Uczeń',
    )
    course = models.ForeignKey(
        'catalog.Course',
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name='chat_sessions',
        verbose_name='Kurs',
    )
    created_at = models.DateTimeField('Utworzono', auto_now_add=True)
    updated_at = models.DateTimeField('Ostatnia wiadomość', auto_now=True)

    class Meta:
        ordering = ['-updated_at']
        verbose_name = 'Rozmowa z asystentem'
        verbose_name_plural = 'Rozmowy z asystentem'
        constraints = [
            models.UniqueConstraint(
                fields=['student', 'course'],
                condition=Q(course__isnull=False),
                name='unique_chat_session_per_course',
            ),
            models.UniqueConstraint(
                fields=['student'],
                condition=Q(course__isnull=True),
                name='unique_general_chat_session',
            ),
        ]

    def __str__(self) -> str:
        return f"{self.student.username} — {self.title}"

    @property
    def title(self) -> str:
        return self.course.title if self.course_id else 'Rozmowa ogólna'


class ChatMessage(models.Model):
    """Pojedyncza wiadomość w rozmowie."""

    class Sender(models.TextChoices):
        USER = 'user', 'Uczeń'
        BOT = 'bot', 'Asystent'

    session = models.ForeignKey(
        ChatSession,
        on_delete=models.CASCADE,
        related_name='messages',
        verbose_name='Rozmowa',
    )
    sender = models.CharField('Nadawca', max_length=10, choices=Sender.choices)
    message_text = models.TextField('Treść wiadomości')
    lesson = models.ForeignKey(
        'learning.Lesson',
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='chat_messages',
        verbose_name='Lekcja',
        help_text='Lekcja otwarta podczas zadawania pytania.',
    )
    is_refusal = models.BooleanField(
        'Odmowa w trakcie testu',
        default=False,
        help_text='Pytanie zadane w trakcie testu i automatyczna odmowa — nie trafiają do kontekstu modelu.',
    )
    timestamp = models.DateTimeField('Wysłano', auto_now_add=True)

    class Meta:
        ordering = ['timestamp', 'pk']
        verbose_name = 'Wiadomość czatu'
        verbose_name_plural = 'Wiadomości czatu'

    def __str__(self) -> str:
        return f"[{self.sender}] {self.message_text[:60]}"

    @property
    def is_bot(self) -> bool:
        return self.sender == self.Sender.BOT

    @property
    def html(self):
        """Odpowiedź asystenta jako bezpieczny HTML (escape + podstawowy Markdown)."""
        from apps.chatbot.formatting import render_reply

        return render_reply(self.message_text)
