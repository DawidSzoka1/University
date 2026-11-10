from __future__ import annotations

from django.contrib.auth.models import AbstractUser
from django.db import models


class CustomUser(AbstractUser):
    """Rozszerzony model użytkownika z rolami, reputacją i personalizacją."""

    class Role(models.TextChoices):
        STUDENT = 'student', 'Student'
        # Nauczyciel = moderator ze specyfikacji (tworzy kursy, akceptuje zgłoszenia, ocenia testy)
        MODERATOR = 'moderator', 'Nauczyciel (moderator)'
        ADMIN = 'admin', 'Administrator'

    class Rank(models.TextChoices):
        OBSERVER = 'Obserwator', 'Obserwator'
        NOVICE = 'Nowicjusz', 'Nowicjusz'
        HELPER = 'Pomocnik', 'Pomocnik'
        EXPERT = 'Ekspert', 'Ekspert'
        MASTER = 'Mistrz', 'Mistrz'

    # Górne progi punktów dla rang (włącznie); powyżej ostatniego — Mistrz.
    RANK_THRESHOLDS = (
        (-1, Rank.OBSERVER),
        (10, Rank.NOVICE),
        (50, Rank.HELPER),
        (200, Rank.EXPERT),
    )

    # --- Rola ---
    role = models.CharField(
        max_length=20,
        choices=Role.choices,
        default=Role.STUDENT,
        verbose_name='Rola',
    )

    # --- Avatar ---
    avatar = models.ImageField(
        upload_to='avatars/',
        blank=True,
        null=True,
        verbose_name='Awatar',
    )
    avatar_url = models.URLField(
        blank=True,
        default='',
        verbose_name='URL awatara społecznościowego',
    )

    # --- Motyw kolorystyczny ---
    selected_theme = models.ForeignKey(
        'core.ColorTheme',
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='users',
        verbose_name='Wybrany motyw',
    )
    learning_style = models.ForeignKey(
        'learning.LearningStyle',
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='learners',
        verbose_name='Styl uczenia się',
        help_text='Wybrany przez ucznia; lekcje z wersjami pokazują treść dla tego stylu.',
    )
    learning_style_updated_at = models.DateTimeField('Zmiana stylu uczenia się', null=True, blank=True)

    # --- Adres ---
    street_address = models.CharField(
        max_length=255,
        blank=True,
        default='',
        verbose_name='Ulica i numer',
    )
    postal_code = models.CharField(
        max_length=20,
        blank=True,
        default='',
        verbose_name='Kod pocztowy',
    )
    city = models.CharField(
        max_length=100,
        blank=True,
        default='',
        verbose_name='Miasto',
    )
    phone_number = models.CharField(
        max_length=20,
        blank=True,
        default='',
        verbose_name='Numer telefonu',
    )

    # --- Reputacja ---
    reputation_points = models.IntegerField(
        default=0,
        verbose_name='Punkty reputacji',
    )
    rank_title = models.CharField(
        max_length=50,
        choices=Rank.choices,
        default=Rank.NOVICE,
        verbose_name='Tytuł rangi',
    )

    class Meta:
        verbose_name = 'Użytkownik'
        verbose_name_plural = 'Użytkownicy'

    def __str__(self) -> str:
        return self.username

    # --- Właściwości roli ---

    @property
    def is_student(self) -> bool:
        """Zwraca True jeśli użytkownik ma rolę studenta."""
        return self.role == self.Role.STUDENT

    @property
    def is_moderator(self) -> bool:
        """Zwraca True jeśli użytkownik ma rolę moderatora."""
        return self.role == self.Role.MODERATOR

    @property
    def is_admin(self) -> bool:
        """Zwraca True jeśli użytkownik ma rolę administratora."""
        return self.role == self.Role.ADMIN

    # --- Metody reputacji ---

    @classmethod
    def rank_for_points(cls, points: int) -> str:
        """Ranga dla sumy punktów: < 0 Obserwator, 0–10 Nowicjusz, 11–50 Pomocnik,
        51–200 Ekspert, > 200 Mistrz."""
        for upper_bound, rank in cls.RANK_THRESHOLDS:
            if points <= upper_bound:
                return rank
        return cls.Rank.MASTER

    def update_rank(self) -> None:
        """Ustawia tytuł rangi na podstawie punktów reputacji (bez zapisu)."""
        self.rank_title = self.rank_for_points(self.reputation_points)

    def recalculate_reputation(self) -> None:
        """Przelicza reputację na podstawie głosów na komentarze użytkownika.

        Sumuje wartości wszystkich CommentVote powiązanych z komentarzami
        tego użytkownika, aktualizuje reputation_points i rank_title,
        a następnie zapisuje model.
        """
        # Import lokalny: catalog importuje model użytkownika (cykl importów).
        from django.db.models import Sum

        from apps.catalog.models import CommentVote

        total = (
            CommentVote.objects.filter(comment__author=self)
            .aggregate(total=Sum('value'))
            .get('total')
        ) or 0

        self.reputation_points = total
        self.update_rank()
        self.save(update_fields=['reputation_points', 'rank_title'])
