from __future__ import annotations

import uuid
from pathlib import Path

from django.conf import settings
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models


def _quiz_media_path(prefix: str, filename: str) -> str:
    return f"quizzes/{prefix}/{uuid.uuid4().hex}/{Path(filename).name}"


def question_image_path(instance, filename: str) -> str:
    return _quiz_media_path("images", filename)


def question_video_path(instance, filename: str) -> str:
    return _quiz_media_path("videos", filename)


def question_file_path(instance, filename: str) -> str:
    return _quiz_media_path("files", filename)


def answer_file_path(instance, filename: str) -> str:
    # Odpowiedzi uczniów: losowa ścieżka, pobierane wyłącznie przez chroniony widok
    return _quiz_media_path("answers", filename)


class Quiz(models.Model):
    """Test w kursie. Stoi po wybranej lekcji (``after_lesson``) albo na końcu kursu (puste pole).
    Test obowiązkowy blokuje dalsze lekcje, dopóki uczeń go nie zda."""

    course = models.ForeignKey(
        "catalog.Course",
        on_delete=models.CASCADE,
        related_name="quizzes",
        verbose_name="Kurs",
    )
    after_lesson = models.ForeignKey(
        "Lesson",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="quizzes_after",
        verbose_name="Po lekcji",
        help_text="Puste — test na końcu kursu.",
    )
    title = models.CharField("Tytuł", max_length=200)
    description = models.TextField("Instrukcja dla ucznia", blank=True)
    is_required = models.BooleanField(
        "Trzeba zdać, aby przejść dalej",
        default=False,
        help_text="Kolejne lekcje i testy odblokują się dopiero po zdaniu tego testu.",
    )
    pass_percent = models.PositiveSmallIntegerField(
        "Próg zaliczenia (%)",
        default=60,
        validators=[MinValueValidator(1), MaxValueValidator(100)],
    )
    max_attempts = models.PositiveSmallIntegerField(
        "Limit podejść",
        default=0,
        help_text="0 — bez limitu.",
    )
    order = models.PositiveIntegerField("Kolejność", default=0)
    created_at = models.DateTimeField("Utworzono", auto_now_add=True)
    updated_at = models.DateTimeField("Zaktualizowano", auto_now=True)

    class Meta:
        ordering = ["order", "pk"]
        verbose_name = "Test"
        verbose_name_plural = "Testy"

    def __str__(self) -> str:
        return self.title

    @property
    def max_score(self) -> int:
        return sum(q.points for q in self.questions.all())

    @property
    def has_open_questions(self) -> bool:
        return any(q.is_open for q in self.questions.all())


class Question(models.Model):
    """Pytanie: zamknięte (ABC, sprawdzane automatycznie) albo otwarte (ocenia nauczyciel).
    Treść może mieć zdjęcie, film, fragment kodu i plik do pobrania."""

    class QuestionType(models.TextChoices):
        CLOSED_SINGLE = "closed_single", "Jednokrotny wybór (ABC)"
        CLOSED_MULTIPLE = "closed_multiple", "Wielokrotny wybór"
        OPEN = "open", "Otwarte (ocenia nauczyciel)"

    class AnswerFormat(models.TextChoices):
        TEXT = "text", "Tekst"
        CODE = "code", "Kod"
        IMAGE = "image", "Zdjęcie"
        FILE = "file", "Plik"

    quiz = models.ForeignKey(
        Quiz,
        on_delete=models.CASCADE,
        related_name="questions",
        verbose_name="Test",
    )
    question_type = models.CharField(
        "Typ pytania",
        max_length=20,
        choices=QuestionType.choices,
        default=QuestionType.CLOSED_SINGLE,
    )
    answer_format = models.CharField(
        "Forma odpowiedzi (pytanie otwarte)",
        max_length=10,
        choices=AnswerFormat.choices,
        default=AnswerFormat.TEXT,
    )
    text = models.TextField("Treść pytania")
    points = models.PositiveSmallIntegerField(
        "Punkty", default=1, validators=[MinValueValidator(1), MaxValueValidator(100)]
    )
    image = models.ImageField("Zdjęcie", upload_to=question_image_path, blank=True)
    image_alt = models.CharField("Opis zdjęcia (tekst alternatywny)", max_length=255, blank=True)
    video_url = models.URLField("Film z YouTube lub Vimeo", blank=True)
    video_file = models.FileField("Własny plik wideo", upload_to=question_video_path, blank=True)
    code = models.TextField("Fragment kodu", blank=True)
    attachment = models.FileField("Plik do pobrania", upload_to=question_file_path, blank=True)
    attachment_name = models.CharField("Nazwa pliku", max_length=255, blank=True)
    order = models.PositiveIntegerField("Kolejność", default=0)

    class Meta:
        ordering = ["order", "pk"]
        verbose_name = "Pytanie"
        verbose_name_plural = "Pytania"

    def __str__(self) -> str:
        return self.text[:80]

    @property
    def is_open(self) -> bool:
        return self.question_type == self.QuestionType.OPEN

    @property
    def correct_choice_ids(self) -> set[int]:
        return {c.pk for c in self.choices.all() if c.is_correct}


class Choice(models.Model):
    """Odpowiedź do pytania zamkniętego; nauczyciel zaznacza poprawne."""

    question = models.ForeignKey(
        Question,
        on_delete=models.CASCADE,
        related_name="choices",
        verbose_name="Pytanie",
    )
    text = models.CharField("Treść", max_length=255)
    is_correct = models.BooleanField("Poprawna", default=False)
    order = models.PositiveIntegerField("Kolejność", default=0)

    class Meta:
        ordering = ["order", "pk"]
        verbose_name = "Odpowiedź"
        verbose_name_plural = "Odpowiedzi"

    def __str__(self) -> str:
        return self.text


class Attempt(models.Model):
    """Podejście ucznia do testu. Z pytaniami otwartymi czeka na ocenę nauczyciela."""

    class Status(models.TextChoices):
        PENDING_REVIEW = "pending_review", "Czeka na ocenę"
        GRADED = "graded", "Oceniony"

    quiz = models.ForeignKey(Quiz, on_delete=models.CASCADE, related_name="attempts", verbose_name="Test")
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="quiz_attempts",
        verbose_name="Uczeń",
    )
    status = models.CharField("Status", max_length=20, choices=Status.choices, default=Status.PENDING_REVIEW)
    score = models.PositiveIntegerField("Punkty", default=0)
    max_score = models.PositiveIntegerField("Maks. punktów", default=0)
    passed = models.BooleanField("Zaliczony", default=False)
    feedback = models.TextField("Komentarz nauczyciela", blank=True)
    graded_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="graded_attempts",
        verbose_name="Oceniający",
    )
    learning_style = models.ForeignKey(
        "learning.LearningStyle",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="attempts",
        verbose_name="Styl uczenia się",
        help_text="Styl, w którym uczeń przerabiał lekcje kursu przed testem (podstawa rekomendacji stylu).",
    )
    submitted_at = models.DateTimeField("Wysłano", auto_now_add=True)
    graded_at = models.DateTimeField("Oceniono", null=True, blank=True)

    class Meta:
        ordering = ["-submitted_at"]
        verbose_name = "Podejście do testu"
        verbose_name_plural = "Podejścia do testów"

    def __str__(self) -> str:
        return f"{self.user} → {self.quiz} ({self.get_status_display()})"

    @property
    def percent(self) -> int:
        return round(100 * self.score / self.max_score) if self.max_score else 0


class QuizSitting(models.Model):
    """Test otwarty do rozwiązania, jeszcze niewysłany. Blokuje asystenta AI do wysłania
    odpowiedzi albo do upływu ``CHATBOT_QUIZ_LOCK_MINUTES``."""

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="quiz_sittings",
        verbose_name="Uczeń",
    )
    quiz = models.ForeignKey(Quiz, on_delete=models.CASCADE, related_name="sittings", verbose_name="Test")
    started_at = models.DateTimeField("Rozpoczęto")

    class Meta:
        ordering = ["-started_at"]
        verbose_name = "Rozpoczęty test"
        verbose_name_plural = "Rozpoczęte testy"
        constraints = [models.UniqueConstraint(fields=["user", "quiz"], name="unique_quiz_sitting")]

    def __str__(self) -> str:
        return f"{self.user} — {self.quiz} (od {self.started_at:%H:%M})"


class Answer(models.Model):
    """Odpowiedź na jedno pytanie. ``points`` puste = jeszcze nieocenione (pytanie otwarte)."""

    attempt = models.ForeignKey(Attempt, on_delete=models.CASCADE, related_name="answers", verbose_name="Podejście")
    question = models.ForeignKey(Question, on_delete=models.CASCADE, related_name="answers", verbose_name="Pytanie")
    selected_choices = models.ManyToManyField(Choice, blank=True, related_name="answers", verbose_name="Wybrane odpowiedzi")
    text_answer = models.TextField("Odpowiedź tekstowa / kod", blank=True)
    file = models.FileField("Przesłany plik", upload_to=answer_file_path, blank=True)
    original_name = models.CharField("Nazwa pliku", max_length=255, blank=True)
    points = models.PositiveSmallIntegerField("Przyznane punkty", null=True, blank=True)
    feedback = models.TextField("Komentarz do odpowiedzi", blank=True)

    class Meta:
        ordering = ["question__order", "question__pk"]
        verbose_name = "Odpowiedź ucznia"
        verbose_name_plural = "Odpowiedzi uczniów"
        constraints = [
            models.UniqueConstraint(fields=["attempt", "question"], name="unique_attempt_question"),
        ]

    def __str__(self) -> str:
        return f"{self.attempt} / {self.question}"

    @property
    def is_image(self) -> bool:
        return self.question.answer_format == Question.AnswerFormat.IMAGE


class Certificate(models.Model):
    """Dyplom ukończenia kursu. Dane (imię, tytuł) zapisane w chwili wystawienia; PDF
    generowany przy pobraniu, a kod pozwala sprawdzić autentyczność."""

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="certificates",
        verbose_name="Uczeń",
    )
    course = models.ForeignKey(
        "catalog.Course",
        on_delete=models.CASCADE,
        related_name="certificates",
        verbose_name="Kurs",
    )
    code = models.CharField("Numer dyplomu", max_length=32, unique=True)
    full_name = models.CharField("Imię i nazwisko", max_length=300)
    course_title = models.CharField("Tytuł kursu", max_length=200)
    teacher_name = models.CharField("Prowadzący", max_length=300, blank=True)
    issued_at = models.DateTimeField("Wystawiono", auto_now_add=True)

    class Meta:
        ordering = ["-issued_at"]
        verbose_name = "Dyplom"
        verbose_name_plural = "Dyplomy"
        constraints = [
            models.UniqueConstraint(fields=["user", "course"], name="unique_certificate_user_course"),
        ]

    def __str__(self) -> str:
        return f"{self.code}: {self.full_name} — {self.course_title}"


# ---------------------------------------------------------------------------
# Program kursu: sekcje → lekcje (tekst ze zdjęciami, film, pliki)
# Kolejność (order) to sugerowana ścieżka — uczeń może uczyć się w dowolnej.
# ---------------------------------------------------------------------------

def _random_path(prefix: str, filename: str) -> str:
    # Losowy katalog: adresu pliku nie da się zgadnąć, oryginalna nazwa zostaje do pobrania.
    return f"lessons/{prefix}/{uuid.uuid4().hex}/{Path(filename).name}"


def lesson_video_path(instance, filename: str) -> str:
    return _random_path("videos", filename)


def lesson_attachment_path(instance, filename: str) -> str:
    return _random_path("files", filename)


class CourseSection(models.Model):
    """Sekcja kursu, np. „Instalacja” albo „Pierwsze linie kodu”."""

    course = models.ForeignKey(
        "catalog.Course",
        on_delete=models.CASCADE,
        related_name="sections",
        verbose_name="Kurs",
    )
    title = models.CharField("Tytuł sekcji", max_length=200)
    order = models.PositiveIntegerField("Kolejność", default=0)
    created_at = models.DateTimeField("Utworzono", auto_now_add=True)
    updated_at = models.DateTimeField("Zaktualizowano", auto_now=True)

    class Meta:
        ordering = ["order", "pk"]
        verbose_name = "Sekcja kursu"
        verbose_name_plural = "Sekcje kursu"

    def __str__(self) -> str:
        return f"{self.course} → {self.title}"


class Lesson(models.Model):
    """Lekcja: treść HTML (zdjęcia, kod, tabele), opcjonalny film i pliki do pobrania."""

    section = models.ForeignKey(
        CourseSection,
        on_delete=models.CASCADE,
        related_name="lessons",
        verbose_name="Sekcja",
    )
    title = models.CharField("Tytuł lekcji", max_length=200)
    content_html = models.TextField("Treść lekcji", blank=True)
    video_url = models.URLField(
        "Film z YouTube lub Vimeo",
        blank=True,
        help_text="Link do filmu, np. https://www.youtube.com/watch?v=…",
    )
    video_file = models.FileField(
        "Własny plik wideo",
        upload_to=lesson_video_path,
        blank=True,
        help_text="MP4 lub WebM, do 200 MB.",
    )
    duration_minutes = models.PositiveIntegerField("Czas (minuty)", default=5)
    is_preview = models.BooleanField(
        "Darmowy podgląd",
        default=False,
        help_text="Lekcję zobaczy każdy, także bez zapisu na kurs.",
    )
    same_for_all_styles = models.BooleanField(
        "Ta sama treść dla każdego stylu uczenia się",
        default=True,
        help_text="Odznacz, aby dodać osobne wersje treści dla wybranych stylów uczenia się.",
    )
    order = models.PositiveIntegerField("Kolejność", default=0)
    created_at = models.DateTimeField("Utworzono", auto_now_add=True)
    updated_at = models.DateTimeField("Zaktualizowano", auto_now=True)

    class Meta:
        ordering = ["order", "pk"]
        verbose_name = "Lekcja"
        verbose_name_plural = "Lekcje"

    def __str__(self) -> str:
        return self.title

    @property
    def course(self):
        return self.section.course

    def save(self, *args, **kwargs) -> None:
        from apps.catalog.models import sanitize_html

        # Treść z edytora zawsze czyszczona po stronie serwera (XSS)
        self.content_html = sanitize_html(self.content_html)
        super().save(*args, **kwargs)


class LessonAttachment(models.Model):
    """Plik do pobrania przy lekcji, np. „pobierz plik i na jego podstawie…”."""

    lesson = models.ForeignKey(
        Lesson,
        on_delete=models.CASCADE,
        related_name="attachments",
        verbose_name="Lekcja",
    )
    file = models.FileField("Plik", upload_to=lesson_attachment_path)
    original_name = models.CharField("Nazwa pliku", max_length=255)
    size = models.PositiveBigIntegerField("Rozmiar (bajty)", default=0)
    uploaded_at = models.DateTimeField("Dodano", auto_now_add=True)

    class Meta:
        ordering = ["original_name", "pk"]
        verbose_name = "Plik lekcji"
        verbose_name_plural = "Pliki lekcji"

    def __str__(self) -> str:
        return self.original_name

    @property
    def extension(self) -> str:
        return Path(self.original_name).suffix.lstrip(".").upper()


class LessonProgress(models.Model):
    """Ukończenie lekcji przez ucznia (podstawa paska postępu i podpowiedzi „co dalej”)."""

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="lesson_progress",
        verbose_name="Uczeń",
    )
    lesson = models.ForeignKey(
        Lesson,
        on_delete=models.CASCADE,
        related_name="progress",
        verbose_name="Lekcja",
    )
    style = models.ForeignKey(
        "learning.LearningStyle",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="lesson_progress",
        verbose_name="Wersja dla stylu",
        help_text="Wersja lekcji, którą uczeń przerobił (puste = treść podstawowa).",
    )
    completed_at = models.DateTimeField("Ukończono", auto_now_add=True)

    class Meta:
        verbose_name = "Ukończona lekcja"
        verbose_name_plural = "Ukończone lekcje"
        constraints = [
            models.UniqueConstraint(fields=["user", "lesson"], name="unique_user_lesson_progress"),
        ]

    def __str__(self) -> str:
        return f"{self.user} ✓ {self.lesson}"


# ---------------------------------------------------------------------------
# Style uczenia się: słownik stylów (admin dodaje kolejne) i wersje lekcji
# ---------------------------------------------------------------------------

class LearningStyle(models.Model):
    """Styl uczenia się, np. wzrokowiec. Domyślne cztery tworzy migracja, admin dodaje kolejne."""

    name = models.CharField("Nazwa", max_length=80, unique=True)
    slug = models.SlugField("Identyfikator w adresie", max_length=90, unique=True, blank=True)
    description = models.TextField(
        "Opis dla ucznia",
        help_text="Jak uczy się osoba z tym stylem — wyświetlane przy wyborze stylu.",
    )
    author_hint = models.TextField(
        "Wskazówka dla autora kursu",
        blank=True,
        help_text="Co warto umieścić w wersji lekcji dla tego stylu.",
    )
    order = models.PositiveIntegerField("Kolejność", default=0)
    is_active = models.BooleanField(
        "Aktywny",
        default=True,
        help_text="Nieaktywny styl znika z wyboru i z lekcji; wersje lekcji zostają w bazie.",
    )
    created_at = models.DateTimeField("Utworzono", auto_now_add=True)

    class Meta:
        ordering = ["order", "name"]
        verbose_name = "Styl uczenia się"
        verbose_name_plural = "Style uczenia się"

    def __str__(self) -> str:
        return self.name

    def save(self, *args, **kwargs) -> None:
        if not self.slug:
            from django.utils.text import slugify

            base = slugify(self.name.replace("ł", "l").replace("Ł", "L")) or "styl"
            slug, number = base, 2
            while LearningStyle.objects.filter(slug=slug).exclude(pk=self.pk).exists():
                slug, number = f"{base}-{number}", number + 1
            self.slug = slug
        super().save(*args, **kwargs)


class LessonVariant(models.Model):
    """Wersja treści lekcji dla jednego stylu uczenia się (zastępuje treść i opcjonalnie film)."""

    lesson = models.ForeignKey(Lesson, on_delete=models.CASCADE, related_name="variants", verbose_name="Lekcja")
    style = models.ForeignKey(
        LearningStyle, on_delete=models.CASCADE, related_name="lesson_variants", verbose_name="Styl uczenia się"
    )
    content_html = models.TextField("Treść dla tego stylu")
    video_url = models.URLField(
        "Film dla tego stylu (YouTube lub Vimeo)",
        blank=True,
        help_text="Puste — uczeń zobaczy film z wersji podstawowej (jeśli jest).",
    )
    created_at = models.DateTimeField("Utworzono", auto_now_add=True)
    updated_at = models.DateTimeField("Zaktualizowano", auto_now=True)

    class Meta:
        ordering = ["style__order", "style__name"]
        verbose_name = "Wersja lekcji dla stylu"
        verbose_name_plural = "Wersje lekcji dla stylów"
        constraints = [
            models.UniqueConstraint(fields=["lesson", "style"], name="unique_lesson_style_variant"),
        ]

    def __str__(self) -> str:
        return f"{self.lesson} — {self.style}"

    def save(self, *args, **kwargs) -> None:
        from apps.catalog.models import sanitize_html

        self.content_html = sanitize_html(self.content_html)
        super().save(*args, **kwargs)


# ---------------------------------------------------------------------------
# Ankieta diagnostyczna stylu uczenia się (pytania sytuacyjne → profil stylów)
# ---------------------------------------------------------------------------

class SurveyQuestion(models.Model):
    """Pytanie sytuacyjne ankiety; każda odpowiedź wskazuje jeden styl uczenia się."""

    text = models.CharField("Treść pytania", max_length=300)
    order = models.PositiveIntegerField("Kolejność", default=0)
    is_active = models.BooleanField("Aktywne", default=True)

    class Meta:
        ordering = ["order", "pk"]
        verbose_name = "Pytanie ankiety stylu"
        verbose_name_plural = "Pytania ankiety stylu"

    def __str__(self) -> str:
        return self.text


class SurveyOption(models.Model):
    question = models.ForeignKey(SurveyQuestion, on_delete=models.CASCADE, related_name="options", verbose_name="Pytanie")
    text = models.CharField("Odpowiedź", max_length=300)
    style = models.ForeignKey(
        LearningStyle, on_delete=models.CASCADE, related_name="survey_options", verbose_name="Wskazuje styl"
    )
    order = models.PositiveIntegerField("Kolejność", default=0)

    class Meta:
        ordering = ["order", "pk"]
        verbose_name = "Odpowiedź ankiety"
        verbose_name_plural = "Odpowiedzi ankiety"

    def __str__(self) -> str:
        return self.text


class SurveyResponse(models.Model):
    """Jedno wypełnienie ankiety. Wynik: udział odpowiedzi wskazujących każdy styl."""

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="style_surveys", verbose_name="Uczeń"
    )
    top_style = models.ForeignKey(
        LearningStyle,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="survey_wins",
        verbose_name="Dominujący styl",
    )
    created_at = models.DateTimeField("Wypełniono", auto_now_add=True)

    class Meta:
        ordering = ["-created_at", "-pk"]
        verbose_name = "Wypełniona ankieta stylu"
        verbose_name_plural = "Wypełnione ankiety stylu"

    def __str__(self) -> str:
        return f"{self.user} → {self.top_style or '—'}"


class SurveyAnswer(models.Model):
    response = models.ForeignKey(SurveyResponse, on_delete=models.CASCADE, related_name="answers", verbose_name="Ankieta")
    question = models.ForeignKey(SurveyQuestion, on_delete=models.CASCADE, related_name="answers", verbose_name="Pytanie")
    option = models.ForeignKey(SurveyOption, on_delete=models.CASCADE, related_name="answers", verbose_name="Odpowiedź")

    class Meta:
        verbose_name = "Odpowiedź w ankiecie"
        verbose_name_plural = "Odpowiedzi w ankietach"
        constraints = [models.UniqueConstraint(fields=["response", "question"], name="unique_survey_answer")]
