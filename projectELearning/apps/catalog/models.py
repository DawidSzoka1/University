from __future__ import annotations

from decimal import Decimal
from typing import Any

import bleach
from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import models
from django.utils.text import slugify
from mptt.models import MPTTModel, TreeForeignKey

ALLOWED_TAGS = [
    "a", "abbr", "acronym", "b", "blockquote", "code", "em", "i", "li", "ol",
    "strong", "ul", "h2", "h3", "h4", "h5", "h6", "p", "br", "pre", "hr",
    "table", "thead", "tbody", "tr", "th", "td", "span", "img", "u", "s",
    # Lekcje: zdjęcia z podpisem i tabele z nagłówkiem
    "figure", "figcaption", "caption", "sub", "sup",
]

# Bez atrybutu target — link w nowej karcie bez rel=noopener to ryzyko (tabnabbing).
ALLOWED_ATTRIBUTES = {
    "a": ["href", "title", "rel"],
    "img": ["src", "alt", "title", "width", "height"],
    "*": ["class"],
}

ALLOWED_PROTOCOLS = ["http", "https", "mailto"]


def sanitize_html(html: str) -> str:
    """Sanitize HTML content with bleach to prevent XSS attacks."""
    if not html:
        return ""
    return bleach.clean(
        html,
        tags=ALLOWED_TAGS,
        attributes=ALLOWED_ATTRIBUTES,
        protocols=ALLOWED_PROTOCOLS,
        strip=True,
    )


class Category(MPTTModel):
    """Hierarchiczna kategoria kursów oparta o django-mptt."""

    name = models.CharField("Nazwa", max_length=120)
    slug = models.SlugField("Slug", max_length=140, unique=True, blank=True)
    parent = TreeForeignKey(
        "self",
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name="children",
        verbose_name="Kategoria nadrzędna",
    )
    description = models.TextField("Opis", blank=True)
    order = models.PositiveIntegerField("Kolejność", default=0)
    is_active = models.BooleanField("Aktywna", default=True)
    moderators = models.ManyToManyField(
        settings.AUTH_USER_MODEL,
        blank=True,
        related_name="moderated_categories",
        verbose_name="Przypisani moderatorzy",
    )

    class MPTTMeta:
        order_insertion_by = ["order", "name"]

    class Meta:
        verbose_name = "Kategoria"
        verbose_name_plural = "Kategorie"

    def __str__(self) -> str:
        return self.name

    def save(self, *args: Any, **kwargs: Any) -> None:
        if not self.slug:
            base_slug = slugify(self.name) or "kategoria"
            slug = base_slug
            counter = 1
            while Category.objects.filter(slug=slug).exclude(pk=self.pk).exists():
                slug = f"{base_slug}-{counter}"
                counter += 1
            self.slug = slug
        super().save(*args, **kwargs)

    def get_published_courses(self):
        """Zwraca opublikowane kursy dla tej kategorii i jej podkategorii."""
        descendants = self.get_descendants(include_self=True)
        return Course.objects.filter(category__in=descendants, is_published=True)


class CourseQuerySet(models.QuerySet):
    """QuerySet z metodami wyszukiwania, filtrowania i sortowania kursów."""

    def published(self):
        """Zwraca tylko opublikowane i zatwierdzone kursy."""
        return self.filter(is_published=True, status=Course.Status.APPROVED)

    def search(self, query: str):
        """Wyszukiwanie po nazwie kursu, opisie oraz imieniu, nazwisku i loginie autora."""
        if not query:
            return self
        query = query.strip()
        return self.filter(
            models.Q(title__icontains=query)
            | models.Q(description__icontains=query)
            | models.Q(author__username__icontains=query)
            | models.Q(author__first_name__icontains=query)
            | models.Q(author__last_name__icontains=query)
        )

    def sort_by(self, sort_key: str):
        """Sortowanie po zadanym parametrze:
        - 'popularity' -> liczba osób, które skorzystały (-students_count)
        - 'completions' -> liczba osób, które ukończyły (-completed_count)
        - 'rating' -> średnia ocena użytkowników (-average_rating)
        - 'newest' -> data dodania (-created_at)
        - 'oldest' -> data dodania (created_at)
        - 'price_asc' -> cena rosnąco
        - 'price_desc' -> cena malejąco
        """
        mapping = {
            "popularity": ["-students_count", "-created_at"],
            "completions": ["-completed_count", "-created_at"],
            "rating": ["-average_rating", "-reviews_count", "-created_at"],
            "newest": ["-created_at"],
            "oldest": ["created_at"],
            "price_asc": ["price", "-created_at"],
            "price_desc": ["-price", "-created_at"],
        }
        order = mapping.get(sort_key, ["-created_at"])
        return self.order_by(*order)


class Course(models.Model):
    """Model kursu / produktu edukacyjnego."""

    class Status(models.TextChoices):
        DRAFT = "draft", "Szkic"
        PENDING = "pending", "Oczekuje na akceptację"
        APPROVED = "approved", "Zatwierdzony"
        REJECTED = "rejected", "Odrzucony"

    class Level(models.TextChoices):
        BEGINNER = "beginner", "Podstawowy"
        INTERMEDIATE = "intermediate", "Średniozaawansowany"
        ADVANCED = "advanced", "Zaawansowany"

    title = models.CharField("Tytuł", max_length=200)
    slug = models.SlugField("Slug", max_length=220, unique=True, blank=True)
    category = models.ForeignKey(
        Category,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="courses",
        verbose_name="Kategoria",
    )
    author = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="created_courses",
        verbose_name="Autor / Twórca kursu",
    )
    description = models.TextField("Krótki opis / zajawka", blank=True)
    description_html = models.TextField("Opis kursu (O kursie)", blank=True)
    thumbnail = models.ImageField("Miniatura", upload_to="courses/", blank=True, null=True)

    # Cena i darmowość kursu
    price = models.DecimalField("Cena (PLN)", max_digits=10, decimal_places=2, default=Decimal("0.00"))
    is_free = models.BooleanField(
        "Kurs bezpłatny",
        default=False,
        help_text="Zaznacz, jeśli kurs ma być dostępny za darmo dla wszystkich studentów.",
    )

    difficulty_level = models.CharField(
        "Poziom trudności",
        max_length=20,
        choices=Level.choices,
        default=Level.BEGINNER,
    )
    duration_hours = models.PositiveIntegerField("Szacowany czas (godziny)", default=1)
    recommended_styles = models.ManyToManyField(
        "learning.LearningStyle",
        blank=True,
        related_name="recommended_courses",
        verbose_name="Szczególnie dla stylów",
        help_text="Style, dla których kurs jest szczególnie odpowiedni (wpływa na polecanie kursów).",
    )
    has_certificate = models.BooleanField(
        "Dyplom ukończenia",
        default=True,
        help_text="Po ukończeniu wszystkich lekcji i zdaniu testów uczeń pobierze dyplom w PDF.",
    )

    # Statystyki do sortowania i prezentacji
    students_count = models.PositiveIntegerField(
        "Liczba uczestników",
        default=0,
        db_index=True,
        help_text="Liczba osób, które zapisały się na kurs.",
    )
    completed_count = models.PositiveIntegerField(
        "Liczba ukończeń",
        default=0,
        db_index=True,
        help_text="Liczba osób, które ukończyły ten kurs.",
    )
    average_rating = models.DecimalField(
        "Średnia ocena",
        max_digits=3,
        decimal_places=2,
        default=Decimal("0.00"),
        db_index=True,
    )
    reviews_count = models.PositiveIntegerField("Liczba recenzji", default=0)

    status = models.CharField(
        "Status moderacji",
        max_length=20,
        choices=Status.choices,
        default=Status.APPROVED,
    )
    rejection_reason = models.TextField("Powód odrzucenia", blank=True)
    is_published = models.BooleanField("Opublikowany publicznie", default=True)

    created_at = models.DateTimeField("Utworzono", auto_now_add=True)
    updated_at = models.DateTimeField("Zaktualizowano", auto_now=True)

    objects = CourseQuerySet.as_manager()

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Kurs"
        verbose_name_plural = "Kursy"
        indexes = [
            models.Index(fields=["-created_at"]),
            models.Index(fields=["-students_count"]),
            models.Index(fields=["-completed_count"]),
            models.Index(fields=["-average_rating"]),
            models.Index(fields=["price"]),
        ]

    def __str__(self) -> str:
        return self.title

    @property
    def is_course_free(self) -> bool:
        """Zwraca True, jeśli kurs jest oznaczony jako bezpłatny lub ma cenę 0 zł."""
        return self.is_free or self.price == Decimal("0.00")

    def get_price_display(self) -> str:
        """Czytelny format ceny dla szablonów i interfejsu."""
        if self.is_course_free:
            return "Bezpłatny"
        from apps.core.money import format_pln

        return format_pln(self.price)

    def completion_stats(self) -> tuple[int, int, int]:
        """(zapisani, ukończyli, procent ukończenia) liczone na bieżąco z zapisów."""
        started = self.enrollments.count()
        completed = self.enrollments.filter(is_completed=True).count()
        return started, completed, round(100 * completed / started) if started else 0

    def recalculate_stats(self) -> None:
        """Przelicza liczbę zapisanych uczestników, ukończeń oraz średnią ocen."""
        from django.db.models import Avg

        self.students_count = self.enrollments.count()
        self.completed_count = self.enrollments.filter(is_completed=True).count()
        approved_reviews = self.comments.filter(
            is_approved=True, is_removed=False, parent__isnull=True, rating__isnull=False
        )
        self.reviews_count = approved_reviews.count()
        avg = approved_reviews.aggregate(avg=Avg("rating"))["avg"]
        self.average_rating = Decimal(str(round(avg, 2))) if avg is not None else Decimal("0.00")

        self.save(
            update_fields=[
                "students_count",
                "completed_count",
                "reviews_count",
                "average_rating",
            ]
        )

    def save(self, *args: Any, **kwargs: Any) -> None:
        # Automatyczny unikalny slug
        if not self.slug:
            base_slug = slugify(self.title) or "kurs"
            slug = base_slug
            counter = 1
            while Course.objects.filter(slug=slug).exclude(pk=self.pk).exists():
                slug = f"{base_slug}-{counter}"
                counter += 1
            self.slug = slug

        # Synchronizacja ceny z flagą is_free (formularz kursu ustawia is_free z ceny)
        if self.is_free:
            self.price = Decimal("0.00")
        elif self.price == Decimal("0.00"):
            self.is_free = True

        # Nowo przesłana miniatura: skalowanie lokalne Pillow (16:9, maks. 1280×720)
        if self.thumbnail:
            from apps.core.images import resize_image_field

            resize_image_field(self.thumbnail, (1280, 720))

        # Sanityzacja opisu HTML (ochrona XSS)
        if self.description_html:
            self.description_html = sanitize_html(self.description_html)

        super().save(*args, **kwargs)


class Enrollment(models.Model):
    """Zapis studenta na kurs (dostęp do materiałów i testów)."""

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="enrollments",
        verbose_name="Student",
    )
    course = models.ForeignKey(
        Course,
        on_delete=models.CASCADE,
        related_name="enrollments",
        verbose_name="Kurs",
    )
    is_completed = models.BooleanField("Czy ukończono kurs", default=False, db_index=True)
    completed_at = models.DateTimeField("Data ukończenia", null=True, blank=True)
    progress_percent = models.PositiveSmallIntegerField("Postęp (%)", default=0)
    created_at = models.DateTimeField("Data zapisu", auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Zapis na kurs"
        verbose_name_plural = "Zapisy na kursy"
        constraints = [
            models.UniqueConstraint(
                fields=["user", "course"],
                name="unique_student_course_enrollment",
            )
        ]

    def __str__(self) -> str:
        status = " (ukończono)" if self.is_completed else f" ({self.progress_percent}%)"
        return f"{self.user} → {self.course}{status}"

    def mark_completed(self) -> None:
        """Oznacza kurs jako ukończony przez studenta i aktualizuje statystyki kursu."""
        from django.utils import timezone

        if not self.is_completed:
            self.is_completed = True
            self.progress_percent = 100
            self.completed_at = timezone.now()
            self.save(update_fields=["is_completed", "progress_percent", "completed_at"])
            self.course.recalculate_stats()

    def save(self, *args: Any, **kwargs: Any) -> None:
        is_new = self.pk is None
        super().save(*args, **kwargs)
        if is_new:
            self.course.recalculate_stats()


RATING_CHOICES = [
    (1, "1 — słabo"),
    (2, "2 — przeciętnie"),
    (3, "3 — dobrze"),
    (4, "4 — bardzo dobrze"),
    (5, "5 — świetnie"),
]


class Comment(models.Model):
    """Komentarz pod kursem. Drzewo odpowiedzi przez ``parent`` (dowolna głębokość w bazie;
    widok spłaszcza zbyt głębokie wątki). Komentarz główny może zawierać ocenę kursu."""

    course = models.ForeignKey(
        Course,
        on_delete=models.CASCADE,
        related_name="comments",
        verbose_name="Kurs",
    )
    author = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="catalog_comments",
        verbose_name="Autor",
    )
    parent = models.ForeignKey(
        "self",
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name="replies",
        verbose_name="Odpowiedź na komentarz",
    )
    content = models.TextField("Treść recenzji / komentarza")
    # Ocenę wystawia tylko komentarz główny (recenzja); odpowiedzi i zwykłe komentarze — brak.
    rating = models.PositiveSmallIntegerField(
        "Ocena (1-5)",
        null=True,
        blank=True,
        choices=RATING_CHOICES,
    )
    # Nowy komentarz widzą tylko autor i moderatorzy kategorii, dopóki ktoś go nie zatwierdzi.
    is_approved = models.BooleanField("Zatwierdzony", default=False)
    approved_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="approved_catalog_comments",
        verbose_name="Zatwierdzający moderator",
    )
    approved_at = models.DateTimeField("Data zatwierdzenia", null=True, blank=True)
    # Usunięty komentarz z odpowiedziami zostaje jako „Komentarz usunięty”, żeby wątek się nie rozpadł.
    is_removed = models.BooleanField("Usunięty", default=False)
    edited_at = models.DateTimeField("Edytowano", null=True, blank=True)
    created_at = models.DateTimeField("Utworzono", auto_now_add=True)
    updated_at = models.DateTimeField("Zaktualizowano", auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Komentarz / Recenzja"
        verbose_name_plural = "Komentarze i recenzje"

    def __str__(self) -> str:
        if self.rating:
            return f"Ocena {self.rating}/5 od {self.author} dla {self.course}"
        return f"Komentarz {self.author} do {self.course}"

    @property
    def is_public(self) -> bool:
        """Widoczny dla wszystkich: zatwierdzony i nieusunięty."""
        return self.is_approved and not self.is_removed

    @property
    def score(self) -> int:
        """Suma punktów z głosów pomocności."""
        return sum(v.value for v in self.votes.all())

    def save(self, *args: Any, **kwargs: Any) -> None:
        super().save(*args, **kwargs)
        if self.course_id:
            self.course.recalculate_stats()


class CommentVote(models.Model):
    """Głos pomocności (+1 / -1) oddany na komentarz/recenzję."""

    comment = models.ForeignKey(
        Comment,
        on_delete=models.CASCADE,
        related_name="votes",
        verbose_name="Komentarz",
    )
    voter = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="comment_votes",
        verbose_name="Głosujący",
    )
    value = models.SmallIntegerField(
        "Wartość głosu",
        choices=((-1, "-1 (Niepomocny)"), (1, "+1 (Pomocny)")),
    )
    created_at = models.DateTimeField("Data głosu", auto_now_add=True)

    class Meta:
        verbose_name = "Głos na komentarz"
        verbose_name_plural = "Głosy na komentarze"
        constraints = [
            models.UniqueConstraint(fields=["comment", "voter"], name="unique_comment_voter")
        ]

    def __str__(self) -> str:
        return f"{self.voter} ({'+1' if self.value > 0 else '-1'}) na #{self.comment_id}"

    def clean(self) -> None:
        super().clean()
        if self.comment.author_id and self.voter_id == self.comment.author_id:
            raise ValidationError("Nie możesz głosować na własny komentarz.")

    def save(self, *args: Any, **kwargs: Any) -> None:
        # Reputację autora komentarza przelicza sygnał (apps/catalog/signals.py) —
        # także przy zmianie i usunięciu głosu.
        self.full_clean()
        super().save(*args, **kwargs)


class CourseSubmissionMessage(models.Model):
    """Wiadomość w wątku zgłoszenia / weryfikacji kursu między nauczycielem a moderatorem."""

    course = models.ForeignKey(
        Course,
        on_delete=models.CASCADE,
        related_name="submission_messages",
        verbose_name="Kurs",
    )
    author = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="submission_messages",
        verbose_name="Autor",
    )
    content = models.TextField("Treść wiadomości")
    created_at = models.DateTimeField("Wysłano", auto_now_add=True)

    class Meta:
        ordering = ["created_at"]
        verbose_name = "Wiadomość zgłoszenia"
        verbose_name_plural = "Wiadomości zgłoszeń"

    def __str__(self) -> str:
        return f"Zgłoszenie #{self.id} dla {self.course}"
