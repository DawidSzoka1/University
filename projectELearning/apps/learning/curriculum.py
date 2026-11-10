"""Program kursu: dostęp do lekcji, numeracja 1.1/2.1, postęp ucznia i zmiana kolejności."""

from __future__ import annotations

from dataclasses import dataclass, field

from django.db import transaction
from django.db.models import Max, QuerySet
from django.urls import reverse

from apps.catalog.models import Course, Enrollment
from apps.catalog.permissions import can_edit_course
from apps.learning.models import CourseSection, Lesson, LessonProgress, Quiz


def is_enrolled(user, course: Course) -> bool:
    return user.is_authenticated and Enrollment.objects.filter(user=user, course=course).exists()


# ---------------------------------------------------------------------------
# Tryb nauki i podgląd autora
# ---------------------------------------------------------------------------
# Autor, nauczyciel kategorii i administrator mogą kurs edytować. Gdy są na niego zapisani
# (kupili go albo zapisali się na bezpłatny), domyślnie uczą się jak każdy uczeń: rozwiązują
# testy, obowiązują ich blokady, liczy się postęp i dyplom. Podgląd autora (wszystko otwarte,
# test bez wysyłania) włączają przełącznikiem — wybór trzymamy w sesji, osobno dla każdego kursu.
# Autor niezapisany na kurs zawsze widzi podgląd.

PREVIEW_SESSION_KEY = "learning_preview_courses"


def enrolled_as_learner(user, course: Course) -> bool:
    return course.is_published and course.status == Course.Status.APPROVED and is_enrolled(user, course)


def can_switch_mode(user, course: Course) -> bool:
    """Może przełączać się między nauką a podglądem autora."""
    return user.is_authenticated and can_edit_course(user, course) and enrolled_as_learner(user, course)


def preview_mode(request, course: Course) -> bool:
    """Czy bieżące żądanie ogląda kurs w trybie podglądu autora (bez blokad i bez wysyłania testów)."""
    user = request.user
    if not user.is_authenticated or not can_edit_course(user, course):
        return False
    if not enrolled_as_learner(user, course):
        return True
    return course.pk in request.session.get(PREVIEW_SESSION_KEY, [])


def set_preview_mode(request, course: Course, enabled: bool) -> None:
    chosen = set(request.session.get(PREVIEW_SESSION_KEY, []))
    (chosen.add if enabled else chosen.discard)(course.pk)
    request.session[PREVIEW_SESSION_KEY] = sorted(chosen)


def can_access_lesson(user, lesson: Lesson) -> bool:
    """Autor i moderatorzy zawsze; pozostali — w opublikowanym kursie po zapisie albo w podglądzie.
    Blokadę za testem obowiązkowym sprawdza ``build_outline`` (``entry.accessible``)."""
    course = lesson.section.course
    if can_edit_course(user, course):
        return True
    if not (course.is_published and course.status == Course.Status.APPROVED):
        return False
    return lesson.is_preview or is_enrolled(user, course)


@dataclass
class OutlineQuiz:
    """Test w programie. ``locked`` — za niezdanym testem obowiązkowym."""

    quiz: Quiz
    number: str
    passed: bool = False
    pending: bool = False  # czeka na ocenę nauczyciela
    accessible: bool = False
    locked: bool = False
    is_quiz: bool = True

    @property
    def title(self) -> str:
        return self.quiz.title

    @property
    def completed(self) -> bool:
        return self.passed

    @property
    def url(self) -> str:
        return reverse("learning:quiz_detail", kwargs={"slug": self.quiz.course.slug, "pk": self.quiz.pk})


@dataclass
class OutlineLesson:
    lesson: Lesson
    number: str
    completed: bool = False
    accessible: bool = False
    locked: bool = False
    quizzes: list[OutlineQuiz] = field(default_factory=list)  # testy stojące po tej lekcji
    is_quiz: bool = False

    @property
    def title(self) -> str:
        return self.lesson.title

    @property
    def url(self) -> str:
        return reverse("learning:lesson_detail", kwargs={"slug": self.lesson.section.course.slug, "pk": self.lesson.pk})


@dataclass
class OutlineSection:
    section: CourseSection
    number: str
    lessons: list[OutlineLesson] = field(default_factory=list)

    @property
    def minutes(self) -> int:
        return sum(item.lesson.duration_minutes for item in self.lessons)


@dataclass
class CourseOutline:
    sections: list[OutlineSection]
    final_quizzes: list[OutlineQuiz] = field(default_factory=list)  # testy na końcu kursu

    @property
    def lessons(self) -> list[OutlineLesson]:
        return [item for section in self.sections for item in section.lessons]

    @property
    def items(self) -> list[OutlineLesson | OutlineQuiz]:
        """Pełna ścieżka: lekcje i testy w kolejności, w jakiej uczeń je przechodzi."""
        sequence: list[OutlineLesson | OutlineQuiz] = []
        for lesson in self.lessons:
            sequence.append(lesson)
            sequence.extend(lesson.quizzes)
        sequence.extend(self.final_quizzes)
        return sequence

    @property
    def quizzes(self) -> list[OutlineQuiz]:
        return [item for item in self.items if item.is_quiz]

    @property
    def lesson_count(self) -> int:
        return len(self.lessons)

    @property
    def quiz_count(self) -> int:
        return len(self.quizzes)

    @property
    def minutes(self) -> int:
        return sum(section.minutes for section in self.sections)

    @property
    def completed_count(self) -> int:
        return sum(1 for item in self.lessons if item.completed)

    @property
    def passed_quiz_count(self) -> int:
        return sum(1 for item in self.quizzes if item.passed)

    @property
    def percent(self) -> int:
        """Postęp: ukończone lekcje + zdane testy."""
        total = self.lesson_count + self.quiz_count
        done = self.completed_count + self.passed_quiz_count
        return round(100 * done / total) if total else 0

    @property
    def is_complete(self) -> bool:
        return bool(self.lesson_count) and self.percent == 100

    def find(self, lesson: Lesson) -> OutlineLesson | None:
        return next((item for item in self.lessons if item.lesson.pk == lesson.pk), None)

    def find_quiz(self, quiz: Quiz) -> OutlineQuiz | None:
        return next((item for item in self.quizzes if item.quiz.pk == quiz.pk), None)

    def neighbours(self, current) -> tuple:
        """Poprzedni i następny element ścieżki (lekcja albo test)."""
        items = self.items
        is_quiz = isinstance(current, Quiz)
        index = next(
            (i for i, item in enumerate(items)
             if item.is_quiz == is_quiz and (item.quiz if is_quiz else item.lesson).pk == current.pk),
            None,
        )
        if index is None:
            return None, None
        previous = items[index - 1] if index > 0 else None
        following = items[index + 1] if index + 1 < len(items) else None
        return previous, following

    def suggested_next(self):
        """Pierwszy nieukończony i dostępny element (podpowiedź „kontynuuj”)."""
        return next((item for item in self.items if not item.completed and item.accessible), None)

    @property
    def blocking_quiz(self) -> OutlineQuiz | None:
        """Test obowiązkowy, który blokuje dalszą część kursu."""
        return next((item for item in self.quizzes if item.quiz.is_required and not item.passed and not item.locked), None)


def quiz_status(user, course: Course) -> dict[int, tuple[bool, bool]]:
    """{quiz_id: (zdany, czeka_na_ocenę)} dla ucznia."""
    from apps.learning.models import Attempt

    status: dict[int, tuple[bool, bool]] = {}
    if user is None or not user.is_authenticated:
        return status
    for quiz_id, passed, state in Attempt.objects.filter(user=user, quiz__course=course).values_list(
        "quiz_id", "passed", "status"
    ):
        was_passed, was_pending = status.get(quiz_id, (False, False))
        status[quiz_id] = (was_passed or passed, was_pending or state == Attempt.Status.PENDING_REVIEW)
    return status


def build_outline(course: Course, user=None, preview: bool | None = None) -> CourseOutline:
    """Program z dostępnością i postępem. ``preview`` — podgląd autora (bez blokad); domyślnie
    tylko dla autora/moderatora niezapisanego na kurs (zapisany uczy się jak uczeń)."""
    sections = course.sections.prefetch_related("lessons__attachments")
    quizzes = list(course.quizzes.select_related("course").prefetch_related("questions"))
    completed: set[int] = set()
    full_access = editor = False
    if user is not None and user.is_authenticated:
        completed = set(
            LessonProgress.objects.filter(user=user, lesson__section__course=course).values_list("lesson_id", flat=True)
        )
        learner = enrolled_as_learner(user, course)
        if preview is None:
            preview = can_edit_course(user, course) and not learner
        editor = bool(preview) and can_edit_course(user, course)
        full_access = editor or learner
    public = course.is_published and course.status == Course.Status.APPROVED
    statuses = quiz_status(user, course)

    def make_quiz(quiz: Quiz) -> OutlineQuiz:
        passed, pending = statuses.get(quiz.pk, (False, False))
        return OutlineQuiz(quiz=quiz, number="Test", passed=passed, pending=pending and not passed, accessible=full_access)

    lessons_quizzes: dict[int, list[OutlineQuiz]] = {}
    final: list[OutlineQuiz] = []
    for quiz in quizzes:
        (lessons_quizzes.setdefault(quiz.after_lesson_id, []) if quiz.after_lesson_id else final).append(make_quiz(quiz))

    outline = []
    for s_index, section in enumerate(sections, start=1):
        item = OutlineSection(section=section, number=f"{s_index:02d}")
        for l_index, lesson in enumerate(section.lessons.all(), start=1):
            item.lessons.append(
                OutlineLesson(
                    lesson=lesson,
                    number=f"{s_index}.{l_index}",
                    completed=lesson.pk in completed,
                    accessible=full_access or (lesson.is_preview and public),
                    quizzes=lessons_quizzes.get(lesson.pk, []),
                )
            )
        outline.append(item)
    result = CourseOutline(sections=outline, final_quizzes=final)

    # Blokada: za niezdanym testem obowiązkowym uczeń nie wchodzi dalej (podgląd autora — bez blokad)
    if not editor:
        blocked = False
        for entry in result.items:
            if blocked:
                entry.locked = True
                if entry.is_quiz or not (entry.lesson.is_preview and public):
                    entry.accessible = False
            elif entry.is_quiz and entry.quiz.is_required and not entry.passed:
                blocked = True
    return result


def sync_enrollment_progress(user, course: Course) -> None:
    """Zapisuje postęp (%) w zapisie na kurs; po ukończeniu całości wystawia dyplom."""
    enrollment = Enrollment.objects.filter(user=user, course=course).first()
    if enrollment is None:
        return
    outline = build_outline(course, user)
    enrollment.progress_percent = outline.percent
    if outline.is_complete:
        enrollment.save(update_fields=["progress_percent"])
        was_completed = enrollment.is_completed
        enrollment.mark_completed()
        if not was_completed:
            from apps.learning.quizzes import on_course_completed

            on_course_completed(user, course)
    else:
        was_completed = enrollment.is_completed
        enrollment.is_completed = False
        enrollment.completed_at = None
        enrollment.save(update_fields=["progress_percent", "is_completed", "completed_at"])
        if was_completed:
            course.recalculate_stats()  # liczba ukończeń w katalogu i statystykach


def next_order(queryset: QuerySet) -> int:
    return (queryset.aggregate(top=Max("order"))["top"] or 0) + 1


@transaction.atomic
def move(item, siblings: QuerySet, direction: str) -> bool:
    """Przesuwa element o jedno miejsce w górę/dół wśród rodzeństwa. Zwraca True, jeśli się ruszył."""
    ordered = list(siblings.order_by("order", "pk"))
    index = next(i for i, obj in enumerate(ordered) if obj.pk == item.pk)
    target = index - 1 if direction == "up" else index + 1
    if target < 0 or target >= len(ordered):
        return False
    ordered[index], ordered[target] = ordered[target], ordered[index]
    # Przenumerowanie 1..n usuwa ewentualne duplikaty kolejności
    for position, obj in enumerate(ordered, start=1):
        if obj.order != position:
            obj.order = position
            obj.save(update_fields=["order"])
    return True
