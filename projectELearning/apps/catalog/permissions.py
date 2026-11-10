"""Uprawnienia katalogu: kto publikuje od razu, kto wysyła zgłoszenie, kto edytuje."""

from __future__ import annotations

from apps.catalog.models import Category, Course


def is_platform_admin(user) -> bool:
    """Administrator platformy (rola admin) albo obsługa Django."""
    return bool(
        user.is_authenticated
        and (user.is_superuser or user.is_staff or getattr(user, "role", "") == "admin")
    )


def moderates_category(user, category: Category | None) -> bool:
    """Admin moderuje wszystko; nauczyciel (moderator) — przypisane kategorie i ich podkategorie."""
    if category is None or not user.is_authenticated:
        return False
    if is_platform_admin(user):
        return True
    if getattr(user, "role", "") != "moderator":
        return False
    return category.get_ancestors(include_self=True).filter(moderators=user).exists()


def can_edit_course(user, course: Course) -> bool:
    """Autor kursu albo moderator jego kategorii."""
    if not user.is_authenticated:
        return False
    return course.author_id == user.pk or moderates_category(user, course.category)


def can_view_course(user, course: Course) -> bool:
    """Opublikowany i zatwierdzony kurs widzą wszyscy; pozostałe — autor i moderatorzy."""
    if course.is_published and course.status == Course.Status.APPROVED:
        return True
    return can_edit_course(user, course)


def apply_edit_rules(course: Course, user) -> bool:
    """Zmiana danych lub treści kursu. Zwraca True, gdy kurs wrócił do akceptacji.

    Szkic, zgłoszenie i kurs odrzucony zachowują status — wysyła je dopiero autor
    (``submit_course``). Opublikowany kurs zmieniony przez kogoś, kto nie moderuje
    jego kategorii (np. ucznia-autora), wraca do akceptacji.
    """
    public = course.is_published and course.status == Course.Status.APPROVED
    if public and not moderates_category(user, course.category):
        course.status = Course.Status.PENDING
        course.is_published = False
        return True
    return False


def course_has_lessons(course: Course) -> bool:
    from apps.learning.models import Lesson

    return Lesson.objects.filter(section__course=course).exists()


def can_submit_course(user, course: Course) -> bool:
    """Wysłać/opublikować szkic (albo poprawiony kurs odrzucony) może jego autor lub admin."""
    if course.status not in {Course.Status.DRAFT, Course.Status.REJECTED}:
        return False
    return user.is_authenticated and (course.author_id == user.pk or is_platform_admin(user))


def submit_course(course: Course, user) -> bool:
    """Wysyła kurs. Nauczyciel swojej kategorii i admin publikują od razu (zwraca True),
    pozostali wysyłają zgłoszenie do akceptacji (zwraca False)."""
    course.rejection_reason = ""
    if moderates_category(user, course.category):
        course.status = Course.Status.APPROVED
        course.is_published = True
        return True
    course.status = Course.Status.PENDING
    course.is_published = False
    return False


def course_reviewers(course: Course):
    """Nauczyciele kategorii kursu (i jej kategorii nadrzędnych); gdy ich brak — administratorzy."""
    from django.contrib.auth import get_user_model
    from django.db.models import Q

    User = get_user_model()
    if course.category_id:
        ancestors = course.category.get_ancestors(include_self=True)
        teachers = User.objects.filter(role="moderator", moderated_categories__in=ancestors, is_active=True).distinct()
        if teachers.exists():
            return teachers
    return User.objects.filter(Q(role="admin") | Q(is_superuser=True), is_active=True).distinct()


# --------------------------------------------------------------------- kategorie
def manageable_categories(user):
    """Kategorie widoczne w zarządzaniu: admin — wszystkie; nauczyciel — przypisane z podkategoriami."""
    if is_platform_admin(user):
        return Category.objects.all()
    if not user.is_authenticated or getattr(user, "role", "") != "moderator":
        return Category.objects.none()
    ids: set[int] = set()
    for assigned in user.moderated_categories.all():
        ids.update(assigned.get_descendants(include_self=True).values_list("id", flat=True))
    return Category.objects.filter(id__in=ids)


def can_add_category(user, parent: Category | None) -> bool:
    """Kategorie główne dodaje admin; podkategorię — admin albo nauczyciel tej gałęzi."""
    if parent is None:
        return is_platform_admin(user)
    return moderates_category(user, parent)


def can_manage_category(user, category: Category) -> bool:
    """Edycja, kolejność, ukrywanie, usuwanie: admin — każda; nauczyciel — tylko podkategorie
    poniżej przydzielonej mu kategorii (samą przydzieloną zmienia admin)."""
    if is_platform_admin(user):
        return True
    if getattr(user, "role", "") != "moderator" or category.parent_id is None:
        return False
    return category.get_ancestors().filter(moderators=user).exists()
