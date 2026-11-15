"""Panele startowe po zalogowaniu: uczeń, nauczyciel (moderator) i administrator.

Strona główna gościa to prezentacja platformy; zalogowany dostaje panel swojej roli
(``core:home``). Prezentacja pozostaje dostępna pod ``core:about``.
"""

from __future__ import annotations

from django.contrib.auth import get_user_model
from django.db.models import Count, Q, Sum
from django.utils import timezone

from apps.catalog.comments import is_moderator_somewhere, pending_comments_for
from apps.catalog.models import Category, Course, Enrollment
from apps.catalog.moderation_views import pending_submissions_for
from apps.catalog.permissions import is_platform_admin
from apps.core.models import Notification

PREVIEW_LIMIT = 3


def dashboard_role(user) -> str:
    """Który panel pokazać: ``admin`` › ``teacher`` › ``student``."""
    if is_platform_admin(user):
        return "admin"
    if getattr(user, "role", "") == "moderator":
        return "teacher"
    return "student"


def _common(user) -> dict:
    return {
        "unread_count": Notification.objects.filter(user=user, is_read=False).count(),
        "recent_unread": Notification.objects.filter(user=user, is_read=False).order_by("-created_at")[:PREVIEW_LIMIT],
    }


def _queues(user) -> dict:
    """Kolejki moderacji z liczbą i najstarszymi pozycjami (nauczyciel — swoje kategorie, admin — wszystko)."""
    from apps.learning.quizzes import gradable_attempts

    submissions = pending_submissions_for(user).order_by("updated_at")
    comments = pending_comments_for(user)
    attempts = gradable_attempts(user).order_by("submitted_at")
    return {
        "submission_count": submissions.count(),
        "submissions": submissions[:PREVIEW_LIMIT],
        "comment_count": comments.count(),
        "comments": comments[:PREVIEW_LIMIT],
        "grading_count": attempts.count(),
        "attempts": attempts[:PREVIEW_LIMIT],
    }


def student_context(user) -> dict:
    from apps.learning.curriculum import build_outline
    from apps.learning.models import Attempt, Certificate
    from apps.learning.styles import recommend_courses

    enrollments = Enrollment.objects.filter(user=user).select_related("course", "course__category")
    in_progress = list(enrollments.filter(is_completed=False).order_by("-created_at")[:PREVIEW_LIMIT])
    for enrollment in in_progress:
        enrollment.next_lesson = build_outline(enrollment.course, user).suggested_next()
    attempts = Attempt.objects.filter(user=user).select_related("quiz", "quiz__course").order_by("-submitted_at")
    return {
        "in_progress": in_progress,
        "in_progress_count": enrollments.filter(is_completed=False).count(),
        "completed_count": enrollments.filter(is_completed=True).count(),
        "certificate_count": Certificate.objects.filter(user=user).count(),
        "recent_attempts": attempts[:PREVIEW_LIMIT],
        "pending_attempt_count": attempts.filter(status=Attempt.Status.PENDING_REVIEW).count(),
        "recommended": recommend_courses(user, limit=PREVIEW_LIMIT),
        "proposed_count": Course.objects.filter(author=user).count(),
    }


def teacher_context(user) -> dict:
    authored = (
        Course.objects.filter(author=user)
        .select_related("category")
        .annotate(
            student_total=Count("enrollments", distinct=True),
            done_total=Count("enrollments", filter=Q(enrollments__is_completed=True), distinct=True),
        )
        .order_by("-updated_at")
    )
    categories = (
        user.moderated_categories.filter(is_active=True).order_by("tree_id", "lft")
        if is_moderator_somewhere(user)
        else Category.objects.none()
    )
    category_rows = []
    for category in categories:
        branch = category.get_descendants(include_self=True)
        category_rows.append(
            {
                "category": category,
                "subcategory_count": branch.count() - 1,
                "course_count": Course.objects.published().filter(category__in=branch).count(),
            }
        )
    return {
        **_queues(user),
        "authored": authored[:5],
        "authored_count": authored.count(),
        "student_total": Enrollment.objects.filter(course__author=user).count(),
        "category_rows": category_rows,
        "learning_count": Enrollment.objects.filter(user=user, is_completed=False).count(),
    }


def admin_context(user) -> dict:
    from apps.orders.models import Order

    User = get_user_model()
    month_ago = timezone.now() - timezone.timedelta(days=30)
    paid = Order.objects.filter(status__in=[Order.Status.PAID, Order.Status.COMPLETED])

    roots = Category.objects.filter(parent__isnull=True).prefetch_related("moderators").order_by("order", "name")
    published = Course.objects.published()
    category_rows = []
    for category in roots:
        branch = category.get_descendants(include_self=True)
        category_rows.append(
            {
                "category": category,
                "subcategory_count": branch.count() - 1,
                "course_count": published.filter(category__in=branch).count(),
                "teachers": list(category.moderators.all()),
            }
        )
    return {
        **_queues(user),
        "user_total": User.objects.filter(is_active=True).count(),
        "user_new_30": User.objects.filter(date_joined__gte=month_ago).count(),
        "course_published": published.count(),
        "category_total": Category.objects.count(),
        "category_rows": category_rows,
        "uncovered_count": sum(1 for row in category_rows if not row["teachers"]),
        "orders_to_fulfil": Order.objects.filter(status=Order.Status.PAID).count(),
        "revenue_30": paid.filter(paid_at__gte=month_ago).aggregate(total=Sum("total"))["total"] or 0,
        "recent_orders": Order.objects.select_related("user").order_by("-created_at")[:5],
        "recent_users": User.objects.order_by("-date_joined")[:5],
    }


def dashboard_context(user) -> dict:
    role = dashboard_role(user)
    builder = {"admin": admin_context, "teacher": teacher_context, "student": student_context}[role]
    return {"dashboard_role": role, **_common(user), **builder(user)}
