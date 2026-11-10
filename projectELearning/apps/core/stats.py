"""Statystyki platformy dla administratora: użytkownicy, kursy i ukończenia, testy, style, asystent, sklep."""

from collections import Counter

from django.contrib.auth import get_user_model
from django.contrib.auth.decorators import login_required
from django.db.models import Count, Q, Sum
from django.http import Http404
from django.shortcuts import render
from django.utils import timezone

from apps.catalog.models import Course, Enrollment
from apps.catalog.moderation_views import _queue_counts
from apps.catalog.permissions import is_platform_admin
from apps.core.pagination import paginate_queryset


def _percent(part: int, whole: int) -> int:
    return round(100 * part / whole) if whole else 0


@login_required
def platform_stats(request):
    if not is_platform_admin(request.user):
        raise Http404("Statystyki są dostępne dla administratora.")
    from apps.chatbot.models import ChatMessage, ChatSession
    from apps.learning.models import Attempt, LearningStyle, LessonVariant, SurveyResponse
    from apps.orders.models import Order

    User = get_user_model()
    month_ago = timezone.now() - timezone.timedelta(days=30)
    roles = dict(User.Role.choices)
    role_counts = Counter(User.objects.filter(is_active=True).values_list("role", flat=True))
    users = {
        "total": sum(role_counts.values()),
        "new_30": User.objects.filter(date_joined__gte=month_ago).count(),
        "roles": [(label, role_counts.get(key, 0)) for key, label in roles.items()],
    }

    status_labels = dict(Course.Status.choices)
    course_status = Counter(Course.objects.values_list("status", flat=True))
    enrollments_total = Enrollment.objects.count()
    enrollments_done = Enrollment.objects.filter(is_completed=True).count()
    courses = {
        "published": Course.objects.published().count(),
        "statuses": [(label, course_status.get(key, 0)) for key, label in status_labels.items()],
        "started": enrollments_total,
        "completed": enrollments_done,
        "in_progress": enrollments_total - enrollments_done,
        "rate": _percent(enrollments_done, enrollments_total),
    }

    per_course = (
        Course.objects.annotate(
            started=Count("enrollments", distinct=True),
            done=Count("enrollments", filter=Q(enrollments__is_completed=True), distinct=True),
        )
        .filter(started__gt=0)
        .order_by("-started", "title")
    )
    course_page, _ = paginate_queryset(request, per_course, per_page=10, page_param="kursy")
    for course in course_page.object_list:
        course.rate = _percent(course.done, course.started)

    graded = Attempt.objects.filter(status=Attempt.Status.GRADED)
    graded_count = graded.count()
    quizzes = {
        "graded": graded_count,
        "passed": graded.filter(passed=True).count(),
        "pass_rate": _percent(graded.filter(passed=True).count(), graded_count),
        "pending": Attempt.objects.filter(status=Attempt.Status.PENDING_REVIEW).count(),
    }

    style_rows = []
    for style in LearningStyle.objects.annotate(
        learner_total=Count("learners", distinct=True),
        survey_total=Count("survey_wins", distinct=True),
        variant_total=Count("lesson_variants", distinct=True),
    ):
        style_rows.append(style)
    styles = {
        "rows": style_rows,
        "without": User.objects.filter(role=User.Role.STUDENT, learning_style__isnull=True).count(),
        "surveys": SurveyResponse.objects.count(),
        "surveyed_users": SurveyResponse.objects.values("user").distinct().count(),
        "variants": LessonVariant.objects.count(),
    }

    assistant = {
        "sessions": ChatSession.objects.count(),
        "questions": ChatMessage.objects.filter(sender=ChatMessage.Sender.USER).count(),
        "refusals": ChatMessage.objects.filter(is_refusal=True).count(),
        "users": ChatSession.objects.values("student").distinct().count(),
    }

    order_labels = dict(Order.Status.choices)
    order_status = Counter(Order.objects.values_list("status", flat=True))
    revenue = Order.objects.filter(status__in=[Order.Status.PAID, Order.Status.COMPLETED]).aggregate(s=Sum("total"))["s"] or 0
    orders = {
        "statuses": [(label, order_status.get(key, 0)) for key, label in order_labels.items()],
        "total": sum(order_status.values()),
        "revenue": revenue,
    }

    return render(
        request,
        "core/stats.html",
        {
            "users": users,
            "courses": courses,
            "course_page": course_page,
            "quizzes": quizzes,
            "styles": styles,
            "assistant": assistant,
            "orders": orders,
            "active": "stats",
            **_queue_counts(request.user),
        },
    )
