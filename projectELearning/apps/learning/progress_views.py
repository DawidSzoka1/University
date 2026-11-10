"""Panel nauczyciela: uczniowie kursu — postęp, wyniki testów, styl uczenia się i ankieta."""

from collections import Counter, defaultdict

from django.contrib.auth.decorators import login_required
from django.db.models import Avg
from django.shortcuts import render

from apps.catalog.models import Enrollment
from apps.core.pagination import paginate_queryset
from apps.learning.models import Attempt, LearningStyle, Lesson, LessonProgress, Quiz, SurveyResponse
from apps.learning.views import _editable_course

FILTERS = {
    "": "Wszyscy",
    "w-trakcie": "W trakcie",
    "ukonczyli": "Ukończyli",
}


@login_required
def course_students(request, slug: str):
    course = _editable_course(request, slug)
    enrollments = Enrollment.objects.filter(course=course).select_related("user", "user__learning_style")
    started = enrollments.count()
    completed = enrollments.filter(is_completed=True).count()
    average_progress = enrollments.aggregate(avg=Avg("progress_percent"))["avg"] or 0

    graded = Attempt.objects.filter(quiz__course=course, status=Attempt.Status.GRADED, max_score__gt=0)
    graded_percents = [a.percent for a in graded.only("score", "max_score")]
    pending = Attempt.objects.filter(quiz__course=course, status=Attempt.Status.PENDING_REVIEW).count()

    style_counts = Counter(e.user.learning_style if e.user.learning_style and e.user.learning_style.is_active else None for e in enrollments)
    style_rows = [(style, style_counts.get(style, 0)) for style in LearningStyle.objects.filter(is_active=True)]
    no_style = style_counts.get(None, 0)

    current = request.GET.get("pokaz", "")
    if current == "w-trakcie":
        enrollments = enrollments.filter(is_completed=False)
    elif current == "ukonczyli":
        enrollments = enrollments.filter(is_completed=True)
    else:
        current = ""
    page_obj, paginator = paginate_queryset(request, enrollments.order_by("-created_at", "-pk"), per_page=20)

    rows = list(page_obj.object_list)
    user_ids = [e.user_id for e in rows]
    lesson_total = Lesson.objects.filter(section__course=course).count()
    quiz_total = Quiz.objects.filter(course=course).count()
    lessons_done = Counter(
        LessonProgress.objects.filter(user_id__in=user_ids, lesson__section__course=course).values_list("user_id", flat=True)
    )
    passed = defaultdict(set)
    scores = defaultdict(list)
    for attempt in Attempt.objects.filter(user_id__in=user_ids, quiz__course=course).only(
        "user_id", "quiz_id", "passed", "status", "score", "max_score"
    ):
        if attempt.passed:
            passed[attempt.user_id].add(attempt.quiz_id)
        if attempt.status == Attempt.Status.GRADED and attempt.max_score:
            scores[attempt.user_id].append(attempt.percent)
    surveys = {}
    for response in SurveyResponse.objects.filter(user_id__in=user_ids).select_related("top_style").order_by("user_id", "-created_at"):
        surveys.setdefault(response.user_id, response)
    for enrollment in rows:
        uid = enrollment.user_id
        enrollment.lessons_done = lessons_done.get(uid, 0)
        enrollment.quizzes_passed = len(passed.get(uid, ()))
        user_scores = scores.get(uid, [])
        enrollment.average_score = round(sum(user_scores) / len(user_scores)) if user_scores else None
        enrollment.survey = surveys.get(uid)

    return render(
        request,
        "learning/course_students.html",
        {
            "course": course,
            "started": started,
            "completed": completed,
            "completion_rate": round(100 * completed / started) if started else 0,
            "average_progress": round(average_progress),
            "average_score": round(sum(graded_percents) / len(graded_percents)) if graded_percents else None,
            "pending": pending,
            "style_rows": style_rows,
            "no_style": no_style,
            "filters": FILTERS,
            "current": current,
            "rows": rows,
            "page_obj": page_obj,
            "paginator": paginator,
            "lesson_total": lesson_total,
            "quiz_total": quiz_total,
        },
    )
