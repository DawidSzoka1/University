"""Moderacja: kolejka komentarzy, kolejka zgłoszeń kursów i trwały wątek zgłoszenia."""

from __future__ import annotations

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db.models import Count, Max
from django.http import Http404, HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.decorators.http import require_POST

from apps.catalog.comments import is_moderator_somewhere, moderated_category_ids, pending_comments_for
from apps.catalog.forms import CourseSubmissionMessageForm, SubmissionRejectForm
from apps.catalog.models import Course, CourseSubmissionMessage
from apps.catalog.permissions import course_reviewers, moderates_category
from apps.core.models import Notification
from apps.core.notifications import notify, notify_many
from apps.core.pagination import paginate_queryset


def pending_submissions_for(user):
    qs = Course.objects.filter(status=Course.Status.PENDING).select_related("author", "category")
    ids = moderated_category_ids(user)
    if ids is not None:
        qs = qs.filter(category_id__in=ids)
    return qs


def _require_moderator(user) -> None:
    if not is_moderator_somewhere(user):
        raise Http404("Ta strona jest dostępna dla nauczycieli (moderatorów) i administratorów.")


def _queue_counts(user) -> dict:
    from apps.learning.quizzes import gradable_attempts
    from apps.orders.models import Order

    return {
        "pending_comment_count": pending_comments_for(user).count(),
        "pending_submission_count": pending_submissions_for(user).count(),
        "pending_grading_count": gradable_attempts(user).count(),
        "paid_order_count": Order.objects.filter(status=Order.Status.PAID).count(),
    }


@login_required
def moderation_comments(request: HttpRequest) -> HttpResponse:
    """Komentarze czekające na akceptację w kategoriach nauczyciela (najstarsze pierwsze)."""
    _require_moderator(request.user)
    page_obj, paginator = paginate_queryset(request, pending_comments_for(request.user), per_page=10)
    return render(
        request,
        "catalog/moderation_comments.html",
        {"page_obj": page_obj, "paginator": paginator, "comments": page_obj.object_list, **_queue_counts(request.user)},
    )


@login_required
def moderation_submissions(request: HttpRequest) -> HttpResponse:
    """Kursy wysłane do akceptacji w kategoriach nauczyciela."""
    _require_moderator(request.user)
    courses = pending_submissions_for(request.user).annotate(
        message_count=Count("submission_messages"),
        last_message_at=Max("submission_messages__created_at"),
    ).order_by("updated_at")
    page_obj, paginator = paginate_queryset(request, courses, per_page=10)
    return render(
        request,
        "catalog/moderation_submissions.html",
        {"page_obj": page_obj, "paginator": paginator, "courses": page_obj.object_list, **_queue_counts(request.user)},
    )


def _thread_course(request: HttpRequest, slug: str) -> tuple[Course, bool]:
    """Wątek widzi autor kursu i moderatorzy jego kategorii. Zwraca (kurs, czy_moderator)."""
    course = get_object_or_404(Course.objects.select_related("category", "author"), slug=slug)
    is_moderator = moderates_category(request.user, course.category)
    if not (is_moderator or course.author_id == request.user.pk):
        raise Http404("Zgłoszenie nie istnieje albo nie masz do niego dostępu.")
    return course, is_moderator


def _render_thread(request: HttpRequest, course: Course, is_moderator: bool, *,
                   form: CourseSubmissionMessageForm | None = None,
                   reject_form: SubmissionRejectForm | None = None) -> HttpResponse:
    thread = course.submission_messages.select_related("author")
    page_obj, paginator = paginate_queryset(request, thread, per_page=20)
    if "page" not in request.GET and paginator.num_pages > 1:
        page_obj = paginator.page(paginator.num_pages)  # domyślnie najnowsze wiadomości
    return render(
        request,
        "catalog/submission_thread.html",
        {
            "course": course,
            "is_moderator": is_moderator,
            "can_decide": is_moderator and course.status == Course.Status.PENDING,
            "messages_page": page_obj.object_list,
            "page_obj": page_obj,
            "paginator": paginator,
            "form": form or CourseSubmissionMessageForm(),
            "reject_form": reject_form or SubmissionRejectForm(),
        },
    )


def _thread_url(course: Course) -> str:
    return reverse("catalog:submission_thread", kwargs={"slug": course.slug})


@login_required
def submission_thread(request: HttpRequest, slug: str) -> HttpResponse:
    """Dyskusja autora z nauczycielami kategorii — zapisana w bazie, widoczna po ponownym logowaniu."""
    course, is_moderator = _thread_course(request, slug)
    if request.method != "POST":
        return _render_thread(request, course, is_moderator)

    form = CourseSubmissionMessageForm(request.POST)
    if not form.is_valid():
        messages.error(request, "Wpisz treść wiadomości.")
        return _render_thread(request, course, is_moderator, form=form)

    message = form.save(commit=False)
    message.course = course
    message.author = request.user
    message.save()

    name = request.user.get_full_name() or request.user.username
    url = _thread_url(course) + f"#message-{message.pk}"
    if course.author_id == request.user.pk:
        notify_many(
            course_reviewers(course),
            Notification.Kind.SUBMISSION_MESSAGE,
            "Nowa wiadomość w zgłoszeniu kursu",
            f"{name} pisze w zgłoszeniu „{course.title}”.",
            url=url,
            exclude=request.user,
        )
    elif course.author_id:
        notify(
            course.author,
            Notification.Kind.SUBMISSION_MESSAGE,
            "Nowa wiadomość od moderatora",
            f"{name} odpisuje w zgłoszeniu „{course.title}”.",
            url=url,
        )
    messages.success(request, "Wysłano wiadomość.")
    return redirect(url)


@login_required
@require_POST
def submission_decide(request: HttpRequest, slug: str) -> HttpResponse:
    """Nauczyciel kategorii akceptuje zgłoszenie albo odrzuca je z podanym powodem."""
    course, is_moderator = _thread_course(request, slug)
    if not is_moderator or course.status != Course.Status.PENDING:
        raise Http404("Tego zgłoszenia nie można teraz rozpatrzyć.")

    action = request.POST.get("action")
    if action == "approve":
        course.status = Course.Status.APPROVED
        course.is_published = True
        course.rejection_reason = ""
        course.save()
        if course.author_id:
            notify(
                course.author,
                Notification.Kind.SUBMISSION_STATUS,
                "Kurs zaakceptowany",
                f"„{course.title}” jest już w katalogu.",
                url=reverse("catalog:course_detail", kwargs={"slug": course.slug}),
            )
        messages.success(request, f"Zaakceptowano kurs „{course.title}” — jest już w katalogu.")
        return redirect("catalog:moderation_submissions")

    if action != "reject":
        messages.error(request, "Wybierz: zaakceptuj albo odrzuć.")
        return redirect(_thread_url(course))

    reject_form = SubmissionRejectForm(request.POST)
    if not reject_form.is_valid():
        messages.error(request, "Podaj powód odrzucenia.")
        return _render_thread(request, course, is_moderator, reject_form=reject_form)

    reason = reject_form.cleaned_data["reason"]
    course.status = Course.Status.REJECTED
    course.is_published = False
    course.rejection_reason = reason
    course.save()
    CourseSubmissionMessage.objects.create(course=course, author=request.user, content=f"Zgłoszenie odrzucone. Powód: {reason}")
    if course.author_id:
        notify(
            course.author,
            Notification.Kind.SUBMISSION_STATUS,
            "Zgłoszenie kursu odrzucone",
            f"„{course.title}”: {reason}",
            url=_thread_url(course),
        )
    messages.success(request, f"Odrzucono zgłoszenie „{course.title}”. Autor dostał powiadomienie z powodem.")
    return redirect("catalog:moderation_submissions")
