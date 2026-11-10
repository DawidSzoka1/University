"""Komentarze pod kursem: dodawanie, odpowiedzi, edycja, usuwanie, akceptacja i głosy."""

from __future__ import annotations

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import Http404, HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils import timezone
from django.utils.http import url_has_allowed_host_and_scheme
from django.views.decorators.http import require_POST

from apps.catalog.comments import (
    approve_comment,
    can_delete_comment,
    can_edit_comment,
    can_moderate_comment,
    can_rate_course,
    can_reply_to,
    can_vote_on,
    comment_url,
    notify_pending,
    notify_reply,
    remove_comment,
    toggle_vote,
    vote_counts,
)
from apps.catalog.forms import CommentForm
from apps.catalog.models import RATING_CHOICES, Comment, Course
from apps.catalog.permissions import moderates_category
from apps.core.models import Notification
from apps.core.notifications import notify


def _safe_next(request: HttpRequest, fallback: str) -> str:
    target = request.POST.get("next") or request.GET.get("next") or ""
    if target and url_has_allowed_host_and_scheme(target, allowed_hosts={request.get_host()}):
        return target
    return fallback


def _wants_json(request: HttpRequest) -> bool:
    return "application/json" in request.headers.get("Accept", "")


def _get_comment(pk: int) -> Comment:
    return get_object_or_404(
        Comment.objects.select_related("course", "course__category", "author", "parent__author"), pk=pk
    )


def _render_form(request: HttpRequest, form: CommentForm, *, course: Course, mode: str,
                 comment: Comment | None = None, parent: Comment | None = None) -> HttpResponse:
    """Pełna strona formularza: edycja bez JS i powrót po błędzie walidacji (z wpisaną treścią)."""
    return render(
        request,
        "catalog/comment_form.html",
        {
            "form": form,
            "course": course,
            "mode": mode,
            "comment": comment,
            "parent": parent,
            "rating_choices": RATING_CHOICES,
        },
    )


@login_required
@require_POST
def comment_create(request: HttpRequest, slug: str) -> HttpResponse:
    """Nowy komentarz albo odpowiedź (pole ``parent``). Moderator kategorii publikuje od razu."""
    course = get_object_or_404(Course.objects.published().select_related("category"), slug=slug)
    parent = None
    if request.POST.get("parent"):
        parent = Comment.objects.select_related("author", "course").filter(
            pk=request.POST["parent"], course=course
        ).first()
        if parent is None or not can_reply_to(request.user, parent):
            raise Http404("Nie można odpowiedzieć na ten komentarz.")

    allow_rating = parent is None and can_rate_course(request.user, course)
    form = CommentForm(request.POST, allow_rating=allow_rating, is_reply=parent is not None)
    if not form.is_valid():
        messages.error(request, "Popraw treść komentarza.")
        return _render_form(request, form, course=course, mode="reply" if parent else "create", parent=parent)

    comment = form.save(commit=False)
    comment.course = course
    comment.author = request.user
    comment.parent = parent
    if moderates_category(request.user, course.category):
        comment.is_approved = True
        comment.approved_by = request.user
        comment.approved_at = timezone.now()
    comment.save()

    if comment.is_approved:
        notify_reply(comment)
        messages.success(request, "Dodano odpowiedź." if parent else "Dodano komentarz.")
    else:
        notify_pending(comment)
        messages.success(
            request,
            "Wysłano komentarz. Pojawi się publicznie po akceptacji nauczyciela — do tego czasu widzisz go tylko Ty.",
        )
    return redirect(comment_url(comment))


@login_required
def comment_edit(request: HttpRequest, pk: int) -> HttpResponse:
    """Autor zmienia swój komentarz. Zmieniony zatwierdzony komentarz wraca do akceptacji."""
    comment = _get_comment(pk)
    if not can_edit_comment(request.user, comment):
        raise Http404("Nie możesz edytować tego komentarza.")
    course = comment.course
    allow_rating = comment.parent_id is None and (
        comment.rating is not None or can_rate_course(request.user, course, exclude=comment)
    )

    if request.method == "POST":
        form = CommentForm(request.POST, instance=comment, allow_rating=allow_rating, is_reply=comment.parent_id is not None)
        if form.is_valid():
            if not form.has_changed():
                messages.info(request, "Komentarz bez zmian.")
                return redirect(comment_url(comment))
            comment = form.save(commit=False)
            comment.edited_at = timezone.now()
            needs_review = comment.is_approved and not moderates_category(request.user, course.category)
            if needs_review:
                comment.is_approved = False
                comment.approved_by = None
                comment.approved_at = None
            comment.save()
            if needs_review:
                notify_pending(comment)
                messages.warning(request, "Zapisano zmiany. Komentarz wróci na stronę po ponownej akceptacji nauczyciela.")
            else:
                messages.success(request, "Zapisano zmiany w komentarzu.")
            return redirect(comment_url(comment))
        messages.error(request, "Popraw treść komentarza.")
    else:
        form = CommentForm(instance=comment, allow_rating=allow_rating, is_reply=comment.parent_id is not None)
    return _render_form(request, form, course=course, mode="edit", comment=comment, parent=comment.parent)


@login_required
@require_POST
def comment_delete(request: HttpRequest, pk: int) -> HttpResponse:
    comment = _get_comment(pk)
    if not can_delete_comment(request.user, comment):
        raise Http404("Nie możesz usunąć tego komentarza.")
    course_url = reverse("catalog:course_detail", kwargs={"slug": comment.course.slug}) + "#comments"
    by_moderator = comment.author_id != request.user.pk
    if by_moderator and comment.author_id:
        excerpt = comment.content if len(comment.content) <= 80 else comment.content[:77] + "…"
        notify(
            comment.author,
            Notification.Kind.COMMENT_REJECTED,
            "Komentarz odrzucony przez moderatora" if not comment.is_approved else "Komentarz usunięty przez moderatora",
            f"Kurs „{comment.course.title}”: „{excerpt}”",
            url=course_url,
        )
    kept_marker = remove_comment(comment)
    if kept_marker:
        messages.success(request, "Usunięto komentarz. Odpowiedzi innych osób zostały w wątku.")
    elif by_moderator:
        messages.success(request, "Usunięto komentarz. Autor dostał powiadomienie.")
    else:
        messages.success(request, "Usunięto komentarz.")
    return redirect(_safe_next(request, course_url))


@login_required
@require_POST
def comment_approve(request: HttpRequest, pk: int) -> HttpResponse:
    """Akceptacja jednym kliknięciem — pod kursem albo w kolejce moderacji."""
    comment = _get_comment(pk)
    if comment.is_removed or not can_moderate_comment(request.user, comment):
        raise Http404("Nie możesz zatwierdzić tego komentarza.")
    if comment.is_approved:
        messages.info(request, "Ten komentarz jest już zatwierdzony.")
    else:
        approve_comment(comment, request.user)
        messages.success(request, "Zatwierdzono komentarz — jest już widoczny dla wszystkich.")
    return redirect(_safe_next(request, comment_url(comment)))


@login_required
@require_POST
def comment_vote(request: HttpRequest, pk: int) -> HttpResponse:
    """Lubię / nie lubię. Ponowne kliknięcie tego samego cofa głos. JSON dla fetch, redirect bez JS."""
    comment = _get_comment(pk)
    try:
        value = int(request.POST.get("value", ""))
    except ValueError:
        value = 0
    if value not in (1, -1):
        if _wants_json(request):
            return JsonResponse({"error": "Nieprawidłowy głos."}, status=400)
        messages.error(request, "Nieprawidłowy głos.")
        return redirect(comment_url(comment))

    if not can_vote_on(request.user, comment):
        error = (
            "Nie możesz oceniać własnego komentarza."
            if comment.author_id == request.user.pk
            else "Tego komentarza nie można oceniać."
        )
        if _wants_json(request):
            return JsonResponse({"error": error}, status=403)
        messages.error(request, error)
        return redirect(comment_url(comment))

    vote = toggle_vote(comment, request.user, value)
    likes, dislikes = vote_counts(comment)
    if _wants_json(request):
        return JsonResponse({"vote": vote, "likes": likes, "dislikes": dislikes})
    return redirect(comment_url(comment))
