"""Style uczenia się: wybór stylu przez ucznia i wersje lekcji dla stylów (autor kursu)."""

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils.http import url_has_allowed_host_and_scheme
from django.views.decorators.http import require_POST

from apps.learning.forms import LearningStyleChoiceForm, LessonVariantForm
from apps.learning.models import LearningStyle, Lesson, LessonVariant
from apps.learning.styles import (
    active_styles,
    analyse,
    latest_survey,
    recommend_courses,
    set_user_style,
    submit_survey,
    survey_profile,
    survey_questions,
    user_style,
)
from apps.learning.views import _content_changed, _editable_course


def _safe_next(request, fallback: str) -> str:
    target = request.POST.get("next") or request.GET.get("next") or ""
    if target and url_has_allowed_host_and_scheme(target, {request.get_host()}, require_https=request.is_secure()):
        return target
    return fallback


@login_required
def learning_style(request):
    """Wybór własnego stylu (w każdej chwili) i analiza wyników z rekomendacją."""
    here = reverse("learning:learning_style")
    if request.method == "POST":
        form = LearningStyleChoiceForm(request.POST)
        if form.is_valid():
            style = form.cleaned_data["style"]
            set_user_style(request.user, style)
            if style:
                messages.success(request, f"Twój styl uczenia się: {style.name}. Lekcje z wersjami pokażą treść dla tego stylu.")
            else:
                messages.info(request, "Nie masz wybranego stylu — lekcje pokazują wersję podstawową.")
            return redirect(_safe_next(request, here))
        messages.error(request, "Wybierz styl z listy.")
    else:
        form = LearningStyleChoiceForm(initial={"style": user_style(request.user)})
    survey = latest_survey(request.user)
    return render(
        request,
        "learning/learning_style.html",
        {
            "form": form,
            "styles": active_styles(),
            "current": user_style(request.user),
            "analysis": analyse(request.user),
            "survey": survey,
            "survey_shares": survey_profile(survey),
            "recommended": recommend_courses(request.user, limit=3),
            "next": _safe_next(request, ""),
        },
    )


@login_required
def style_survey(request):
    """Ankieta diagnostyczna: każde pytanie wymaga jednej odpowiedzi; wynik → dominujący styl."""
    questions = survey_questions()
    errors: dict[int, str] = {}
    selected: dict[int, int] = {}
    if request.method == "POST":
        picked = {}
        for question in questions:
            raw = request.POST.get(f"q{question.pk}", "")
            option = next((o for o in question.options.all() if str(o.pk) == raw), None)
            if option is None:
                errors[question.pk] = "Wybierz jedną odpowiedź."
            else:
                picked[question] = option
                selected[question.pk] = option.pk
        if questions and not errors:
            response = submit_survey(request.user, picked)
            top = response.top_style
            if top and user_style(request.user) is None:
                set_user_style(request.user, top)
                messages.success(request, f"Dziękujemy! Według ankiety Twój dominujący styl to „{top.name}” — ustawiliśmy go jako Twój styl.")
            elif top:
                messages.success(request, f"Dziękujemy! Według ankiety Twój dominujący styl to „{top.name}”.")
            return redirect(reverse("learning:learning_style") + "#ankieta")
        messages.error(request, f"Odpowiedz na wszystkie pytania — brakuje {len(errors)}.")
    return render(
        request,
        "learning/style_survey.html",
        {"questions": questions, "errors": errors, "selected": selected},
    )


# ------------------------------------------------------------- wersje lekcji
def _lesson(request, slug: str, pk: int):
    course = _editable_course(request, slug)
    lesson = get_object_or_404(Lesson.objects.select_related("section"), pk=pk, section__course=course)
    return course, lesson


@login_required
def lesson_variants(request, slug: str, pk: int):
    """Lista stylów z informacją, które mają własną wersję tej lekcji."""
    course, lesson = _lesson(request, slug, pk)
    existing = {variant.style_id: variant for variant in lesson.variants.select_related("style")}
    rows = [(style, existing.get(style.pk)) for style in active_styles()]
    inactive = [v for v in existing.values() if not v.style.is_active]
    return render(
        request,
        "learning/lesson_variants.html",
        {"course": course, "lesson": lesson, "rows": rows, "inactive": inactive, "variant_count": len(existing)},
    )


@login_required
@require_POST
def lesson_styles_toggle(request, slug: str, pk: int):
    """Włącza albo wyłącza osobne wersje lekcji dla stylów."""
    course, lesson = _lesson(request, slug, pk)
    lesson.same_for_all_styles = request.POST.get("same") == "1"
    lesson.save(update_fields=["same_for_all_styles", "updated_at"])
    _content_changed(request, course)
    if lesson.same_for_all_styles:
        messages.info(request, "Wszyscy uczniowie widzą teraz tę samą treść lekcji. Zapisane wersje zostały zachowane.")
    else:
        messages.success(request, "Wersje dla stylów są włączone — uczeń zobaczy wersję dla swojego stylu, jeśli ją dodasz.")
    return redirect("learning:lesson_variants", slug=course.slug, pk=lesson.pk)


@login_required
def lesson_variant_edit(request, slug: str, pk: int, style_slug: str):
    """Dodanie albo edycja wersji lekcji dla stylu. Nowa wersja startuje od treści podstawowej."""
    course, lesson = _lesson(request, slug, pk)
    style = get_object_or_404(LearningStyle, slug=style_slug, is_active=True)
    variant = LessonVariant.objects.filter(lesson=lesson, style=style).first()
    if request.method == "POST":
        form = LessonVariantForm(request.POST, instance=variant)
        if form.is_valid():
            saved = form.save(commit=False)
            saved.lesson, saved.style = lesson, style
            saved.save()
            if lesson.same_for_all_styles:
                lesson.same_for_all_styles = False
                lesson.save(update_fields=["same_for_all_styles", "updated_at"])
            _content_changed(request, course)
            messages.success(request, f"Zapisano wersję lekcji „{lesson.title}” dla stylu „{style.name}”.")
            return redirect("learning:lesson_variants", slug=course.slug, pk=lesson.pk)
        messages.error(request, "Popraw zaznaczone pola formularza.")
    else:
        initial = None if variant else {"content_html": lesson.content_html, "video_url": ""}
        form = LessonVariantForm(instance=variant, initial=initial)
    return render(
        request,
        "learning/lesson_variant_form.html",
        {"course": course, "lesson": lesson, "style": style, "variant": variant, "form": form},
    )


@login_required
@require_POST
def lesson_variant_delete(request, slug: str, pk: int, style_slug: str):
    course, lesson = _lesson(request, slug, pk)
    variant = get_object_or_404(LessonVariant, lesson=lesson, style__slug=style_slug)
    name = variant.style.name
    variant.delete()
    _content_changed(request, course)
    messages.success(request, f"Usunięto wersję dla stylu „{name}” — uczniowie z tym stylem zobaczą treść podstawową.")
    return redirect("learning:lesson_variants", slug=course.slug, pk=lesson.pk)
