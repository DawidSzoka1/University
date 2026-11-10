from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db.models import Count
from django.http import FileResponse, Http404, HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils.http import url_has_allowed_host_and_scheme
from django.views.decorators.http import require_POST

from apps.catalog.models import Course, Enrollment
from apps.catalog.permissions import (
    apply_edit_rules,
    can_edit_course,
    can_submit_course,
    course_has_lessons,
    is_platform_admin,
    moderates_category,
)
from apps.core.images import validate_image_upload
from apps.core.pagination import paginate_queryset
from apps.learning.curriculum import (
    build_outline,
    can_access_lesson,
    can_switch_mode,
    preview_mode,
    set_preview_mode,
    move,
    next_order,
    sync_enrollment_progress,
)
from apps.learning.forms import LessonForm, SectionForm
from apps.learning.media import video_embed_url
from apps.learning.models import CourseSection, Lesson, LessonAttachment, LessonProgress
from apps.learning.quizzes import gradable_attempts
from apps.learning.styles import analyse, recommend_courses, resolve_version, variant_counts


@login_required
def my_courses(request: HttpRequest) -> HttpResponse:
    """Kursy, których użytkownik się uczy (z postępem) i które tworzy."""
    enrollments = Enrollment.objects.filter(user=request.user).select_related("course", "course__category")
    page_obj, paginator = paginate_queryset(request, enrollments, per_page=10)
    for enrollment in page_obj.object_list:
        enrollment.next_lesson = build_outline(enrollment.course, request.user).suggested_next()
    authored = (
        Course.objects.filter(author=request.user)
        .select_related("category")
        .annotate(message_count=Count("submission_messages"))
        .order_by("-updated_at")
    )
    authored_page, _ = paginate_queryset(request, authored, per_page=10, page_param="authored_page")
    return render(
        request,
        "learning/my_courses.html",
        {
            "page_obj": page_obj,
            "paginator": paginator,
            "enrollments": page_obj.object_list,
            "authored_page": authored_page,
            "grading_count": gradable_attempts(request.user).count(),
            "recommended": recommend_courses(request.user, limit=3),
        },
    )


# ---------------------------------------------------------------------------
# Kreator programu (autor kursu i moderatorzy kategorii)
# ---------------------------------------------------------------------------

def _editable_course(request: HttpRequest, slug: str) -> Course:
    course = get_object_or_404(Course.objects.select_related("category"), slug=slug)
    if not can_edit_course(request.user, course):
        raise Http404("Kurs nie istnieje albo nie masz do niego dostępu.")
    return course


def _content_changed(request: HttpRequest, course: Course) -> None:
    """Zmiana treści przez nie-moderatora cofa opublikowany kurs do akceptacji."""
    if apply_edit_rules(course, request.user):
        course.save(update_fields=["status", "is_published", "updated_at"])
        from apps.catalog.views import notify_submission

        notify_submission(course, request.user, changed=True)
        messages.warning(request, "Kurs wrócił do akceptacji moderatora i do tego czasu nie jest widoczny w katalogu.")


@login_required
def curriculum_edit(request: HttpRequest, slug: str) -> HttpResponse:
    """Program kursu: sekcje i lekcje w sugerowanej kolejności; dodawanie sekcji."""
    course = _editable_course(request, slug)
    form = SectionForm()
    if request.method == "POST":
        form = SectionForm(request.POST)
        if form.is_valid():
            section = form.save(commit=False)
            section.course = course
            section.order = next_order(course.sections.all())
            section.save()
            _content_changed(request, course)
            messages.success(request, f"Dodano sekcję „{section.title}”.")
            return redirect("learning:curriculum_edit", slug=course.slug)
        messages.error(request, "Podaj nazwę sekcji.")
    outline = build_outline(course)
    lessons = [entry.lesson for section in outline.sections for entry in section.lessons]
    counts = variant_counts(lessons)
    for lesson in lessons:
        lesson.variant_count = counts.get(lesson.pk, 0)
    return render(
        request,
        "learning/curriculum_edit.html",
        {
            "course": course,
            "outline": outline,
            "form": form,
            "can_submit": can_submit_course(request.user, course),
            "has_lessons": course_has_lessons(course),
            "publishes_directly": moderates_category(request.user, course.category),
        },
    )


@login_required
@require_POST
def section_rename(request: HttpRequest, slug: str, pk: int) -> HttpResponse:
    course = _editable_course(request, slug)
    section = get_object_or_404(CourseSection, pk=pk, course=course)
    form = SectionForm(request.POST, instance=section)
    if form.is_valid():
        form.save()
        _content_changed(request, course)
        messages.success(request, "Zmieniono nazwę sekcji.")
    else:
        messages.error(request, "Nazwa sekcji nie może być pusta.")
    return redirect("learning:curriculum_edit", slug=course.slug)


@login_required
@require_POST
def section_delete(request: HttpRequest, slug: str, pk: int) -> HttpResponse:
    course = _editable_course(request, slug)
    section = get_object_or_404(CourseSection, pk=pk, course=course)
    title = section.title
    section.delete()
    _content_changed(request, course)
    messages.success(request, f"Usunięto sekcję „{title}” razem z jej lekcjami.")
    return redirect("learning:curriculum_edit", slug=course.slug)


@login_required
@require_POST
def section_move(request: HttpRequest, slug: str, pk: int, direction: str) -> HttpResponse:
    course = _editable_course(request, slug)
    if direction not in {"up", "down"}:
        raise Http404
    section = get_object_or_404(CourseSection, pk=pk, course=course)
    if move(section, course.sections.all(), direction):
        messages.success(request, f"Przesunięto sekcję „{section.title}” {'wyżej' if direction == 'up' else 'niżej'}.")
    return redirect(f"{course_program_url(course)}#section-{section.pk}")


@login_required
def lesson_create(request: HttpRequest, slug: str, section_pk: int) -> HttpResponse:
    course = _editable_course(request, slug)
    section = get_object_or_404(CourseSection, pk=section_pk, course=course)
    if request.method == "POST":
        form = LessonForm(request.POST, request.FILES, course=course)
        if form.is_valid():
            lesson = form.save(commit=False)
            lesson.order = next_order(lesson.section.lessons.all())
            form.save()  # zapis lekcji + plików do pobrania
            _content_changed(request, course)
            messages.success(request, f"Dodano lekcję „{lesson.title}”.")
            return redirect("learning:curriculum_edit", slug=course.slug)
        messages.error(request, "Popraw zaznaczone pola formularza.")
    else:
        form = LessonForm(course=course, initial={"section": section})
    return render(request, "learning/lesson_form.html", {"course": course, "form": form, "lesson": None})


@login_required
def lesson_edit(request: HttpRequest, slug: str, pk: int) -> HttpResponse:
    course = _editable_course(request, slug)
    lesson = get_object_or_404(Lesson, pk=pk, section__course=course)
    if request.method == "POST":
        previous_section = lesson.section_id
        form = LessonForm(request.POST, request.FILES, instance=lesson, course=course)
        if form.is_valid():
            lesson = form.save(commit=False)
            if lesson.section_id != previous_section:
                lesson.order = next_order(lesson.section.lessons.exclude(pk=lesson.pk))
            form.save()  # zapis lekcji + plików do pobrania/usuwanych
            _content_changed(request, course)
            messages.success(request, "Zapisano lekcję.")
            return redirect("learning:curriculum_edit", slug=course.slug)
        messages.error(request, "Popraw zaznaczone pola formularza.")
    else:
        form = LessonForm(instance=lesson, course=course)
    return render(request, "learning/lesson_form.html", {"course": course, "form": form, "lesson": lesson})


@login_required
@require_POST
def lesson_delete(request: HttpRequest, slug: str, pk: int) -> HttpResponse:
    course = _editable_course(request, slug)
    lesson = get_object_or_404(Lesson, pk=pk, section__course=course)
    title = lesson.title
    for attachment in lesson.attachments.all():
        attachment.file.delete(save=False)
    if lesson.video_file:
        lesson.video_file.delete(save=False)
    lesson.delete()
    _content_changed(request, course)
    messages.success(request, f"Usunięto lekcję „{title}”.")
    return redirect("learning:curriculum_edit", slug=course.slug)


@login_required
@require_POST
def lesson_move(request: HttpRequest, slug: str, pk: int, direction: str) -> HttpResponse:
    course = _editable_course(request, slug)
    if direction not in {"up", "down"}:
        raise Http404
    lesson = get_object_or_404(Lesson, pk=pk, section__course=course)
    if move(lesson, lesson.section.lessons.all(), direction):
        messages.success(request, f"Przesunięto lekcję „{lesson.title}” {'wyżej' if direction == 'up' else 'niżej'}.")
    return redirect(f"{course_program_url(course)}#lesson-{lesson.pk}")


def course_program_url(course: Course) -> str:
    return reverse("learning:curriculum_edit", kwargs={"slug": course.slug})


# ---------------------------------------------------------------------------
# Nauka: widok lekcji, ukończenie, pliki
# ---------------------------------------------------------------------------

def lesson_detail(request: HttpRequest, slug: str, pk: int) -> HttpResponse:
    lesson = get_object_or_404(
        Lesson.objects.select_related("section__course__category"), pk=pk, section__course__slug=slug
    )
    course = lesson.section.course
    if not can_access_lesson(request.user, lesson):
        if not request.user.is_authenticated:
            return redirect(f"/accounts/login/?next={request.path}")
        messages.info(request, "Ta lekcja jest dostępna po zapisaniu się na kurs.")
        return redirect("catalog:course_detail", slug=course.slug)

    preview = preview_mode(request, course)
    outline = build_outline(course, request.user, preview=preview)
    current = outline.find(lesson)
    if current is not None and not current.accessible:
        return _locked_redirect(request, outline, course)
    previous, following = outline.neighbours(lesson)
    version = resolve_version(lesson, request.user, request.GET.get("wersja"))
    analysis = analyse(request.user) if request.user.is_authenticated else None
    return render(
        request,
        "learning/lesson_detail.html",
        {
            "version": version,
            "style_tip": analysis if analysis and analysis.differs else None,
            "course": course,
            "lesson": lesson,
            "current": current,
            "outline": outline,
            "previous": previous,
            "following": following,
            "embed_url": video_embed_url(version.video_url),
            "can_edit": can_edit_course(request.user, course),
            "can_track": request.user.is_authenticated,
            "preview": preview,
            "can_switch": can_switch_mode(request.user, course),
        },
    )


@login_required
@require_POST
def learning_mode(request: HttpRequest, slug: str) -> HttpResponse:
    """Przełącza autora/moderatora zapisanego na kurs między nauką a podglądem autora."""
    course = get_object_or_404(Course, slug=slug)
    if not can_switch_mode(request.user, course):
        raise Http404
    preview = request.POST.get("mode") == "preview"
    set_preview_mode(request, course, preview)
    messages.info(
        request,
        "Podgląd autora: wszystkie lekcje otwarte, odpowiedzi w testach nie są wysyłane."
        if preview else
        "Tryb nauki: rozwiązujesz testy i robisz postępy jak uczeń.",
    )
    fallback = reverse("catalog:course_detail", kwargs={"slug": course.slug})
    target = request.POST.get("next", "")
    return redirect(target if url_has_allowed_host_and_scheme(target, {request.get_host()}) else fallback)


def _locked_redirect(request: HttpRequest, outline, course: Course) -> HttpResponse:
    """Lekcja za niezdanym testem obowiązkowym — przekierowanie do testu, który ją blokuje."""
    blocker = outline.blocking_quiz
    if blocker is None:
        messages.info(request, "Ta część kursu jest jeszcze zablokowana.")
        return redirect("catalog:course_detail", slug=course.slug)
    if blocker.pending:
        messages.info(request, f"Dalsza część kursu odblokuje się po ocenie testu „{blocker.quiz.title}” przez nauczyciela.")
    else:
        messages.info(request, f"Żeby przejść dalej, zalicz test „{blocker.quiz.title}”.")
    return redirect(blocker.url)


@login_required
@require_POST
def lesson_toggle_complete(request: HttpRequest, slug: str, pk: int) -> HttpResponse:
    lesson = get_object_or_404(Lesson.objects.select_related("section__course"), pk=pk, section__course__slug=slug)
    if not can_access_lesson(request.user, lesson):
        raise Http404
    current = build_outline(lesson.section.course, request.user).find(lesson)
    if current is not None and not current.accessible:
        raise Http404
    # Wersja, którą uczeń właśnie przerobił (do analizy stylów); obca/nieistniejąca = podstawowa
    version = resolve_version(lesson, request.user, request.POST.get("wersja") or None)
    progress, created = LessonProgress.objects.get_or_create(
        user=request.user, lesson=lesson, defaults={"style": version.style}
    )
    if not created:
        progress.delete()
        messages.info(request, f"Lekcja „{lesson.title}” oznaczona jako nieukończona.")
    else:
        messages.success(request, f"Ukończono lekcję „{lesson.title}”.")
    sync_enrollment_progress(request.user, lesson.section.course)
    target = reverse("learning:lesson_detail", kwargs={"slug": slug, "pk": lesson.pk})
    if version.requested:
        target += f"?wersja={request.POST.get('wersja')}"
    return redirect(target)


def attachment_download(request: HttpRequest, pk: int) -> FileResponse:
    """Plik lekcji pobiera tylko ktoś, kto ma dostęp do lekcji."""
    attachment = get_object_or_404(LessonAttachment.objects.select_related("lesson__section__course"), pk=pk)
    if not can_access_lesson(request.user, attachment.lesson):
        raise Http404
    return FileResponse(attachment.file.open("rb"), as_attachment=True, filename=attachment.original_name)


# ---------------------------------------------------------------------------
# Zdjęcia w edytorze lekcji (nadpisuje upload django-ckeditor-5)
# ---------------------------------------------------------------------------

def may_upload_editor_images(user) -> bool:
    """Wgrywać zdjęcia do treści mogą nauczyciele, admin i autorzy kursów."""
    if not user.is_authenticated:
        return False
    if is_platform_admin(user) or getattr(user, "role", "") == "moderator":
        return True
    return Course.objects.filter(author=user).exists()


@require_POST
def editor_image_upload(request: HttpRequest) -> JsonResponse:
    from django_ckeditor_5.views import upload_file

    if not may_upload_editor_images(request.user):
        return JsonResponse({"error": {"message": "Nie masz uprawnień do dodawania zdjęć."}}, status=403)
    upload = request.FILES.get("upload")
    if upload is None:
        return JsonResponse({"error": {"message": "Nie przesłano pliku."}}, status=400)
    try:
        validate_image_upload(upload)
    except Exception as error:  # ValidationError z czytelnym komunikatem
        message = getattr(error, "messages", [str(error)])[0]
        return JsonResponse({"error": {"message": message}}, status=400)
    return upload_file(request)
