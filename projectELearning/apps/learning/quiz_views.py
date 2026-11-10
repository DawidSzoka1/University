"""Testy w kursie: kreator (nauczyciel/autor), rozwiązywanie (uczeń), ocenianie i dyplomy."""

from __future__ import annotations

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db import transaction
from django.http import FileResponse, Http404, HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.decorators.http import require_POST

from apps.catalog.models import Course
from apps.catalog.permissions import can_edit_course, is_platform_admin
from apps.core.pagination import paginate_queryset
from apps.learning.certificates import render_certificate
from apps.learning.curriculum import build_outline, can_switch_mode, is_enrolled, move, next_order, preview_mode
from apps.learning.forms import GradeAnswerForm, QuestionForm, QuizForm, choice_formset
from apps.learning.media import video_embed_url
from apps.learning.models import Answer, Attempt, Certificate, Question, Quiz
from apps.learning.quizzes import (
    attempts_left,
    can_grade,
    finalize_attempt,
    gradable_attempts,
    read_submission,
    start_sitting,
    submit_attempt,
)
from apps.learning.views import _content_changed, _editable_course, course_program_url


def _quiz_of(course: Course, pk: int) -> Quiz:
    return get_object_or_404(Quiz.objects.select_related("course", "after_lesson"), pk=pk, course=course)


# ---------------------------------------------------------------------------
# Kreator: ustawienia testu i pytania
# ---------------------------------------------------------------------------

@login_required
def quiz_create(request: HttpRequest, slug: str) -> HttpResponse:
    course = _editable_course(request, slug)
    if request.method == "POST":
        form = QuizForm(request.POST, course=course)
        if form.is_valid():
            quiz = form.save(commit=False)
            quiz.course = course
            quiz.order = next_order(course.quizzes.filter(after_lesson=quiz.after_lesson))
            quiz.save()
            _content_changed(request, course)
            messages.success(request, f"Utworzono test „{quiz.title}”. Dodaj pytania.")
            return redirect("learning:quiz_edit", slug=course.slug, pk=quiz.pk)
        messages.error(request, "Popraw zaznaczone pola formularza.")
    else:
        initial = {"pass_percent": 60, "max_attempts": 0}
        after = request.GET.get("po_lekcji")
        if after and after.isdigit():
            initial["after_lesson"] = int(after)
        form = QuizForm(initial=initial, course=course)
    return render(request, "learning/quiz_settings.html", {"course": course, "form": form, "quiz": None})


@login_required
def quiz_edit(request: HttpRequest, slug: str, pk: int) -> HttpResponse:
    """Ustawienia testu i lista pytań."""
    course = _editable_course(request, slug)
    quiz = _quiz_of(course, pk)
    if request.method == "POST":
        form = QuizForm(request.POST, instance=quiz, course=course)
        if form.is_valid():
            form.save()
            _content_changed(request, course)
            messages.success(request, "Zapisano ustawienia testu.")
            return redirect("learning:quiz_edit", slug=course.slug, pk=quiz.pk)
        messages.error(request, "Popraw zaznaczone pola formularza.")
    else:
        form = QuizForm(instance=quiz, course=course)
    questions = quiz.questions.prefetch_related("choices")
    return render(
        request,
        "learning/quiz_edit.html",
        {"course": course, "quiz": quiz, "form": form, "questions": questions},
    )


@login_required
@require_POST
def quiz_delete(request: HttpRequest, slug: str, pk: int) -> HttpResponse:
    course = _editable_course(request, slug)
    quiz = _quiz_of(course, pk)
    title = quiz.title
    quiz.delete()
    _content_changed(request, course)
    messages.success(request, f"Usunięto test „{title}”.")
    return redirect(course_program_url(course))


def _question_form_response(request, course, quiz, form, formset, question=None) -> HttpResponse:
    return render(
        request,
        "learning/question_form.html",
        {"course": course, "quiz": quiz, "form": form, "formset": formset, "question": question},
    )


def _save_question(request, course, quiz, question=None) -> HttpResponse:
    instance = question or Question(quiz=quiz)
    if request.method == "POST":
        form = QuestionForm(request.POST, request.FILES, instance=instance)
        question_type = request.POST.get("question_type", "")
        # Liczbę wierszy przy POST wyznacza formularz zarządzający (TOTAL_FORMS)
        formset = choice_formset()(request.POST, instance=instance, prefix="choices", question_type=question_type)
        if form.is_valid() and formset.is_valid():
            with transaction.atomic():
                saved = form.save(commit=False)
                if not saved.pk:
                    saved.order = next_order(quiz.questions.all())
                saved.save()
                if saved.is_open:
                    saved.choices.all().delete()
                else:
                    position = 0
                    for choice_form in formset.forms:
                        data = choice_form.cleaned_data
                        if not data:
                            continue
                        if data.get("DELETE") or not (data.get("text") or "").strip():
                            if choice_form.instance.pk:
                                choice_form.instance.delete()
                            continue
                        position += 1
                        choice = choice_form.save(commit=False)
                        choice.question = saved
                        choice.order = position
                        choice.save()
            _content_changed(request, course)
            messages.success(request, "Zapisano pytanie.")
            return redirect(f"{reverse('learning:quiz_edit', kwargs={'slug': course.slug, 'pk': quiz.pk})}#question-{saved.pk}")
        messages.error(request, "Popraw zaznaczone pola pytania.")
    else:
        form = QuestionForm(instance=instance)
        existing = instance.choices.count() if instance.pk else 0
        extra = max(4 - existing, 2) if existing < 8 else 0
        formset = choice_formset(min(extra, 8 - existing))(instance=instance, prefix="choices", question_type=instance.question_type)
    return _question_form_response(request, course, quiz, form, formset, question)


@login_required
def question_create(request: HttpRequest, slug: str, quiz_pk: int) -> HttpResponse:
    course = _editable_course(request, slug)
    quiz = _quiz_of(course, quiz_pk)
    return _save_question(request, course, quiz)


@login_required
def question_edit(request: HttpRequest, slug: str, pk: int) -> HttpResponse:
    course = _editable_course(request, slug)
    question = get_object_or_404(Question.objects.select_related("quiz"), pk=pk, quiz__course=course)
    return _save_question(request, course, question.quiz, question)


@login_required
@require_POST
def question_delete(request: HttpRequest, slug: str, pk: int) -> HttpResponse:
    course = _editable_course(request, slug)
    question = get_object_or_404(Question.objects.select_related("quiz"), pk=pk, quiz__course=course)
    quiz = question.quiz
    question.delete()
    _content_changed(request, course)
    messages.success(request, "Usunięto pytanie.")
    return redirect("learning:quiz_edit", slug=course.slug, pk=quiz.pk)


@login_required
@require_POST
def question_move(request: HttpRequest, slug: str, pk: int, direction: str) -> HttpResponse:
    course = _editable_course(request, slug)
    question = get_object_or_404(Question.objects.select_related("quiz"), pk=pk, quiz__course=course)
    if direction not in {"up", "down"}:
        raise Http404
    if move(question, question.quiz.questions.all(), direction):
        messages.success(request, f"Przesunięto pytanie {'wyżej' if direction == 'up' else 'niżej'}.")
    return redirect(f"{reverse('learning:quiz_edit', kwargs={'slug': course.slug, 'pk': question.quiz_id})}#question-{question.pk}")


# ---------------------------------------------------------------------------
# Uczeń: rozwiązywanie testu i wynik
# ---------------------------------------------------------------------------

def quiz_detail(request: HttpRequest, slug: str, pk: int) -> HttpResponse:
    """Test do rozwiązania (uczeń) albo podgląd (autor/moderator). POST wysyła odpowiedzi."""
    course = get_object_or_404(Course.objects.select_related("category", "author"), slug=slug)
    quiz = _quiz_of(course, pk)
    editor = can_edit_course(request.user, course)
    if not request.user.is_authenticated:
        return redirect(f"{reverse('accounts:login')}?next={request.path}")
    if not editor and not is_enrolled(request.user, course):
        messages.info(request, "Test jest dostępny po zapisaniu się na kurs.")
        return redirect("catalog:course_detail", slug=course.slug)

    preview = preview_mode(request, course)  # autor bez zapisu albo wybrany podgląd — bez wysyłania
    outline = build_outline(course, request.user, preview=preview)
    entry = outline.find_quiz(quiz)
    if entry is None or not entry.accessible:
        blocker = outline.blocking_quiz
        messages.info(
            request,
            f"Najpierw zalicz test „{blocker.quiz.title}”." if blocker else "Ten test jest jeszcze zablokowany.",
        )
        return redirect(blocker.url if blocker else reverse("catalog:course_detail", kwargs={"slug": course.slug}))

    questions = list(quiz.questions.prefetch_related("choices"))
    attempts = quiz.attempts.filter(user=request.user)
    left = attempts_left(quiz, request.user)
    submission = None
    can_submit = not preview and bool(questions) and left != 0 and not entry.passed and not entry.pending

    if request.method == "POST":
        if not can_submit:
            messages.error(request, "Nie możesz teraz wysłać tego testu.")
            return redirect("learning:quiz_detail", slug=course.slug, pk=quiz.pk)
        submission = read_submission(quiz, request.POST, request.FILES)
        if submission.is_valid:
            attempt = submit_attempt(quiz, request.user, submission)
            if attempt.status == Attempt.Status.PENDING_REVIEW:
                messages.success(request, "Wysłano odpowiedzi. Pytania otwarte sprawdzi nauczyciel — dostaniesz powiadomienie z wynikiem.")
            elif attempt.passed:
                messages.success(request, f"Zaliczone! {attempt.score}/{attempt.max_score} pkt ({attempt.percent}%).")
            else:
                messages.warning(request, f"Niezaliczone: {attempt.score}/{attempt.max_score} pkt ({attempt.percent}%). Próg to {quiz.pass_percent}%.")
            return redirect("learning:attempt_detail", pk=attempt.pk)
        messages.error(request, "Uzupełnij odpowiedzi zaznaczone na czerwono.")

    if can_submit and request.method == "GET":
        start_sitting(quiz, request.user)
    previous, following = outline.neighbours(quiz)
    for question in questions:
        question.embed_url = video_embed_url(question.video_url)
        question.error = submission.errors.get(question.pk) if submission else None
        if submission and request.method == "POST":
            question.posted = request.POST.getlist(f"q{question.pk}")
            question.posted_text = request.POST.get(f"q{question.pk}", "")
    return render(
        request,
        "learning/quiz_detail.html",
        {
            "course": course,
            "quiz": quiz,
            "entry": entry,
            "questions": questions,
            "attempts": attempts[:5],
            "attempt_count": attempts.count(),
            "attempts_left": left,
            "can_submit": can_submit,
            "editor": preview,
            "can_edit": editor,
            "can_switch": can_switch_mode(request.user, course),
            "outline": outline,
            "previous": previous,
            "following": following,
            "has_file_questions": any(q.is_open and q.answer_format in ("image", "file") for q in questions),
        },
    )


def _attempt_for(request: HttpRequest, pk: int) -> tuple[Attempt, bool]:
    attempt = get_object_or_404(Attempt.objects.select_related("quiz__course__category", "user", "graded_by"), pk=pk)
    # Własnego podejścia nie oceniasz (np. admin, który kupił kurs) — robi to inny nauczyciel
    grader = can_grade(request.user, attempt.quiz.course) and attempt.user_id != request.user.pk
    if attempt.user_id != request.user.pk and not grader:
        raise Http404("Nie masz dostępu do tego podejścia.")
    return attempt, grader


def _answers_with_selection(attempt: Attempt) -> list[Answer]:
    answers = list(attempt.answers.select_related("question").prefetch_related("selected_choices", "question__choices"))
    for answer in answers:
        answer.selected_ids = {choice.pk for choice in answer.selected_choices.all()}
    return answers


@login_required
def attempt_detail(request: HttpRequest, pk: int) -> HttpResponse:
    """Wynik podejścia: punkty za każde pytanie i komentarze nauczyciela."""
    attempt, grader = _attempt_for(request, pk)
    answers = _answers_with_selection(attempt)
    quiz = attempt.quiz
    outline = build_outline(quiz.course, attempt.user)
    entry = outline.find_quiz(quiz)
    # Poprawne odpowiedzi ABC pokazujemy po zaliczeniu albo gdy nie ma już podejść
    reveal = attempt.passed or attempts_left(quiz, attempt.user) == 0 or grader
    _, following = outline.neighbours(quiz)
    return render(
        request,
        "learning/attempt_detail.html",
        {
            "attempt": attempt,
            "quiz": quiz,
            "course": quiz.course,
            "answers": answers,
            "reveal": reveal,
            "grader": grader,
            "is_owner": attempt.user_id == request.user.pk,
            "entry": entry,
            "following": following if attempt.passed else None,
            "can_retry": attempt.user_id == request.user.pk and not (entry and (entry.passed or entry.pending)) and attempts_left(quiz, attempt.user) != 0,
        },
    )


def question_file(request: HttpRequest, pk: int) -> FileResponse:
    """Plik do pytania — dla osób z dostępem do testu (zapisanych i autorów)."""
    question = get_object_or_404(Question.objects.select_related("quiz__course"), pk=pk)
    course = question.quiz.course
    if not question.attachment or not (can_edit_course(request.user, course) or is_enrolled(request.user, course)):
        raise Http404
    return FileResponse(question.attachment.open("rb"), as_attachment=True, filename=question.attachment_name or "plik")


@login_required
def answer_file(request: HttpRequest, pk: int) -> FileResponse:
    """Plik/zdjęcie z odpowiedzi — tylko autor odpowiedzi i oceniający."""
    answer = get_object_or_404(Answer.objects.select_related("attempt__quiz__course"), pk=pk)
    if answer.attempt.user_id != request.user.pk and not can_grade(request.user, answer.attempt.quiz.course):
        raise Http404
    if not answer.file:
        raise Http404
    inline = answer.question.answer_format == "image" and request.GET.get("pobierz") is None
    return FileResponse(answer.file.open("rb"), as_attachment=not inline, filename=answer.original_name or "odpowiedz")


# ---------------------------------------------------------------------------
# Nauczyciel: kolejka i ocena odpowiedzi otwartych
# ---------------------------------------------------------------------------

@login_required
def grading_queue(request: HttpRequest) -> HttpResponse:
    attempts = gradable_attempts(request.user).order_by("submitted_at")
    is_teacher = is_platform_admin(request.user) or getattr(request.user, "role", "") == "moderator"
    if not is_teacher and not Course.objects.filter(author=request.user).exists():
        raise Http404("Kolejka ocen jest dla nauczycieli i autorów kursów.")
    page_obj, paginator = paginate_queryset(request, attempts, per_page=10)
    context = {"page_obj": page_obj, "paginator": paginator, "attempts": page_obj.object_list, "is_teacher": is_teacher}
    if is_teacher:
        from apps.catalog.moderation_views import _queue_counts

        context.update(_queue_counts(request.user))
    return render(request, "learning/grading_queue.html", context)


@login_required
def attempt_grade(request: HttpRequest, pk: int) -> HttpResponse:
    attempt, grader = _attempt_for(request, pk)
    if not grader:
        raise Http404("Nie możesz oceniać tego testu.")
    answers = _answers_with_selection(attempt)
    open_answers = [a for a in answers if a.question.is_open]
    if attempt.status == Attempt.Status.GRADED and request.method != "POST":
        messages.info(request, "To podejście jest już ocenione — możesz poprawić ocenę.")

    forms_by_answer = {}
    valid = True
    for answer in open_answers:
        initial = {"points": answer.points, "feedback": answer.feedback}
        form = GradeAnswerForm(
            request.POST if request.method == "POST" else None,
            prefix=f"a{answer.pk}",
            initial=initial,
            max_points=answer.question.points,
        )
        if request.method == "POST" and not form.is_valid():
            valid = False
        forms_by_answer[answer.pk] = form
        answer.grade_form = form

    if request.method == "POST":
        if valid:
            with transaction.atomic():
                for answer in open_answers:
                    data = forms_by_answer[answer.pk].cleaned_data
                    answer.points = data["points"]
                    answer.feedback = data["feedback"].strip()
                    answer.save(update_fields=["points", "feedback"])
                attempt.feedback = request.POST.get("overall_feedback", "").strip()[:3000]
                finalize_attempt(attempt, grader=request.user)
            result = "zaliczony" if attempt.passed else "niezaliczony"
            messages.success(request, f"Oceniono: {attempt.score}/{attempt.max_score} pkt ({attempt.percent}%) — {result}. Uczeń dostał powiadomienie.")
            return redirect("learning:grading_queue")
        messages.error(request, "Popraw punkty zaznaczone przy odpowiedziach.")

    for answer in answers:
        answer.embed_url = video_embed_url(answer.question.video_url)
    return render(
        request,
        "learning/attempt_grade.html",
        {"attempt": attempt, "quiz": attempt.quiz, "course": attempt.quiz.course, "answers": answers},
    )


# ---------------------------------------------------------------------------
# Dyplomy
# ---------------------------------------------------------------------------

def certificate_verify(request: HttpRequest, code: str) -> HttpResponse:
    """Publiczna strona: czy dyplom o tym numerze został wystawiony."""
    certificate = Certificate.objects.select_related("course").filter(code=code.upper()).first()
    is_owner = certificate is not None and certificate.user_id == request.user.pk
    return render(
        request,
        "learning/certificate_verify.html",
        {"certificate": certificate, "code": code, "is_owner": is_owner},
        status=200 if certificate else 404,
    )


@login_required
def certificate_download(request: HttpRequest, code: str) -> HttpResponse:
    certificate = get_object_or_404(Certificate, code=code.upper())
    if certificate.user_id != request.user.pk and not is_platform_admin(request.user):
        raise Http404
    verify_url = request.build_absolute_uri(reverse("learning:certificate_verify", kwargs={"code": certificate.code}))
    pdf = render_certificate(certificate, verify_url)
    response = HttpResponse(pdf, content_type="application/pdf")
    response["Content-Disposition"] = f'attachment; filename="dyplom-{certificate.code}.pdf"'
    return response
