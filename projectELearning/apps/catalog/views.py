from __future__ import annotations

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import Http404, HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.decorators.http import require_POST

from apps.catalog.comments import MAX_DEPTH as MAX_COMMENT_DEPTH
from apps.catalog.comments import COMMENT_ORDERS, build_comment_tree, can_rate_course, public_comment_count
from apps.catalog.forms import CommentForm, CourseForm
from apps.catalog.models import RATING_CHOICES, Category, Course, Enrollment
from apps.catalog.permissions import (
    apply_edit_rules,
    can_edit_course,
    can_submit_course,
    can_view_course,
    course_has_lessons,
    course_reviewers,
    moderates_category,
    submit_course,
)
from apps.core.models import Notification
from apps.learning.models import Certificate
from apps.core.notifications import notify_many
from apps.core.pagination import paginate_queryset


def course_list(request: HttpRequest) -> HttpResponse:
    """Publiczny katalog kursów z wyszukiwarką, filtrami, sortowaniem i paginacją."""
    q = request.GET.get("q", "").strip()
    category_slug = request.GET.get("category", "").strip()
    price_type = request.GET.get("type", "").strip()
    difficulty = request.GET.get("difficulty", "").strip()
    sort_key = request.GET.get("sort", "popularity").strip()

    courses = Course.objects.published().select_related("category", "author")

    selected_category = None
    if category_slug:
        selected_category = Category.objects.filter(slug=category_slug, is_active=True).first()
        if selected_category:
            descendant_ids = selected_category.get_descendants(include_self=True).values_list("id", flat=True)
            courses = courses.filter(category_id__in=descendant_ids)

    if q:
        courses = courses.search(q)

    if price_type == "free":
        courses = courses.filter(is_free=True)
    elif price_type == "paid":
        courses = courses.filter(is_free=False)

    if difficulty in dict(Course.Level.choices):
        courses = courses.filter(difficulty_level=difficulty)

    from apps.learning.styles import learner_profile, rank_courses, recommend_courses

    profile = learner_profile(request.user) if request.user.is_authenticated else None
    can_match = bool(profile and not profile.is_empty)
    if sort_key == "dopasowane" and can_match:
        ranked = rank_courses(request.user, courses.sort_by("popularity"), profile)
        matched_ids = set()
        listing = []
        for fit in ranked:
            fit.course.fit_reason = fit.reason
            matched_ids.add(fit.course.pk)
            listing.append(fit.course)
        listing += [course for course in courses.sort_by("popularity") if course.pk not in matched_ids]
        courses = listing
    else:
        if sort_key == "dopasowane":
            sort_key = "popularity"
        courses = courses.sort_by(sort_key)

    page_obj, paginator = paginate_queryset(request, courses, per_page=9)

    recommended = []
    if can_match and page_obj.number == 1 and not (q or category_slug or price_type or difficulty) and sort_key != "dopasowane":
        for fit in recommend_courses(request.user, limit=3):
            fit.course.fit_reason = fit.reason
            recommended.append(fit.course)

    categories = Category.objects.filter(is_active=True)

    # Podkategorie wybranej kategorii (nawigacja po drzewie) z liczbą opublikowanych kursów
    subcategories = []
    if selected_category:
        published = Course.objects.published()
        for sub in selected_category.get_children().filter(is_active=True):
            branch = sub.get_descendants(include_self=True).values_list("id", flat=True)
            sub.course_total = published.filter(category_id__in=branch).count()
            subcategories.append(sub)

    sort_options = [
        {"key": "popularity", "label": "Najpopularniejsze"},
        {"key": "rating", "label": "Najwyżej oceniane"},
        {"key": "completions", "label": "Najwięcej ukończeń"},
        {"key": "newest", "label": "Najnowsze"},
        {"key": "price_asc", "label": "Cena: od najniższej"},
        {"key": "price_desc", "label": "Cena: od najwyższej"},
    ]
    if can_match:
        sort_options.insert(0, {"key": "dopasowane", "label": "Dopasowane do mnie (styl uczenia się)"})

    has_active_filters = bool(
        q or category_slug or price_type or difficulty or (sort_key and sort_key != "popularity")
    )

    context = {
        "page_obj": page_obj,
        "paginator": paginator,
        "total_count": paginator.count,
        "courses": page_obj.object_list,
        "categories": categories,
        "selected_category": selected_category,
        "subcategories": subcategories,
        "current_q": q,
        "current_category": category_slug,
        "current_type": price_type,
        "current_difficulty": difficulty,
        "current_sort": sort_key,
        "sort_options": sort_options,
        "difficulty_choices": Course.Level.choices,
        "has_active_filters": has_active_filters,
        "recommended": recommended,
        "profile": profile,
    }
    return render(request, "catalog/course_list.html", context)


def course_detail(request: HttpRequest, slug: str) -> HttpResponse:
    """Szczegóły kursu, program, opinie i zapisy."""
    course = get_object_or_404(
        Course.objects.select_related("category", "author"),
        slug=slug,
    )
    # Niezatwierdzony kurs widzą tylko autor i moderatorzy kategorii (404 dla reszty)
    if not can_view_course(request.user, course):
        raise Http404("Kurs nie istnieje albo nie został jeszcze opublikowany.")
    is_enrolled = False
    enrollment = None
    if request.user.is_authenticated:
        enrollment = Enrollment.objects.filter(user=request.user, course=course).first()
        is_enrolled = enrollment is not None

    from apps.learning.curriculum import build_outline

    outline = build_outline(course, request.user)
    breadcrumbs = course.category.get_ancestors(include_self=True) if course.category_id else []

    # Wątki komentarzy: paginowane komentarze główne, odpowiedzi w całości pod nimi
    is_moderator = moderates_category(request.user, course.category)
    comment_order = request.GET.get("komentarze", "najnowsze")
    if comment_order not in COMMENT_ORDERS:
        comment_order = "najnowsze"
    threads = build_comment_tree(course, request.user, comment_order)
    page_obj, paginator = paginate_queryset(request, threads, per_page=10)
    is_public = course.is_published and course.status == Course.Status.APPROVED
    can_rate = can_rate_course(request.user, course)
    comment_form = CommentForm(allow_rating=can_rate) if request.user.is_authenticated and is_public else None
    pending_here = (
        course.comments.filter(is_approved=False, is_removed=False).count() if is_moderator else 0
    )

    from apps.orders.cart import get_cart

    cart = get_cart(request)
    cart_item = cart.items.filter(course=course).first() if cart else None

    from apps.learning.models import LearningStyle
    from apps.learning.styles import course_style_fit, rank_courses

    started, completed, completion_rate = course.completion_stats()
    styles_by_id = {style.pk: style for style in LearningStyle.objects.filter(is_active=True)}
    style_versions = [
        {"style": styles_by_id[sid], "count": count, "total": total, "declared": declared}
        for sid, (_, count, total, declared) in course_style_fit([course]).get(course.pk, {}).items()
        if sid in styles_by_id
    ]
    style_versions.sort(key=lambda row: (styles_by_id[row["style"].pk].order, row["style"].name))
    personal_fit = rank_courses(request.user, [course]) if request.user.is_authenticated else []
    return render(
        request,
        "catalog/course_detail.html",
        {
            "started_count": started,
            "completed_count": completed,
            "completion_rate": completion_rate,
            "style_versions": style_versions,
            "personal_fit": personal_fit[0] if personal_fit else None,
            "course": course,
            "cart_quantity": cart_item.quantity if cart_item else 0,
            "can_edit": can_edit_course(request.user, course),
            "outline": outline,
            "next_lesson": outline.suggested_next() if is_enrolled else None,
            "breadcrumbs": breadcrumbs,
            "is_enrolled": is_enrolled,
            "enrollment": enrollment,
            "threads": page_obj.object_list,
            "page_obj": page_obj,
            "paginator": paginator,
            "comment_count": public_comment_count(course),
            "comment_form": comment_form,
            "can_rate": can_rate,
            "rating_choices": RATING_CHOICES,
            "is_moderator": is_moderator,
            "pending_here": pending_here,
            "max_comment_depth": MAX_COMMENT_DEPTH,
            "comment_order": comment_order,
            "comment_orders": COMMENT_ORDERS,
            "has_thread": course.submission_messages.exists(),
            "certificate": (
                Certificate.objects.filter(user=request.user, course=course).first()
                if request.user.is_authenticated else None
            ),
        },
    )


def notify_submission(course: Course, author, *, changed: bool = False) -> int:
    """Powiadamia nauczycieli kategorii (albo adminów) o kursie czekającym na akceptację."""
    name = author.get_full_name() or author.username
    title = "Opublikowany kurs zmieniony — do ponownej akceptacji" if changed else "Nowe zgłoszenie kursu"
    return notify_many(
        course_reviewers(course),
        Notification.Kind.COURSE_SUBMITTED,
        title,
        f"{name}: „{course.title}”" + (f" w kategorii {course.category.name}." if course.category_id else "."),
        url=reverse("catalog:course_detail", kwargs={"slug": course.slug}),
        exclude=author,
    )


def _course_form_context(request: HttpRequest, form: CourseForm, course: Course | None) -> dict:
    user = request.user
    moderated = []
    if user.is_authenticated and getattr(user, "role", "") == "moderator":
        moderated = list(user.moderated_categories.filter(is_active=True))
    return {"form": form, "course": course, "moderated_categories": moderated}


@login_required
def course_create(request: HttpRequest) -> HttpResponse:
    """Krok 1: dane kursu. Kurs powstaje jako szkic — widzi go tylko autor, nikt go nie ocenia,
    dopóki autor nie zbuduje programu i nie wyśle kursu (krok 3, ``course_submit``)."""
    if request.method == "POST":
        form = CourseForm(request.POST, request.FILES)
        if form.is_valid():
            course = form.save(commit=False)
            course.author = request.user
            course.status = Course.Status.DRAFT
            course.is_published = False
            course.save()
            form.save_m2m()  # style, dla których kurs jest szczególnie odpowiedni
            messages.success(request, f"Zapisano szkic „{course.title}”. Teraz dodaj sekcje i lekcje.")
            return redirect("learning:curriculum_edit", slug=course.slug)
        messages.error(request, "Popraw zaznaczone pola formularza.")
    else:
        form = CourseForm()
    return render(request, "catalog/course_form.html", _course_form_context(request, form, None))


@login_required
def course_edit(request: HttpRequest, slug: str) -> HttpResponse:
    """Edycja kursu przez autora albo moderatora jego kategorii."""
    course = get_object_or_404(Course.objects.select_related("category"), slug=slug)
    if not can_edit_course(request.user, course):
        raise Http404("Kurs nie istnieje albo nie masz do niego dostępu.")

    if request.method == "POST":
        form = CourseForm(request.POST, request.FILES, instance=course)
        if form.is_valid():
            course = form.save(commit=False)
            back_to_review = apply_edit_rules(course, request.user)
            course.save()
            form.save_m2m()
            if back_to_review:
                notify_submission(course, request.user, changed=True)
                messages.warning(request, "Zapisano zmiany. Kurs wrócił do akceptacji moderatora i do tego czasu nie jest widoczny w katalogu.")
            else:
                messages.success(request, "Zapisano zmiany w kursie.")
            return redirect("catalog:course_detail", slug=course.slug)
        messages.error(request, "Popraw zaznaczone pola formularza.")
    else:
        form = CourseForm(instance=course)
    return render(request, "catalog/course_form.html", _course_form_context(request, form, course))


@login_required
@require_POST
def course_submit(request: HttpRequest, slug: str) -> HttpResponse:
    """Krok 3: publikacja (nauczyciel w swojej kategorii, admin) albo wysłanie do akceptacji."""
    course = get_object_or_404(Course.objects.select_related("category"), slug=slug)
    if not can_submit_course(request.user, course):
        raise Http404("Kursu nie można teraz wysłać.")
    if not course_has_lessons(course):
        messages.error(request, "Dodaj co najmniej jedną lekcję, zanim wyślesz kurs.")
        return redirect("learning:curriculum_edit", slug=course.slug)
    if submit_course(course, request.user):
        course.save()
        messages.success(request, f"Kurs „{course.title}” jest już w katalogu.")
    else:
        course.save()
        notify_submission(course, request.user)
        messages.success(request, f"Wysłano kurs „{course.title}” do akceptacji. Damy znać, gdy nauczyciel go sprawdzi.")
    return redirect("catalog:course_detail", slug=course.slug)


@login_required
@require_POST
def course_enroll(request: HttpRequest, slug: str) -> HttpResponse:
    """Zapis na darmowy kurs (płatne kursy odblokowuje opłacone zamówienie)."""
    course = get_object_or_404(Course.objects.published(), slug=slug)
    if not course.is_course_free:
        messages.error(request, "Ten kurs jest płatny — dostęp odblokowuje opłacone zamówienie.")
        return redirect("catalog:course_detail", slug=course.slug)

    _, created = Enrollment.objects.get_or_create(user=request.user, course=course)
    if created:
        messages.success(request, f"Masz dostęp do kursu „{course.title}”. Powodzenia!")
    else:
        messages.info(request, "Masz już dostęp do tego kursu.")
    return redirect("catalog:course_detail", slug=course.slug)
