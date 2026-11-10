from django.contrib.auth.decorators import login_required
from django.http import HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils.http import url_has_allowed_host_and_scheme
from django.views.decorators.http import require_POST

from apps.catalog.models import Category, Course
from apps.core.models import Notification
from apps.core.pagination import paginate_queryset


def home(request: HttpRequest) -> HttpResponse:
    """Strona główna — hero z nowościami, liczby, kroki nauki, kategorie, motywy, CTA."""
    published = Course.objects.published()

    root_categories = Category.objects.filter(is_active=True, parent__isnull=True).order_by("order", "name")[:6]
    categories = [
        {
            "category": category,
            "course_count": published.filter(
                category__in=category.get_descendants(include_self=True)
            ).count(),
        }
        for category in root_categories
    ]

    return render(
        request,
        "core/home.html",
        {
            "course_count": published.count(),
            "category_count": Category.objects.filter(is_active=True).count(),
            "latest_courses": published.select_related("category").order_by("-created_at")[:3],
            "categories": categories,
        },
    )


@login_required
def notification_list(request: HttpRequest) -> HttpResponse:
    """Powiadomienia użytkownika: filtr „nieprzeczytane / wszystkie”, najnowsze na górze, paginacja."""
    only_unread = request.GET.get("pokaz") == "nieprzeczytane"
    notifications = Notification.objects.filter(user=request.user).order_by("-created_at")
    if only_unread:
        notifications = notifications.filter(is_read=False)
    page_obj, paginator = paginate_queryset(request, notifications, per_page=15)
    return render(
        request,
        "core/notification_list.html",
        {
            "page_obj": page_obj,
            "paginator": paginator,
            "notifications": page_obj.object_list,
            "only_unread": only_unread,
        },
    )


@login_required
@require_POST
def notification_mark_all_read(request: HttpRequest) -> HttpResponse:
    """Oznacza wszystkie powiadomienia użytkownika jako przeczytane."""
    from django.contrib import messages

    from apps.core.templatetags.core_tags import pl_plural

    updated = Notification.objects.filter(user=request.user, is_read=False).update(is_read=True)
    if updated:
        messages.success(
            request, f"Oznaczono {updated} {pl_plural(updated, 'powiadomienie,powiadomienia,powiadomień')} jako przeczytane."
        )
    else:
        messages.info(request, "Wszystkie powiadomienia były już przeczytane.")
    return redirect("core:notification_list")


@login_required
@require_POST
def notification_open(request: HttpRequest, pk: int) -> HttpResponse:
    """Oznacza powiadomienie jako przeczytane i przenosi do miejsca, którego dotyczy."""
    notification = get_object_or_404(Notification, pk=pk, user=request.user)
    if not notification.is_read:
        notification.is_read = True
        notification.save(update_fields=["is_read"])
    target = notification.url
    if target and url_has_allowed_host_and_scheme(target, allowed_hosts={request.get_host()}, require_https=request.is_secure()):
        return redirect(target)
    return redirect("core:notification_list")


def anti_wcag(request: HttpRequest) -> HttpResponse:
    """Strona Anty-WCAG: CELOWO łamie dostępność (wymóg projektu). Samodzielny szablon z własnym
    CSS/JS — nie dziedziczy base.html ani main.css. Nie poprawiać jej."""
    return render(request, "anti_wcag/page.html")


def anti_wcag_guide(request: HttpRequest) -> HttpResponse:
    """Dostępny opis naruszeń na stronie Anty-WCAG — co jest zepsute i jak to sprawdzić."""
    return render(request, "anti_wcag/guide.html")
