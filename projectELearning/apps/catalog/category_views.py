"""Zarządzanie drzewem kategorii w platformie: admin — całość, nauczyciel — swoja gałąź."""

from __future__ import annotations

from dataclasses import dataclass, field

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db.models import Count
from django.http import Http404, HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.decorators.http import require_POST

from apps.catalog.comments import is_moderator_somewhere
from apps.catalog.forms import CategoryForm
from apps.catalog.models import Category
from apps.catalog.moderation_views import _queue_counts
from apps.catalog.permissions import can_add_category, can_manage_category, is_platform_admin, manageable_categories
from apps.core.pagination import paginate_queryset


@dataclass
class CategoryNode:
    category: Category
    children: list[CategoryNode] = field(default_factory=list)
    can_manage: bool = False
    can_add_child: bool = False
    is_first: bool = False
    is_last: bool = False
    total_courses: int = 0


def _require_category_access(user) -> None:
    if not is_moderator_somewhere(user):
        raise Http404("Kategorie zarządzają administrator i nauczyciele (moderatorzy).")


def _build_tree(user) -> list[CategoryNode]:
    """Drzewo kategorii dostępnych dla użytkownika (korzenie = najwyższe dostępne węzły)."""
    categories = list(
        manageable_categories(user)
        .annotate(course_count=Count("courses", distinct=True))
        .prefetch_related("moderators")
        .order_by("tree_id", "lft")
    )
    by_id = {c.pk: CategoryNode(category=c) for c in categories}
    roots: list[CategoryNode] = []
    for category in categories:
        node = by_id[category.pk]
        node.can_manage = can_manage_category(user, category)
        node.can_add_child = can_add_category(user, category)
        parent = by_id.get(category.parent_id)
        (parent.children if parent else roots).append(node)

    def mark(nodes: list[CategoryNode]) -> int:
        total = 0
        for index, node in enumerate(nodes):
            node.is_first = index == 0
            node.is_last = index == len(nodes) - 1
            node.total_courses = node.category.course_count + mark(node.children)
            total += node.total_courses
        return total

    mark(roots)
    return roots


@login_required
def category_manage(request: HttpRequest) -> HttpResponse:
    _require_category_access(request.user)
    page_obj, paginator = paginate_queryset(request, _build_tree(request.user), per_page=10)
    return render(
        request,
        "catalog/category_manage.html",
        {
            "page_obj": page_obj,
            "paginator": paginator,
            "nodes": page_obj.object_list,
            "is_admin": is_platform_admin(request.user),
            **_queue_counts(request.user),
        },
    )


def _parent_choices(user):
    if is_platform_admin(user):
        return Category.objects.all()
    # Nauczyciel: rodzicem może być każda kategoria jego gałęzi (także przydzielona)
    return manageable_categories(user)


@login_required
def category_create(request: HttpRequest) -> HttpResponse:
    _require_category_access(request.user)
    parent = None
    parent_id = request.POST.get("parent") or request.GET.get("rodzic")
    if parent_id:
        parent = Category.objects.filter(pk=parent_id).first()
    if request.method == "POST":
        form = CategoryForm(request.POST, user=request.user, parent_choices=_parent_choices(request.user))
        if form.is_valid() and can_add_category(request.user, form.cleaned_data.get("parent")):
            category = form.save(commit=False)
            siblings = Category.objects.filter(parent=category.parent)
            category.order = (max(siblings.values_list("order", flat=True), default=0) or 0) + 10
            category.save()
            form.save_m2m()
            messages.success(request, f"Dodano kategorię „{category.name}”.")
            return redirect(reverse("catalog:category_manage") + f"#category-{category.pk}")
        if form.is_valid():
            form.add_error("parent", "Nie możesz dodać kategorii w tym miejscu drzewa.")
        messages.error(request, "Popraw zaznaczone pola formularza.")
    else:
        if parent is not None and not can_add_category(request.user, parent):
            raise Http404("Nie możesz dodać podkategorii w tym miejscu.")
        if parent is None and not is_platform_admin(request.user):
            parent = manageable_categories(request.user).filter(moderators=request.user).first()
        form = CategoryForm(
            user=request.user, parent_choices=_parent_choices(request.user),
            initial={"parent": parent, "is_active": True},
        )
    return render(request, "catalog/category_form.html", {"form": form, "category": None, "parent": parent})


@login_required
def category_edit(request: HttpRequest, pk: int) -> HttpResponse:
    category = get_object_or_404(Category, pk=pk)
    if not can_manage_category(request.user, category):
        raise Http404("Nie możesz edytować tej kategorii.")
    if request.method == "POST":
        form = CategoryForm(request.POST, instance=category, user=request.user, parent_choices=_parent_choices(request.user))
        if form.is_valid():
            form.save()
            Category.objects.rebuild()
            messages.success(request, f"Zapisano kategorię „{category.name}”.")
            return redirect(reverse("catalog:category_manage") + f"#category-{category.pk}")
        messages.error(request, "Popraw zaznaczone pola formularza.")
    else:
        form = CategoryForm(instance=category, user=request.user, parent_choices=_parent_choices(request.user))
    return render(request, "catalog/category_form.html", {"form": form, "category": category, "parent": category.parent})


@login_required
@require_POST
def category_move(request: HttpRequest, pk: int, direction: str) -> HttpResponse:
    """Zmiana kolejności wśród rodzeństwa (alternatywa klawiaturowa dla przeciągania)."""
    category = get_object_or_404(Category, pk=pk)
    if direction not in {"gora", "dol"} or not can_manage_category(request.user, category):
        raise Http404("Nie możesz zmienić kolejności tej kategorii.")
    siblings = list(Category.objects.filter(parent=category.parent).order_by("order", "name"))
    index = siblings.index(category)
    target = index - 1 if direction == "gora" else index + 1
    if 0 <= target < len(siblings):
        siblings[index], siblings[target] = siblings[target], siblings[index]
        for position, sibling in enumerate(siblings):
            Category.objects.filter(pk=sibling.pk).update(order=(position + 1) * 10)
        Category.objects.rebuild()
        messages.success(request, f"Przesunięto „{category.name}” {'wyżej' if direction == 'gora' else 'niżej'}.")
    return redirect(reverse("catalog:category_manage") + f"#category-{category.pk}")


@login_required
@require_POST
def category_toggle(request: HttpRequest, pk: int) -> HttpResponse:
    category = get_object_or_404(Category, pk=pk)
    if not can_manage_category(request.user, category):
        raise Http404("Nie możesz zmienić tej kategorii.")
    category.is_active = not category.is_active
    category.save(update_fields=["is_active"])
    state = "widoczna w katalogu" if category.is_active else "ukryta w katalogu"
    messages.success(request, f"Kategoria „{category.name}” jest teraz {state}.")
    return redirect(reverse("catalog:category_manage") + f"#category-{category.pk}")


@login_required
@require_POST
def category_delete(request: HttpRequest, pk: int) -> HttpResponse:
    """Usuwa tylko pustą kategorię — bez kursów i podkategorii (inaczej kursy straciłyby miejsce)."""
    category = get_object_or_404(Category, pk=pk)
    if not can_manage_category(request.user, category):
        raise Http404("Nie możesz usunąć tej kategorii.")
    if category.get_children().exists() or category.courses.exists():
        messages.error(
            request,
            f"Nie można usunąć „{category.name}”: ma kursy lub podkategorie. Przenieś je albo ukryj kategorię.",
        )
        return redirect(reverse("catalog:category_manage") + f"#category-{category.pk}")
    name = category.name
    category.delete()
    messages.success(request, f"Usunięto kategorię „{name}”.")
    return redirect("catalog:category_manage")
