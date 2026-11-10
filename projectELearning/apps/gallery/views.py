"""Galeria: 1) zdjęcia do bazy, 2) konfigurator układów slajdera (admin), 3) slajder publiczny."""

from __future__ import annotations

import json

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import Http404, HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.decorators.http import require_POST

from apps.core.pagination import paginate_queryset
from apps.gallery import slider
from apps.gallery.forms import GalleryItemForm, SliderPresetForm
from apps.gallery.models import GalleryItem, SliderPreset


def _require(check, user) -> None:
    if not check(user):
        raise Http404("Ta strona jest dostępna dla administratora i nauczycieli.")


def _wants_json(request: HttpRequest) -> bool:
    return "application/json" in request.headers.get("Accept", "")


# ---------------------------------------------------------------------------
# 1. Zdjęcia w bazie
# ---------------------------------------------------------------------------

@login_required
def upload_image(request: HttpRequest) -> HttpResponse:
    """Dodawanie zdjęć do bazy (skalowanie Pillow) i lista zdjęć z edycją."""
    _require(slider.can_upload, request.user)
    if request.method == "POST":
        form = GalleryItemForm(request.POST, request.FILES)
        if form.is_valid():
            item = form.save(commit=False)
            item.uploaded_by = request.user
            item.save()
            if item.was_resized:
                messages.success(
                    request,
                    f"Dodano „{item.title}”. Zdjęcie przeskalowano z {item.original_width}×{item.original_height} "
                    f"do {item.width}×{item.height} px.",
                )
            else:
                messages.success(request, f"Dodano „{item.title}” ({item.width}×{item.height} px — bez skalowania).")
            return redirect(f"{reverse('gallery:upload_image')}#item-{item.pk}")
        messages.error(request, "Popraw zaznaczone pola formularza.")
    else:
        form = GalleryItemForm()
    items = GalleryItem.objects.select_related("related_course", "uploaded_by").prefetch_related("preset_appearances__preset")
    page_obj, paginator = paginate_queryset(request, items, per_page=12)
    return render(
        request,
        "gallery/upload_image.html",
        {"form": form, "items": page_obj.object_list, "page_obj": page_obj, "paginator": paginator,
         "can_configure": slider.can_configure(request.user)},
    )


@login_required
def item_edit(request: HttpRequest, pk: int) -> HttpResponse:
    _require(slider.can_upload, request.user)
    item = get_object_or_404(GalleryItem, pk=pk)
    if request.method == "POST":
        form = GalleryItemForm(request.POST, request.FILES, instance=item)
        if form.is_valid():
            item = form.save()
            messages.success(request, f"Zapisano zmiany w „{item.title}”.")
            return redirect(f"{reverse('gallery:upload_image')}#item-{item.pk}")
        messages.error(request, "Popraw zaznaczone pola formularza.")
    else:
        form = GalleryItemForm(instance=item)
    return render(request, "gallery/item_form.html", {"form": form, "item": item})


@login_required
@require_POST
def item_delete(request: HttpRequest, pk: int) -> HttpResponse:
    _require(slider.can_upload, request.user)
    item = get_object_or_404(GalleryItem, pk=pk)
    title = item.title
    presets = list(SliderPreset.objects.filter(preset_items__gallery_item=item))
    item.image.delete(save=False)
    item.delete()
    for preset in presets:
        slider._renumber(slider.ordered_items(preset))
    messages.success(request, f"Usunięto „{title}” z bazy i ze wszystkich układów.")
    return redirect("gallery:upload_image")


# ---------------------------------------------------------------------------
# 2. Konfigurator układów (administrator)
# ---------------------------------------------------------------------------

def _preset_url(preset: SliderPreset, anchor: str = "") -> str:
    return f"{reverse('gallery:slider_configurator')}?uklad={preset.pk}{anchor}"


@login_required
def slider_configurator(request: HttpRequest) -> HttpResponse:
    """Wybór układu do edycji, karty w układzie (przeciąganie i ↑↓) oraz zdjęcia z bazy do dodania."""
    _require(slider.can_configure, request.user)
    presets = list(SliderPreset.objects.order_by("-is_active", "name"))
    current = None
    requested = request.GET.get("uklad")
    if requested and requested.isdigit():
        current = next((p for p in presets if p.pk == int(requested)), None)
    if current is None and presets:
        current = next((p for p in presets if p.is_active), presets[0])

    context = {"presets": presets, "current": current, "new_form": SliderPresetForm(prefix="new")}
    if current is not None:
        items = slider.ordered_items(current)
        used = {item.gallery_item_id for item in items}
        library = GalleryItem.objects.exclude(pk__in=used).select_related("related_course")
        page_obj, paginator = paginate_queryset(request, library, per_page=12, page_param="baza")
        context.update(
            items=items,
            library=page_obj.object_list,
            page_obj=page_obj,
            paginator=paginator,
            edit_form=SliderPresetForm(instance=current, prefix="edit"),
        )
    return render(request, "gallery/slider_configurator.html", context)


@login_required
@require_POST
def preset_create(request: HttpRequest) -> HttpResponse:
    _require(slider.can_configure, request.user)
    form = SliderPresetForm(request.POST, prefix="new")
    if form.is_valid():
        preset = form.save(commit=False)
        preset.created_by = request.user
        preset.is_active = not SliderPreset.objects.exists()
        preset.save()
        messages.success(request, f"Utworzono układ „{preset.name}”. Dodaj do niego zdjęcia.")
        return redirect(_preset_url(preset))
    errors = "; ".join(e for errs in form.errors.values() for e in errs)
    messages.error(request, f"Nie utworzono układu: {errors}")
    return redirect("gallery:slider_configurator")


def _preset(pk: int) -> SliderPreset:
    return get_object_or_404(SliderPreset, pk=pk)


@login_required
@require_POST
def preset_update(request: HttpRequest, pk: int) -> HttpResponse:
    _require(slider.can_configure, request.user)
    preset = _preset(pk)
    form = SliderPresetForm(request.POST, instance=preset, prefix="edit")
    if form.is_valid():
        form.save()
        messages.success(request, f"Zapisano ustawienia układu „{preset.name}”.")
    else:
        errors = "; ".join(e for errs in form.errors.values() for e in errs)
        messages.error(request, f"Nie zapisano: {errors}")
    return redirect(_preset_url(preset, "#ustawienia"))


@login_required
@require_POST
def preset_activate(request: HttpRequest, pk: int) -> HttpResponse:
    _require(slider.can_configure, request.user)
    preset = _preset(pk)
    preset.is_active = True
    preset.save()
    messages.success(request, f"Galeria wyświetla teraz układ „{preset.name}”.")
    return redirect(_preset_url(preset))


@login_required
@require_POST
def preset_delete(request: HttpRequest, pk: int) -> HttpResponse:
    _require(slider.can_configure, request.user)
    preset = _preset(pk)
    name, was_active = preset.name, preset.is_active
    preset.delete()
    messages.success(request, f"Usunięto układ „{name}”." + (" Galeria nie wyświetla teraz żadnego układu — wybierz inny." if was_active else ""))
    return redirect("gallery:slider_configurator")


@login_required
@require_POST
def preset_add_item(request: HttpRequest, pk: int) -> HttpResponse:
    """Dodanie zdjęcia z bazy do układu — przyciskiem (na koniec) albo przeciągnięciem (na pozycję)."""
    _require(slider.can_configure, request.user)
    preset = _preset(pk)
    if _wants_json(request):
        try:
            payload = json.loads(request.body or b"{}")
            item_id, position = int(payload["item"]), payload.get("position")
            position = int(position) if position is not None else None
        except (ValueError, KeyError, TypeError):
            return JsonResponse({"error": "Nieprawidłowe dane."}, status=400)
    else:
        item_id, position = request.POST.get("item", ""), None
        if not str(item_id).isdigit():
            raise Http404
    item = get_object_or_404(GalleryItem, pk=item_id)
    created = slider.add_item(preset, item, position)
    if _wants_json(request):
        if created is None:
            return JsonResponse({"error": "To zdjęcie już jest w układzie."}, status=409)
        position_label = "na koniec" if position is None else f"na pozycję {min(position, preset.preset_items.count() - 1) + 1}"
        messages.success(request, f"Dodano „{item.title}” {position_label} układu.")
        return JsonResponse({"ok": True, "count": preset.preset_items.count()})
    if created:
        messages.success(request, f"Dodano „{item.title}” na koniec układu.")
    else:
        messages.info(request, "To zdjęcie już jest w układzie.")
    return redirect(_preset_url(preset, f"#card-{item.pk}"))


@login_required
@require_POST
def preset_remove_item(request: HttpRequest, pk: int, item_pk: int) -> HttpResponse:
    _require(slider.can_configure, request.user)
    preset = _preset(pk)
    item = get_object_or_404(GalleryItem, pk=item_pk)
    slider.remove_item(preset, item)
    messages.success(request, f"Usunięto „{item.title}” z układu (zdjęcie zostaje w bazie).")
    return redirect(_preset_url(preset, "#karty"))


@login_required
@require_POST
def preset_move_item(request: HttpRequest, pk: int, item_pk: int, direction: str) -> HttpResponse:
    _require(slider.can_configure, request.user)
    if direction not in {"up", "down"}:
        raise Http404
    preset = _preset(pk)
    item = get_object_or_404(GalleryItem, pk=item_pk)
    if slider.move_item(preset, item, direction):
        messages.success(request, f"Przesunięto „{item.title}” {'wyżej' if direction == 'up' else 'niżej'}.")
    return redirect(_preset_url(preset, f"#card-{item.pk}"))


@login_required
@require_POST
def slider_reorder(request: HttpRequest, pk: int) -> JsonResponse:
    """Kolejność po przeciągnięciu: {"items": [id zdjęcia, …]} — aktualizuje pole ``order``."""
    _require(slider.can_configure, request.user)
    preset = _preset(pk)
    try:
        ids = [int(value) for value in json.loads(request.body or b"{}")["items"]]
    except (ValueError, KeyError, TypeError):
        return JsonResponse({"error": "Nieprawidłowe dane."}, status=400)
    if not slider.reorder(preset, ids):
        return JsonResponse({"error": "Lista kart nie zgadza się z układem — odśwież stronę."}, status=409)
    return JsonResponse({"ok": True})


def manage_presets(request: HttpRequest) -> HttpResponse:
    """Dawny adres listy układów — zarządzanie jest teraz w konfiguratorze."""
    return redirect(f"{reverse('gallery:slider_configurator')}#uklady")


# ---------------------------------------------------------------------------
# 3. Slajder publiczny
# ---------------------------------------------------------------------------

def slider_view(request: HttpRequest) -> HttpResponse:
    preset = SliderPreset.get_active()
    items = slider.ordered_items(preset) if preset else []
    return render(
        request,
        "gallery/slider_view.html",
        {
            "preset": preset,
            "items": items,
            "can_upload": slider.can_upload(request.user),
            "can_configure": slider.can_configure(request.user),
        },
    )
