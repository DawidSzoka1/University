"""Operacje na układzie slajdera. Kolejność zmienia się aktualizacją pola ``order`` istniejących
kart (bez usuwania i ponownego dodawania rekordów)."""

from __future__ import annotations

from django.db import transaction

from apps.catalog.permissions import is_platform_admin
from apps.gallery.models import GalleryItem, SliderPreset, SliderPresetItem


def can_upload(user) -> bool:
    """Zdjęcia do bazy dodają administratorzy i nauczyciele (moderatorzy)."""
    return user.is_authenticated and (is_platform_admin(user) or getattr(user, "role", "") == "moderator")


def can_configure(user) -> bool:
    """Układy slajdera ustawia administrator."""
    return is_platform_admin(user)


def ordered_items(preset: SliderPreset) -> list[SliderPresetItem]:
    return list(preset.preset_items.select_related("gallery_item", "gallery_item__related_course").order_by("order", "pk"))


def _renumber(items: list[SliderPresetItem]) -> None:
    changed = []
    for position, item in enumerate(items, start=1):
        if item.order != position:
            item.order = position
            changed.append(item)
    if changed:
        SliderPresetItem.objects.bulk_update(changed, ["order"])


@transaction.atomic
def reorder(preset: SliderPreset, gallery_item_ids: list[int]) -> bool:
    """Nowa kolejność z przeciągania. Musi zawierać dokładnie te same zdjęcia co układ."""
    items = {item.gallery_item_id: item for item in ordered_items(preset)}
    if sorted(gallery_item_ids) != sorted(items) or len(set(gallery_item_ids)) != len(gallery_item_ids):
        return False
    _renumber([items[item_id] for item_id in gallery_item_ids])
    return True


@transaction.atomic
def add_item(preset: SliderPreset, gallery_item: GalleryItem, position: int | None = None) -> SliderPresetItem | None:
    """Dodaje zdjęcie na wskazaną pozycję (0 = początek, None = koniec). None, jeśli już jest."""
    if preset.preset_items.filter(gallery_item=gallery_item).exists():
        return None
    items = ordered_items(preset)
    created = SliderPresetItem.objects.create(preset=preset, gallery_item=gallery_item, order=len(items) + 1)
    if position is not None and 0 <= position < len(items):
        items.insert(position, created)
        _renumber(items)
    return created


@transaction.atomic
def remove_item(preset: SliderPreset, gallery_item: GalleryItem) -> bool:
    deleted, _ = preset.preset_items.filter(gallery_item=gallery_item).delete()
    _renumber(ordered_items(preset))
    return bool(deleted)


@transaction.atomic
def move_item(preset: SliderPreset, gallery_item: GalleryItem, direction: str) -> bool:
    """Alternatywa klawiaturowa dla przeciągania: o jedno miejsce w górę/dół."""
    items = ordered_items(preset)
    index = next((i for i, item in enumerate(items) if item.gallery_item_id == gallery_item.pk), None)
    if index is None:
        return False
    target = index - 1 if direction == "up" else index + 1
    if not 0 <= target < len(items):
        return False
    items[index], items[target] = items[target], items[index]
    _renumber(items)
    return True
