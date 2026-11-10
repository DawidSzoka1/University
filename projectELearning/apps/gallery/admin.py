from django.contrib import admin
from unfold.admin import ModelAdmin, TabularInline

from .models import GalleryItem, SliderPreset, SliderPresetItem


@admin.register(GalleryItem)
class GalleryItemAdmin(ModelAdmin):
    """Admin for gallery images."""

    list_display = ('title', 'related_course', 'width', 'height', 'uploaded_at')
    search_fields = ('title', 'description')
    list_filter = ('uploaded_at', 'related_course')
    raw_id_fields = ('related_course', 'uploaded_by')
    readonly_fields = ('original_width', 'original_height', 'width', 'height')


class SliderPresetItemInline(TabularInline):
    """Inline for ordering gallery items within a slider preset."""

    model = SliderPresetItem
    extra = 1
    raw_id_fields = ('gallery_item',)
    ordering = ('order',)


@admin.register(SliderPreset)
class SliderPresetAdmin(ModelAdmin):
    """Admin for slider presets with inline item ordering."""

    list_display = ('name', 'is_active', 'item_count', 'created_at')
    list_editable = ('is_active',)
    inlines = [SliderPresetItemInline]
    search_fields = ('name',)

    @admin.display(description='Karty')
    def item_count(self, obj: SliderPreset) -> int:
        """Return the number of items in this preset."""
        return obj.preset_items.count()
