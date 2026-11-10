from django.contrib import admin
from django.utils.html import format_html
from unfold.admin import ModelAdmin

from apps.core.forms import ColorThemeForm
from apps.core.models import ColorTheme, Notification


@admin.register(ColorTheme)
class ColorThemeAdmin(ModelAdmin):
    form = ColorThemeForm
    list_display = (
        "name",
        "primary_swatch",
        "secondary_swatch",
        "accent_swatch",
        "surface_swatch",
        "surface_variant_swatch",
        "mode_label",
        "is_default",
    )
    list_editable = ("is_default",)
    search_fields = ("name",)
    readonly_fields = ("preview",)
    fieldsets = (
        (
            "Definicja motywu",
            {
                "fields": (
                    "name",
                    "primary",
                    "secondary",
                    "accent",
                    "surface",
                    "surface_variant",
                    "is_default",
                ),
                "description": (
                    "Pięć kolorów bazowych steruje całą witryną. Kolor tekstu "
                    "i tryb ciemny są wyliczane automatycznie z kontrastu WCAG."
                ),
            },
        ),
        ("Podgląd", {"fields": ("preview",)}),
    )

    # --- Kolumny listy -------------------------------------------------

    @admin.display(description="Kolor główny")
    def primary_swatch(self, obj: ColorTheme) -> str:
        return self._swatch(obj.primary)

    @admin.display(description="Kolor dodatkowy")
    def secondary_swatch(self, obj: ColorTheme) -> str:
        return self._swatch(obj.secondary)

    @admin.display(description="Kolor akcentowy")
    def accent_swatch(self, obj: ColorTheme) -> str:
        return self._swatch(obj.accent)

    @admin.display(description="Tło")
    def surface_swatch(self, obj: ColorTheme) -> str:
        return self._swatch(obj.surface)

    @admin.display(description="Tło kontenerów")
    def surface_variant_swatch(self, obj: ColorTheme) -> str:
        return self._swatch(obj.surface_variant)

    @admin.display(description="Tryb")
    def mode_label(self, obj: ColorTheme) -> str:
        return "ciemny" if obj.is_dark else "jasny"

    # --- Podgląd w formularzu -----------------------------------------

    @admin.display(description="Podgląd motywu")
    def preview(self, obj: ColorTheme | None) -> str:
        if obj is None or not obj.pk:
            return "Podgląd pojawi się po zapisaniu motywu."
        palette = obj.palette
        return format_html(
            '<div style="display:flex;flex-wrap:wrap;gap:.75rem;align-items:stretch;'
            'max-width:44rem;font-size:.85rem;">'
            '<div style="flex:1 1 8rem;min-height:5rem;border-radius:.5rem;'
            'padding:.75rem;background:{surface};color:{on_surface};'
            'border:1px solid {border};display:flex;flex-direction:column;'
            'justify-content:space-between;">'
            "<strong>Tło strony</strong><span>{surface}</span></div>"
            '<div style="flex:1 1 8rem;min-height:5rem;border-radius:.5rem;'
            'padding:.75rem;background:{surface_variant};color:{on_surface_variant};'
            'border:1px solid {border};display:flex;flex-direction:column;'
            'justify-content:space-between;">'
            "<strong>Tło kontenerów</strong><span>{surface_variant}</span></div>"
            '<div style="flex:1 1 8rem;min-height:5rem;border-radius:.5rem;'
            'padding:.75rem;background:{primary};color:{on_primary};'
            "display:flex;flex-direction:column;justify-content:space-between;\">"
            "<strong>Kolor główny</strong><span>{primary}</span></div>"
            '<div style="flex:1 1 8rem;min-height:5rem;border-radius:.5rem;'
            'padding:.75rem;background:{secondary};color:{on_secondary};'
            "display:flex;flex-direction:column;justify-content:space-between;\">"
            "<strong>Kolor dodatkowy</strong><span>{secondary}</span></div>"
            '<div style="flex:1 1 8rem;min-height:5rem;border-radius:.5rem;'
            'padding:.75rem;background:{accent};color:{on_accent};'
            "display:flex;flex-direction:column;justify-content:space-between;\">"
            "<strong>Kolor akcentowy</strong><span>{accent}</span></div>"
            "</div>",
            surface=palette["surface"],
            on_surface=palette["on_surface"],
            surface_variant=palette["surface_variant"],
            on_surface_variant=palette["on_surface_variant"],
            primary=palette["primary"],
            on_primary=palette["on_primary"],
            secondary=palette["secondary"],
            on_secondary=palette["on_secondary"],
            accent=palette["accent"],
            on_accent=palette["on_accent"],
            border="#808080",
        )

    # --- Pomocnicze -----------------------------------------------------

    @staticmethod
    def _swatch(color: str) -> str:
        return format_html(
            '<span style="display:inline-flex;align-items:center;gap:.4rem;">'
            '<span aria-hidden="true" style="width:1.1rem;height:1.1rem;'
            'border-radius:.25rem;border:1px solid #808080;background:{};"></span>'
            "<code>{}</code></span>",
            color,
            color,
        )


@admin.register(Notification)
class NotificationAdmin(ModelAdmin):
    list_display = ("title", "kind", "user", "is_read", "created_at")
    list_filter = ("kind", "is_read", "created_at")
    search_fields = ("title", "message", "user__username")
    raw_id_fields = ("user",)
