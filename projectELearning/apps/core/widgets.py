"""Pole przesyłania zdjęcia z podglądem: obecne zdjęcie widać od razu (także bez JS),
a nowo wybrane pokazuje ``static/js/image_preview.js`` jeszcze przed wysłaniem formularza."""

from __future__ import annotations

from django import forms

IMAGE_ACCEPT = "image/jpeg,image/png,image/webp"


class ImagePreviewInput(forms.ClearableFileInput):
    template_name = "core/widgets/image_input.html"

    def __init__(self, attrs=None, *, max_mb: int | None = None, max_size: tuple[int, int] | None = None,
                 allow_clear: bool = True, show_current: bool = True):
        attrs = {"class": "form-control", "accept": IMAGE_ACCEPT, **(attrs or {})}
        # Limity tylko do podpowiedzi w podglądzie — o przyjęciu pliku decyduje walidacja na serwerze.
        if max_mb:
            attrs["data-max-mb"] = str(max_mb)
        if max_size:
            attrs["data-max-size"] = f"{max_size[0]}x{max_size[1]}"
        super().__init__(attrs)
        self.allow_clear = allow_clear
        self.show_current = show_current

    def get_context(self, name, value, attrs):
        context = super().get_context(name, value, attrs)
        context["widget"].update({"allow_clear": self.allow_clear, "show_current": self.show_current})
        return context
