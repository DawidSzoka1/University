from django import forms

from apps.catalog.models import Course
from apps.core.forms import AccessibleErrorsMixin
from apps.core.images import validate_image_upload
from apps.core.widgets import ImagePreviewInput

from .models import GalleryItem, SliderPreset

GALLERY_UPLOAD_MAX_MB = 10


class GalleryItemForm(AccessibleErrorsMixin, forms.ModelForm):
    """Zdjęcie do bazy: tytuł, pełny opis, tekst alternatywny i opcjonalny kurs."""

    related_course = forms.ModelChoiceField(
        queryset=Course.objects.none(),
        required=False,
        label="Powiązany kurs (opcjonalnie)",
        empty_label="Brak",
        help_text="Karta w slajderze dostanie link do kursu.",
        widget=forms.Select(attrs={"class": "form-control", "data-select-enhance": ""}),
    )

    class Meta:
        model = GalleryItem
        fields = ["title", "image", "image_alt", "description", "related_course"]
        widgets = {
            "title": forms.TextInput(attrs={"class": "form-control"}),
            "image": ImagePreviewInput(max_mb=GALLERY_UPLOAD_MAX_MB, max_size=GalleryItem.MAX_SIZE),
            "image_alt": forms.TextInput(attrs={"class": "form-control"}),
            "description": forms.Textarea(attrs={"class": "form-control", "rows": 5}),
        }
        help_texts = {
            "image": f"JPEG, PNG lub WebP do {GALLERY_UPLOAD_MAX_MB} MB. Większe zdjęcie zostanie automatycznie przeskalowane do maks. 1600×1200 px.",
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["related_course"].queryset = Course.objects.published().order_by("title")
        self.fields["image_alt"].required = True
        if self.instance.pk:
            self.fields["image"].required = False
            # Bez opcji „wyczyść” — karta w slajderze zawsze ma zdjęcie
            self.fields["image"].widget = ImagePreviewInput(
                max_mb=GALLERY_UPLOAD_MAX_MB, max_size=GalleryItem.MAX_SIZE, allow_clear=False, show_current=False
            )
            self.fields["image"].help_text = "Zostaw puste, żeby zachować obecne zdjęcie."

    def clean_image(self):
        image = self.cleaned_data.get("image")
        if image and hasattr(image, "content_type"):
            validate_image_upload(image, max_mb=GALLERY_UPLOAD_MAX_MB)
        return image

    def clean_description(self):
        description = (self.cleaned_data.get("description") or "").strip()
        if not description:
            raise forms.ValidationError("Dodaj opis — wyświetla się pod zdjęciem na karcie.")
        return description


class SliderPresetForm(AccessibleErrorsMixin, forms.ModelForm):
    """Nazwa i opcje układu slajdera."""

    class Meta:
        model = SliderPreset
        fields = ["name", "autoplay", "interval_seconds"]
        widgets = {
            "name": forms.TextInput(attrs={"class": "form-control", "placeholder": "np. Promocja jesienna"}),
            "autoplay": forms.CheckboxInput(attrs={"class": "form-checkbox"}),
            "interval_seconds": forms.NumberInput(attrs={"class": "form-control", "min": 3, "max": 30}),
        }
