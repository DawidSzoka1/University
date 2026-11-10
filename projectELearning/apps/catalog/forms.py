from decimal import Decimal

from django import forms
from django_ckeditor_5.widgets import CKEditor5Widget
from mptt.forms import TreeNodeChoiceField

from apps.core.forms import AccessibleErrorsMixin
from apps.core.images import MAX_UPLOAD_MB, validate_image_upload
from apps.core.widgets import ImagePreviewInput

from .models import RATING_CHOICES, Category, Comment, Course, CourseSubmissionMessage


class CourseForm(AccessibleErrorsMixin, forms.ModelForm):
    """Tworzenie i edycja kursu. Opis HTML z CKEditora czyści ``bleach`` w ``Course.save()``."""

    category = TreeNodeChoiceField(
        queryset=Category.objects.filter(is_active=True),
        label="Kategoria",
        level_indicator="—",
        empty_label="Wybierz kategorię",
        widget=forms.Select(attrs={"class": "form-control", "data-select-enhance": ""}),
    )
    price = forms.DecimalField(
        label="Cena (zł)",
        min_value=Decimal("0.00"),
        max_value=Decimal("9999.99"),
        decimal_places=2,
        initial=Decimal("0.00"),
        help_text="Wpisz 0, aby kurs był bezpłatny.",
        widget=forms.NumberInput(attrs={"class": "form-control", "step": "0.01", "inputmode": "decimal"}),
    )

    class Meta:
        model = Course
        fields = [
            "title",
            "category",
            "description",
            "description_html",
            "difficulty_level",
            "duration_hours",
            "price",
            "thumbnail",
            "has_certificate",
            "recommended_styles",
        ]
        labels = {
            "title": "Tytuł kursu",
            "description": "Krótki opis",
            "description_html": "O kursie",
            "difficulty_level": "Poziom",
            "duration_hours": "Szacowany czas nauki (godziny)",
            "thumbnail": "Miniatura",
            "has_certificate": "Wystawiaj dyplom ukończenia (PDF)",
        }
        help_texts = {
            "description": "Jedno–dwa zdania widoczne w katalogu.",
            "description_html": "Dla kogo jest kurs i czego się nauczysz. Lekcje dodasz w następnym kroku. Pasek narzędzi: Alt+F10.",
            "thumbnail": "JPEG, PNG lub WebP, do 5 MB. Zdjęcie zostanie przeskalowane do 1280×720.",
        }
        widgets = {
            "title": forms.TextInput(attrs={"class": "form-control", "autocomplete": "off"}),
            "description": forms.Textarea(attrs={"class": "form-control", "rows": 3, "maxlength": 300}),
            "description_html": CKEditor5Widget(attrs={"class": "django_ckeditor_5"}, config_name="course"),
            "difficulty_level": forms.Select(attrs={"class": "form-control", "data-select-enhance": ""}),
            "duration_hours": forms.NumberInput(attrs={"class": "form-control", "min": 1, "max": 500}),
            "thumbnail": ImagePreviewInput(max_mb=MAX_UPLOAD_MB, max_size=(1280, 720)),
            "has_certificate": forms.CheckboxInput(attrs={"class": "form-checkbox"}),
            "recommended_styles": forms.CheckboxSelectMultiple(attrs={"class": "form-checkbox"}),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from apps.learning.models import LearningStyle

        self.fields["recommended_styles"].queryset = LearningStyle.objects.filter(is_active=True)
        self.fields["recommended_styles"].label = "Kurs szczególnie dla stylów uczenia się (opcjonalnie)"
        self.fields["recommended_styles"].help_text = (
            "Zaznacz, jeśli forma kursu szczególnie pasuje do tych stylów (np. dużo filmów → wzrokowiec). "
            "Kursy są polecane także na podstawie wersji lekcji dla stylów."
        )
        self.fields["title"].required = True
        self.fields["description"].required = True
        self.fields["duration_hours"].min_value = 1
        self.fields["duration_hours"].widget.attrs["min"] = 1

    def clean_thumbnail(self):
        thumbnail = self.cleaned_data.get("thumbnail")
        # Nowy plik (nie ten już zapisany przy kursie) przechodzi walidację Pillow
        if thumbnail and hasattr(thumbnail, "content_type"):
            validate_image_upload(thumbnail)
        return thumbnail

    def save(self, commit: bool = True) -> Course:
        course = super().save(commit=False)
        # Cena decyduje o darmowości — inaczej kurs raz darmowy nie mógłby dostać ceny.
        course.is_free = course.price == Decimal("0.00")
        if commit:
            course.save()
        return course


class CommentForm(AccessibleErrorsMixin, forms.ModelForm):
    """Komentarz, odpowiedź albo jego edycja. Ocena kursu tylko w komentarzu głównym
    uczestnika (``allow_rating``) — w pozostałych przypadkach pola nie ma."""

    rating = forms.TypedChoiceField(
        label="Ocena kursu (opcjonalnie)",
        choices=[("", "Bez oceny"), *RATING_CHOICES],
        coerce=int,
        empty_value=None,
        required=False,
        widget=forms.RadioSelect,
    )

    class Meta:
        model = Comment
        fields = ["content", "rating"]
        labels = {"content": "Treść komentarza"}
        widgets = {
            "content": forms.Textarea(attrs={"class": "form-control", "rows": 4, "maxlength": 3000}),
        }

    def __init__(self, *args, allow_rating: bool = False, is_reply: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        if is_reply:
            self.fields["content"].label = "Treść odpowiedzi"
        if not allow_rating:
            del self.fields["rating"]

    def clean_content(self) -> str:
        content = (self.cleaned_data.get("content") or "").strip()
        if len(content) < 2:
            raise forms.ValidationError("Napisz co najmniej 2 znaki.")
        if len(content) > 3000:
            raise forms.ValidationError("Komentarz może mieć najwyżej 3000 znaków.")
        return content


class CourseSubmissionMessageForm(AccessibleErrorsMixin, forms.ModelForm):
    """Wiadomość w wątku zgłoszenia kursu (autor ↔ nauczyciele kategorii)."""

    class Meta:
        model = CourseSubmissionMessage
        fields = ["content"]
        labels = {"content": "Wiadomość"}
        widgets = {
            "content": forms.Textarea(attrs={"class": "form-control", "rows": 4, "maxlength": 3000}),
        }

    def clean_content(self) -> str:
        content = (self.cleaned_data.get("content") or "").strip()
        if not content:
            raise forms.ValidationError("Wpisz treść wiadomości.")
        return content


class SubmissionRejectForm(AccessibleErrorsMixin, forms.Form):
    """Odrzucenie zgłoszenia wymaga powodu — autor zobaczy go przy kursie i w wątku."""

    reason = forms.CharField(
        label="Powód odrzucenia",
        help_text="Napisz, co autor powinien poprawić. Zobaczy to przy kursie i w wątku zgłoszenia.",
        max_length=2000,
        widget=forms.Textarea(attrs={"class": "form-control", "rows": 3, "maxlength": 2000}),
    )

    def clean_reason(self) -> str:
        reason = self.cleaned_data["reason"].strip()
        if len(reason) < 5:
            raise forms.ValidationError("Opisz powód w co najmniej 5 znakach.")
        return reason


class CategoryForm(AccessibleErrorsMixin, forms.ModelForm):
    """Kategoria lub podkategoria. Nauczyciel wybiera rodzica tylko ze swojej gałęzi;
    przydział nauczycieli (moderatorów) zmienia wyłącznie administrator."""

    parent = TreeNodeChoiceField(
        queryset=Category.objects.none(),
        label="Kategoria nadrzędna",
        level_indicator="—",
        required=False,
        empty_label="Brak — kategoria główna",
        widget=forms.Select(attrs={"class": "form-control", "data-select-enhance": ""}),
    )

    class Meta:
        model = Category
        fields = ["name", "parent", "description", "is_active", "moderators"]
        labels = {
            "name": "Nazwa",
            "description": "Opis",
            "is_active": "Widoczna w katalogu",
            "moderators": "Nauczyciele (moderatorzy) kategorii",
        }
        help_texts = {
            "description": "Krótko: jakie kursy tu trafiają. Widać to w katalogu.",
            "moderators": "Akceptują kursy i komentarze w tej kategorii i jej podkategoriach.",
        }
        widgets = {
            "name": forms.TextInput(attrs={"class": "form-control", "maxlength": 120, "autocomplete": "off"}),
            "description": forms.Textarea(attrs={"class": "form-control", "rows": 3}),
            "is_active": forms.CheckboxInput(attrs={"class": "form-checkbox"}),
            "moderators": forms.CheckboxSelectMultiple(attrs={"class": "form-checkbox"}),
        }

    def __init__(self, *args, user, parent_choices, **kwargs):
        super().__init__(*args, **kwargs)
        from apps.catalog.permissions import is_platform_admin

        self.user = user
        choices = parent_choices
        if self.instance.pk:
            # Kategoria nie może trafić pod samą siebie ani pod własną podkategorię
            choices = choices.exclude(pk__in=self.instance.get_descendants(include_self=True).values("pk"))
        self.fields["parent"].queryset = choices
        if is_platform_admin(user):
            from django.contrib.auth import get_user_model

            self.fields["moderators"].queryset = get_user_model().objects.filter(role="moderator", is_active=True).order_by("last_name", "username")
            self.fields["moderators"].label_from_instance = lambda u: f"{u.get_full_name() or u.username} ({u.username})"
            self.fields["moderators"].required = False
        else:
            del self.fields["moderators"]
            self.fields["parent"].required = True
            self.fields["parent"].empty_label = None

    def clean_name(self) -> str:
        name = self.cleaned_data["name"].strip()
        if len(name) < 2:
            raise forms.ValidationError("Nazwa musi mieć co najmniej 2 znaki.")
        return name

    def clean(self):
        cleaned = super().clean()
        name, parent = cleaned.get("name"), cleaned.get("parent")
        if name:
            siblings = Category.objects.filter(parent=parent, name__iexact=name)
            if self.instance.pk:
                siblings = siblings.exclude(pk=self.instance.pk)
            if siblings.exists():
                self.add_error("name", "W tym miejscu drzewa jest już kategoria o tej nazwie.")
        return cleaned
