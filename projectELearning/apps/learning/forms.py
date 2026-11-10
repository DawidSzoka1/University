from django import forms
from django.core.validators import MaxValueValidator
from django_ckeditor_5.widgets import CKEditor5Widget

from apps.core.forms import AccessibleErrorsMixin
from apps.core.images import MAX_UPLOAD_MB, resize_image_field, validate_image_upload
from apps.core.widgets import ImagePreviewInput
from apps.learning.media import validate_attachment, validate_video_file, validate_video_url
from apps.learning.models import Choice, CourseSection, LearningStyle, Lesson, LessonAttachment, LessonVariant, Question, Quiz


class SectionForm(AccessibleErrorsMixin, forms.ModelForm):
    class Meta:
        model = CourseSection
        fields = ["title"]
        labels = {"title": "Nazwa sekcji"}
        widgets = {
            "title": forms.TextInput(attrs={"class": "form-control", "placeholder": "np. Zmienne i typy danych"}),
        }


class MultipleFileInput(forms.ClearableFileInput):
    allow_multiple_selected = True


class MultipleFileField(forms.FileField):
    """Kilka plików w jednym polu (Django 5: widget z allow_multiple_selected)."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("widget", MultipleFileInput(attrs={"class": "form-control"}))
        super().__init__(*args, **kwargs)

    def clean(self, data, initial=None):
        single = super().clean
        if isinstance(data, (list, tuple)):
            return [single(item, initial) for item in data]
        return [single(data, initial)] if data else []


class LessonForm(AccessibleErrorsMixin, forms.ModelForm):
    """Lekcja: treść z edytora (zdjęcia, kod, tabele), film z linku albo pliku, pliki do pobrania."""

    new_attachments = MultipleFileField(
        label="Dodaj pliki do pobrania",
        required=False,
        help_text="Np. plik startowy zadania. Można wybrać kilka naraz; do 25 MB każdy.",
    )
    remove_attachments = forms.ModelMultipleChoiceField(
        label="Usuń pliki",
        queryset=LessonAttachment.objects.none(),
        required=False,
        widget=forms.CheckboxSelectMultiple(attrs={"class": "form-checkbox"}),
    )

    class Meta:
        model = Lesson
        fields = [
            "section", "title", "content_html", "video_url", "video_file", "duration_minutes", "is_preview",
            "same_for_all_styles",
        ]
        labels = {"section": "Sekcja"}
        help_texts = {
            "content_html": "Zdjęcie: przycisk obrazka (dodaj tekst alternatywny). Kod: blok kodu. Pasek: Alt+F10.",
        }
        widgets = {
            "section": forms.Select(attrs={"class": "form-control", "data-select-enhance": ""}),
            "title": forms.TextInput(attrs={"class": "form-control"}),
            "content_html": CKEditor5Widget(attrs={"class": "django_ckeditor_5"}, config_name="lesson"),
            "video_url": forms.URLInput(attrs={"class": "form-control", "placeholder": "https://www.youtube.com/watch?v=…"}),
            "video_file": forms.ClearableFileInput(attrs={"class": "form-control", "accept": "video/mp4,video/webm"}),
            "duration_minutes": forms.NumberInput(attrs={"class": "form-control", "min": 1, "max": 600}),
            "is_preview": forms.CheckboxInput(attrs={"class": "form-checkbox"}),
            "same_for_all_styles": forms.CheckboxInput(attrs={"class": "form-checkbox"}),
        }

    def __init__(self, *args, course, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["section"].queryset = course.sections.all()
        self.fields["section"].empty_label = None
        self.fields["title"].required = True
        self.fields["duration_minutes"].min_value = 1
        if self.instance.pk:
            self.fields["remove_attachments"].queryset = self.instance.attachments.all()
            self.fields["remove_attachments"].label_from_instance = lambda a: a.original_name
        else:
            del self.fields["remove_attachments"]

    def clean_video_url(self):
        url = self.cleaned_data.get("video_url", "").strip()
        validate_video_url(url)
        return url

    def clean_video_file(self):
        video = self.cleaned_data.get("video_file")
        if video and hasattr(video, "content_type"):
            validate_video_file(video)
        return video

    def clean_new_attachments(self):
        files = self.cleaned_data.get("new_attachments") or []
        errors = []
        for upload in files:
            try:
                validate_attachment(upload)
            except forms.ValidationError as error:
                errors.extend(error.messages)
        if errors:
            raise forms.ValidationError(errors)
        return files

    def clean(self):
        cleaned = super().clean()
        has_url = bool(cleaned.get("video_url"))
        video_file = cleaned.get("video_file")
        has_file = bool(video_file) and video_file is not False
        if has_url and has_file:
            self.add_error("video_url", "Wybierz jedno źródło filmu: link albo plik.")
        return cleaned

    def save(self, commit: bool = True) -> Lesson:
        lesson = super().save(commit=commit)
        if commit:
            for attachment in self.cleaned_data.get("remove_attachments") or []:
                attachment.file.delete(save=False)
                attachment.delete()
            for upload in self.cleaned_data.get("new_attachments") or []:
                LessonAttachment.objects.create(
                    lesson=lesson, file=upload, original_name=upload.name[:255], size=upload.size
                )
        return lesson


# ---------------------------------------------------------------------------
# Testy: ustawienia, pytania (z mediami) i odpowiedzi do wyboru
# ---------------------------------------------------------------------------

class QuizForm(AccessibleErrorsMixin, forms.ModelForm):
    """Ustawienia testu: miejsce w programie, czy blokuje dalszą naukę, próg i limit podejść."""

    after_lesson = forms.ModelChoiceField(
        queryset=Lesson.objects.none(),
        required=False,
        label="Miejsce w programie",
        empty_label="Na końcu kursu (test końcowy)",
        help_text="Test pojawi się w programie zaraz po wybranej lekcji.",
        widget=forms.Select(attrs={"class": "form-control", "data-select-enhance": ""}),
    )

    class Meta:
        model = Quiz
        fields = ["title", "description", "after_lesson", "is_required", "pass_percent", "max_attempts"]
        widgets = {
            "title": forms.TextInput(attrs={"class": "form-control", "placeholder": "np. Sprawdź się: zmienne i pętle"}),
            "description": forms.Textarea(attrs={"class": "form-control", "rows": 3}),
            "is_required": forms.CheckboxInput(attrs={"class": "form-checkbox"}),
            "pass_percent": forms.NumberInput(attrs={"class": "form-control", "min": 1, "max": 100}),
            "max_attempts": forms.NumberInput(attrs={"class": "form-control", "min": 0, "max": 50}),
        }
        help_texts = {
            "description": "Np. ile czasu zarezerwować i czego dotyczy test.",
            "pass_percent": "Ile procent punktów trzeba zdobyć, żeby zaliczyć.",
        }

    def __init__(self, *args, course, **kwargs):
        super().__init__(*args, **kwargs)
        from apps.learning.curriculum import build_outline

        self.fields["title"].required = True
        outline = build_outline(course)
        numbers = {entry.lesson.pk: entry.number for entry in outline.lessons}
        self.fields["after_lesson"].queryset = Lesson.objects.filter(section__course=course).order_by("section__order", "section__pk", "order", "pk")
        self.fields["after_lesson"].label_from_instance = lambda lesson: f"Po lekcji {numbers.get(lesson.pk, '')} {lesson.title}"
        self.fields["max_attempts"].max_value = 50


class QuestionForm(AccessibleErrorsMixin, forms.ModelForm):
    """Pytanie z opcjonalnym zdjęciem (z opisem), filmem, kodem i plikiem."""

    class Meta:
        model = Question
        fields = [
            "question_type", "answer_format", "text", "points",
            "image", "image_alt", "video_url", "video_file", "code", "attachment",
        ]
        widgets = {
            "question_type": forms.RadioSelect(attrs={"class": "form-radio"}),
            "answer_format": forms.RadioSelect(attrs={"class": "form-radio"}),
            "text": forms.Textarea(attrs={"class": "form-control", "rows": 3}),
            "points": forms.NumberInput(attrs={"class": "form-control", "min": 1, "max": 100}),
            "image": ImagePreviewInput(max_mb=MAX_UPLOAD_MB, max_size=(1600, 1600)),
            "image_alt": forms.TextInput(attrs={"class": "form-control"}),
            "video_url": forms.URLInput(attrs={"class": "form-control", "placeholder": "https://www.youtube.com/watch?v=…"}),
            "video_file": forms.ClearableFileInput(attrs={"class": "form-control", "accept": "video/mp4,video/webm"}),
            "code": forms.Textarea(attrs={"class": "form-control form-control--code", "rows": 6, "spellcheck": "false"}),
            "attachment": forms.ClearableFileInput(attrs={"class": "form-control"}),
        }
        help_texts = {
            "answer_format": "Dotyczy tylko pytań otwartych — w jakiej formie uczeń odpowiada.",
            "image": "JPEG, PNG lub WebP do 5 MB, skalowane do 1600 px.",
            "image_alt": "Co jest na zdjęciu — dla osób, które go nie widzą. Wymagane przy zdjęciu.",
            "code": "Kod pokazany uczniowi w bloku o stałej szerokości znaków.",
            "attachment": "Np. plik z danymi do zadania; do 25 MB.",
        }

    def clean_image(self):
        image = self.cleaned_data.get("image")
        if image and hasattr(image, "content_type"):
            validate_image_upload(image)
        return image

    def clean_video_url(self):
        url = (self.cleaned_data.get("video_url") or "").strip()
        validate_video_url(url)
        return url

    def clean_video_file(self):
        video = self.cleaned_data.get("video_file")
        if video and hasattr(video, "content_type"):
            validate_video_file(video)
        return video

    def clean_attachment(self):
        attachment = self.cleaned_data.get("attachment")
        if attachment and hasattr(attachment, "content_type"):
            validate_attachment(attachment)
        return attachment

    def clean(self):
        cleaned = super().clean()
        image = cleaned.get("image")
        if image and image is not False and not (cleaned.get("image_alt") or "").strip():
            self.add_error("image_alt", "Opisz zdjęcie — bez opisu osoba niewidoma nie zrozumie pytania.")
        video_file = cleaned.get("video_file")
        if cleaned.get("video_url") and video_file and video_file is not False:
            self.add_error("video_url", "Wybierz jedno źródło filmu: link albo plik.")
        return cleaned

    def save(self, commit: bool = True) -> Question:
        question = super().save(commit=False)
        if question.image and not getattr(question.image, "_committed", True):
            resize_image_field(question.image, (1600, 1600))
        attachment = self.cleaned_data.get("attachment")
        if attachment and hasattr(attachment, "name") and hasattr(attachment, "content_type"):
            question.attachment_name = attachment.name[:255]
        elif not question.attachment:
            question.attachment_name = ""
        if commit:
            question.save()
        return question


class BaseChoiceFormSet(forms.BaseInlineFormSet):
    """Odpowiedzi ABC: co najmniej dwie, przynajmniej jedna poprawna (przy jednokrotnym — dokładnie jedna)."""

    def __init__(self, *args, question_type: str = "", **kwargs):
        self.question_type = question_type
        super().__init__(*args, **kwargs)

    def clean(self):
        super().clean()
        if self.question_type == Question.QuestionType.OPEN or any(self.errors):
            return
        filled = [
            form.cleaned_data for form in self.forms
            if form.cleaned_data and not form.cleaned_data.get("DELETE") and (form.cleaned_data.get("text") or "").strip()
        ]
        if len(filled) < 2:
            raise forms.ValidationError("Podaj co najmniej dwie odpowiedzi do wyboru.")
        correct = sum(1 for data in filled if data.get("is_correct"))
        if correct == 0:
            raise forms.ValidationError("Zaznacz, która odpowiedź jest poprawna.")
        if self.question_type == Question.QuestionType.CLOSED_SINGLE and correct > 1:
            raise forms.ValidationError("W pytaniu jednokrotnego wyboru poprawna może być tylko jedna odpowiedź.")


def choice_formset(extra: int = 0):
    """Formset odpowiedzi ABC z ``extra`` pustymi wierszami (max 8 odpowiedzi)."""
    return forms.inlineformset_factory(
        Question,
        Choice,
        formset=BaseChoiceFormSet,
        fields=["text", "is_correct"],
        labels={"text": "Odpowiedź", "is_correct": "Poprawna"},
        widgets={
            "text": forms.TextInput(attrs={"class": "form-control"}),
            "is_correct": forms.CheckboxInput(attrs={"class": "form-checkbox"}),
        },
        extra=extra,
        max_num=8,
        can_delete=True,
    )


class GradeAnswerForm(forms.Form):
    """Ocena jednej odpowiedzi otwartej."""

    points = forms.IntegerField(label="Punkty", min_value=0, widget=forms.NumberInput(attrs={"class": "form-control"}))
    feedback = forms.CharField(
        label="Komentarz dla ucznia",
        required=False,
        max_length=3000,
        widget=forms.Textarea(attrs={"class": "form-control", "rows": 2}),
    )

    def __init__(self, *args, max_points: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["points"].max_value = max_points
        self.fields["points"].widget.attrs["max"] = max_points
        self.fields["points"].help_text = f"Od 0 do {max_points}."
        self.fields["points"].validators.append(MaxValueValidator(max_points))


# ---------------------------------------------------------------------------
# Style uczenia się
# ---------------------------------------------------------------------------

class LearningStyleChoiceForm(forms.Form):
    """Wybór własnego stylu uczenia się (albo brak — zawsze wersja podstawowa)."""

    style = forms.ModelChoiceField(
        label="Twój styl uczenia się",
        queryset=LearningStyle.objects.none(),
        required=False,
        empty_label="Nie wybieram — pokazuj zawsze wersję podstawową",
        widget=forms.RadioSelect,
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["style"].queryset = LearningStyle.objects.filter(is_active=True)


class LessonVariantForm(AccessibleErrorsMixin, forms.ModelForm):
    """Treść lekcji dla jednego stylu: tekst z edytora i opcjonalny film z linku."""

    class Meta:
        model = LessonVariant
        fields = ["content_html", "video_url"]
        help_texts = {
            "content_html": "Ta treść zastąpi treść podstawową u uczniów z tym stylem. Pliki do pobrania są wspólne.",
        }
        widgets = {
            "content_html": CKEditor5Widget(attrs={"class": "django_ckeditor_5"}, config_name="lesson"),
            "video_url": forms.URLInput(attrs={"class": "form-control", "placeholder": "https://www.youtube.com/watch?v=…"}),
        }

    def clean_content_html(self):
        from django.utils.html import strip_tags

        html = self.cleaned_data.get("content_html", "")
        if not strip_tags(html).strip() and "<img" not in html:
            raise forms.ValidationError("Wpisz treść tej wersji lekcji.")
        return html

    def clean_video_url(self):
        url = self.cleaned_data.get("video_url", "").strip()
        validate_video_url(url)
        return url
