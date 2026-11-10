from django import forms
from django.contrib.auth.forms import PasswordChangeForm, SetPasswordForm
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError

from apps.accounts.models import CustomUser
from apps.core.forms import AccessibleErrorsMixin
from apps.core.images import MAX_UPLOAD_MB, resize_image_field, validate_image_upload
from apps.core.models import ColorTheme
from apps.core.widgets import ImagePreviewInput

AVATAR_SIZE = (512, 512)


class LoginForm(AccessibleErrorsMixin, forms.Form):
    """Formularz logowania obsługujący nazwę użytkownika lub adres e-mail."""

    login = forms.CharField(
        label="Nazwa użytkownika lub e-mail",
        max_length=254,
        widget=forms.TextInput(
            attrs={
                "class": "form-control",
                "autocomplete": "username",
                "placeholder": "np. jan_kowalski lub jan@example.com",
                "autofocus": True,
            }
        ),
    )
    password = forms.CharField(
        label="Hasło",
        widget=forms.PasswordInput(
            render_value=True,
            attrs={
                "class": "form-control",
                "autocomplete": "current-password",
                "placeholder": "••••••••",
            },
        ),
    )
    remember_me = forms.BooleanField(
        label="Zapamiętaj sesję na tym urządzeniu",
        required=False,
        initial=True,
        widget=forms.CheckboxInput(attrs={"class": "form-checkbox"}),
    )


class CustomUserRegistrationForm(AccessibleErrorsMixin, forms.ModelForm):
    """Formularz rejestracji nowego studenta.

    Zapamiętuje wszystkie pola przy błędach walidacji (poza hasłami).
    """

    username = forms.CharField(
        label="Nazwa użytkownika",
        max_length=150,
        widget=forms.TextInput(
            attrs={
                "class": "form-control",
                "placeholder": "np. jan_kowalski",
                "autocomplete": "username",
            }
        ),
        help_text="Litery, cyfry oraz znaki @ . + - _ (bez spacji).",
    )
    email = forms.EmailField(
        required=True,
        label="Adres e-mail",
        widget=forms.EmailInput(
            attrs={
                "class": "form-control",
                "placeholder": "twoj@email.pl",
                "autocomplete": "email",
            }
        ),
    )
    first_name = forms.CharField(
        label="Imię",
        max_length=150,
        required=False,
        widget=forms.TextInput(
            attrs={
                "class": "form-control",
                "placeholder": "Jan",
                "autocomplete": "given-name",
            }
        ),
    )
    last_name = forms.CharField(
        label="Nazwisko",
        max_length=150,
        required=False,
        widget=forms.TextInput(
            attrs={
                "class": "form-control",
                "placeholder": "Kowalski",
                "autocomplete": "family-name",
            }
        ),
    )
    password1 = forms.CharField(
        label="Hasło",
        help_text="Co najmniej 8 znaków, nie same cyfry i niepodobne do loginu.",
        widget=forms.PasswordInput(
            render_value=False,
            attrs={
                "class": "form-control",
                "placeholder": "Min. 8 znaków",
                "autocomplete": "new-password",
            },
        ),
    )
    password2 = forms.CharField(
        label="Powtórz hasło",
        widget=forms.PasswordInput(
            render_value=False,
            attrs={
                "class": "form-control",
                "placeholder": "Powtórz hasło",
                "autocomplete": "new-password",
            },
        ),
    )

    class Meta:
        model = CustomUser
        fields = (
            "username",
            "email",
            "first_name",
            "last_name",
            "password1",
            "password2",
        )

    def clean_username(self) -> str:
        username = self.cleaned_data.get("username", "").strip()
        if CustomUser.objects.filter(username__iexact=username).exists():
            raise ValidationError("Użytkownik o takiej nazwie już istnieje.")
        return username

    def clean_email(self) -> str:
        email = self.cleaned_data.get("email", "").strip().lower()
        if CustomUser.objects.filter(email__iexact=email).exists():
            raise ValidationError("Użytkownik z tym adresem e-mail już istnieje.")
        return email

    def clean(self) -> dict:
        cleaned_data = super().clean()
        p1 = cleaned_data.get("password1")
        p2 = cleaned_data.get("password2")

        if p1 and p2:
            if p1 != p2:
                self.add_error("password2", "Podane hasła nie są identyczne.")
            else:
                try:
                    validate_password(p1)
                except ValidationError as exc:
                    self.add_error("password1", exc)
        return cleaned_data

    def save(self, commit: bool = True) -> CustomUser:
        user = super().save(commit=False)
        user.set_password(self.cleaned_data["password1"])
        user.role = CustomUser.Role.STUDENT
        if commit:
            user.save()
        return user


class CustomUserProfileForm(AccessibleErrorsMixin, forms.ModelForm):
    """Formularz edycji profilu użytkownika (dane osobowe + adres)."""

    first_name = forms.CharField(
        label="Imię",
        max_length=150,
        required=False,
        widget=forms.TextInput(
            attrs={"class": "form-control", "placeholder": "np. Jan", "autocomplete": "given-name"}
        ),
    )
    last_name = forms.CharField(
        label="Nazwisko",
        max_length=150,
        required=False,
        widget=forms.TextInput(
            attrs={"class": "form-control", "placeholder": "np. Kowalski", "autocomplete": "family-name"}
        ),
    )
    email = forms.EmailField(
        label="Adres e-mail",
        required=True,
        widget=forms.EmailInput(
            attrs={"class": "form-control", "placeholder": "twoj@email.pl", "autocomplete": "email"}
        ),
    )
    phone_number = forms.CharField(
        label="Numer telefonu",
        max_length=20,
        required=False,
        widget=forms.TextInput(
            attrs={"class": "form-control", "placeholder": "+48 123 456 789", "autocomplete": "tel"}
        ),
    )
    street_address = forms.CharField(
        label="Ulica i numer",
        max_length=255,
        required=False,
        widget=forms.TextInput(
            attrs={"class": "form-control", "placeholder": "ul. Przykładowa 1/2", "autocomplete": "street-address"}
        ),
    )
    postal_code = forms.CharField(
        label="Kod pocztowy",
        max_length=20,
        required=False,
        widget=forms.TextInput(
            attrs={"class": "form-control", "placeholder": "00-000", "autocomplete": "postal-code"}
        ),
    )
    city = forms.CharField(
        label="Miasto",
        max_length=100,
        required=False,
        widget=forms.TextInput(
            attrs={"class": "form-control", "placeholder": "Warszawa", "autocomplete": "address-level2"}
        ),
    )
    avatar = forms.ImageField(
        label="Zdjęcie profilowe (awatar)",
        required=False,
        help_text=f"JPEG, PNG lub WebP do {MAX_UPLOAD_MB} MB. Zdjęcie zostanie przeskalowane do {AVATAR_SIZE[0]}×{AVATAR_SIZE[1]} px.",
        widget=ImagePreviewInput(max_mb=MAX_UPLOAD_MB, max_size=AVATAR_SIZE),
    )

    class Meta:
        model = CustomUser
        fields = (
            'first_name',
            'last_name',
            'email',
            'avatar',
            'phone_number',
            'street_address',
            'postal_code',
            'city',
        )

    def clean_avatar(self):
        avatar = self.cleaned_data.get("avatar")
        if avatar and hasattr(avatar, "content_type"):
            validate_image_upload(avatar)
        return avatar

    def save(self, commit: bool = True) -> CustomUser:
        user = super().save(commit=False)
        resize_image_field(user.avatar, AVATAR_SIZE)
        if commit:
            user.save()
        return user


class ThemeSelectionForm(forms.Form):
    """Formularz wyboru motywu kolorystycznego przez użytkownika."""

    theme = forms.ModelChoiceField(
        queryset=ColorTheme.objects.all(),
        required=False,
        empty_label='Domyślny motyw',
        label='Motyw kolorystyczny',
        widget=forms.Select(attrs={'class': 'form-select'}),
    )


class _PasswordFieldsMixin(AccessibleErrorsMixin):
    """Pola hasła w stylu platformy (etykiety po polsku, autouzupełnianie menedżera haseł)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        labels = {"old_password": "Obecne hasło", "new_password1": "Nowe hasło", "new_password2": "Powtórz nowe hasło"}
        for name, field in self.fields.items():
            field.label = labels.get(name, field.label)
            field.widget.attrs.update({"class": "form-control"})
            field.widget.attrs["autocomplete"] = "current-password" if name == "old_password" else "new-password"
            field.widget.attrs.pop("autofocus", None)
        if "new_password1" in self.fields:
            self.fields["new_password1"].help_text = (
                "Co najmniej 8 znaków. Nie może być podobne do loginu ani e-maila, zbyt popularne "
                "ani składać się z samych cyfr."
            )


class AccessiblePasswordChangeForm(_PasswordFieldsMixin, PasswordChangeForm):
    pass


class AccessibleSetPasswordForm(_PasswordFieldsMixin, SetPasswordForm):
    pass
