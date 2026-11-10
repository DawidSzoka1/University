from django import forms

from apps.chatbot.services import MAX_MESSAGE_LENGTH
from apps.core.forms import AccessibleErrorsMixin


class ChatMessageForm(AccessibleErrorsMixin, forms.Form):
    """Pytanie do asystenta (strona rozmowy, bez JavaScriptu)."""

    message_text = forms.CharField(
        label='Twoje pytanie do asystenta',
        max_length=MAX_MESSAGE_LENGTH,
        help_text=f'Najwyżej {MAX_MESSAGE_LENGTH} znaków. Odpowiedź może potrwać kilkanaście sekund.',
        widget=forms.Textarea(attrs={
            'rows': 1,
            'class': 'form-control chat-form__input',
            'placeholder': 'Twoje pytanie…',
        }),
        error_messages={'required': 'Wpisz pytanie.'},
    )

    def clean_message_text(self) -> str:
        text = self.cleaned_data['message_text'].strip()
        if not text:
            raise forms.ValidationError('Wpisz pytanie.')
        return text
