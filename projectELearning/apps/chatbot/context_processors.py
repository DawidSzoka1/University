"""Nazwa modelu asystenta widoczna w oknie czatu i na stronie rozmowy."""

from __future__ import annotations

from django.http import HttpRequest

from apps.chatbot.llm import model_label


def assistant(request: HttpRequest) -> dict:
    if not getattr(request, "user", None) or not request.user.is_authenticated:
        return {}
    return {"assistant_model": model_label()}
