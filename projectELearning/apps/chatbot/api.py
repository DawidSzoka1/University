"""API asystenta chronione JWT (``Authorization: Bearer …``).

Widżet w przeglądarce pobiera krótkotrwały token przez ``chatbot:api_token`` (sesja + CSRF),
a potem rozmawia wyłącznie z tymi endpointami. Odpowiedź modelu przychodzi strumieniem NDJSON:
``start`` (zapisane pytanie) → ``delta`` (fragmenty) → ``done`` (zapisana odpowiedź) albo ``error``;
w trakcie testu jedna linia ``refusal``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

from django.contrib.auth.decorators import login_required
from django.http import HttpRequest, JsonResponse, StreamingHttpResponse
from django.urls import reverse
from django.views.decorators.http import require_POST
from rest_framework import status
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework_simplejwt.tokens import AccessToken

from apps.chatbot import llm
from apps.chatbot.conversation import cancel_turn, finish_turn, start_turn
from apps.chatbot.services import (
    ContextError,
    find_session,
    is_busy,
    lock_info,
    resolve_context,
    serialize_message,
    validate_text,
)
from apps.learning.quizzes import active_sitting

HISTORY_IN_WIDGET = 30

jwt_only = authentication_classes([JWTAuthentication])
authenticated = permission_classes([IsAuthenticated])


@login_required
@require_POST
def api_token(request: HttpRequest) -> JsonResponse:
    """Token dostępu JWT dla zalogowanego w przeglądarce ucznia (sesja + CSRF)."""
    token = AccessToken.for_user(request.user)
    return JsonResponse({"access": str(token), "expires_in": int(token.lifetime.total_seconds())})


def _context_from(data, user):
    return resolve_context(user, data.get("course"), data.get("lesson"), data.get("attempt"))


@api_view(["GET"])
@jwt_only
@authenticated
def api_history(request):
    """Ostatnie wiadomości rozmowy dla kursu z kontekstu + stan blokady testu."""
    try:
        context = _context_from(request.query_params, request.user)
    except ContextError as error:
        return Response({"detail": str(error)}, status=status.HTTP_404_NOT_FOUND)
    session = find_session(request.user, context.course)
    messages = []
    if session is not None:
        recent = session.messages.select_related("lesson").order_by("-timestamp", "-pk")[:HISTORY_IN_WIDGET]
        messages = [serialize_message(m) for m in reversed(recent)]
    return Response({
        "context": {
            "label": context.label,
            "course": context.course.pk if context.course else None,
            "session_url": reverse("chatbot:session_detail", args=[session.pk]) if session else "",
        },
        "lock": lock_info(active_sitting(request.user)),
        "messages": messages,
    })


def _line(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False) + "\n"


@api_view(["POST"])
@jwt_only
@authenticated
def api_send_message(request):
    """Pytanie do asystenta. Odpowiedź strumieniowana (NDJSON)."""
    try:
        context = _context_from(request.data, request.user)
        text = validate_text(request.data.get("message"))
    except ContextError as error:
        return Response({"detail": str(error)}, status=status.HTTP_404_NOT_FOUND)
    except ValueError as error:
        return Response({"detail": str(error)}, status=status.HTTP_400_BAD_REQUEST)
    if is_busy(request.user):
        return Response(
            {"detail": "Asystent odpowiada jeszcze na poprzednie pytanie. Poczekaj chwilę."},
            status=status.HTTP_429_TOO_MANY_REQUESTS,
        )

    turn = start_turn(request.user, context, text)
    session_url = reverse("chatbot:session_detail", args=[turn.session.pk])

    def stream() -> Iterator[str]:
        if turn.refused:
            yield _line({
                "type": "refusal",
                "messages": [serialize_message(m) for m in turn.refusal],
                "lock": turn.lock,
                "session_url": session_url,
            })
            return
        yield _line({"type": "start", "question": serialize_message(turn.question), "session_url": session_url})
        parts: list[str] = []
        try:
            for piece in llm.stream_reply(turn.prompt):
                parts.append(piece)
                yield _line({"type": "delta", "text": piece})
        except llm.AssistantUnavailable as error:
            cancel_turn(turn)
            yield _line({"type": "error", "detail": str(error)})
            return
        except GeneratorExit:
            # Uczeń przerwał odpowiedź — zachowujemy to, co już przyszło.
            reply = "".join(parts).strip()
            if reply:
                finish_turn(turn, reply + " […]")
            else:
                cancel_turn(turn)
            raise
        reply = "".join(parts).strip()
        if not reply:
            cancel_turn(turn)
            yield _line({"type": "error", "detail": "Asystent nie zwrócił odpowiedzi. Spróbuj zadać pytanie inaczej."})
            return
        yield _line({"type": "done", "message": serialize_message(finish_turn(turn, reply))})

    response = StreamingHttpResponse(stream(), content_type="application/x-ndjson; charset=utf-8")
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return response


@api_view(["POST"])
@jwt_only
@authenticated
def api_clear(request):
    """Usuwa rozmowę dla kursu z kontekstu (albo ogólną)."""
    try:
        context = _context_from(request.data, request.user)
    except ContextError as error:
        return Response({"detail": str(error)}, status=status.HTTP_404_NOT_FOUND)
    session = find_session(request.user, context.course)
    if session is not None:
        session.delete()
    return Response({"detail": "Rozmowa została wyczyszczona."})
