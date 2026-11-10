"""Historia rozmów z asystentem (strony HTML). Okno czatu w rogu ekranu korzysta z ``api.py``;
formularz na stronie rozmowy działa także bez JavaScriptu."""

from __future__ import annotations

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db.models import Count, Max
from django.http import HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.decorators.http import require_POST

from apps.chatbot import llm
from apps.chatbot.conversation import cancel_turn, finish_turn, start_turn
from apps.chatbot.forms import ChatMessageForm
from apps.chatbot.models import ChatSession
from apps.chatbot.services import ChatContext, can_discuss_course, find_session, is_busy, lock_info
from apps.core.pagination import paginate_queryset
from apps.learning.quizzes import active_sitting

MESSAGES_PER_PAGE = 20
OTHER_CONVERSATIONS = 6  # skrót w bocznym panelu; pełna, paginowana lista: session_list


@login_required
def session_list(request: HttpRequest) -> HttpResponse:
    """Rozmowy ucznia: ogólna i po jednej na kurs, od ostatnio używanej."""
    sessions = (
        ChatSession.objects.filter(student=request.user)
        .select_related("course")
        .annotate(message_count=Count("messages"), last_message_at=Max("messages__timestamp"))
        .filter(message_count__gt=0)
        .order_by("-updated_at")
    )
    page_obj, paginator = paginate_queryset(request, sessions, per_page=10)
    for session in page_obj:
        session.last_message = session.messages.order_by("-timestamp", "-pk").first()
    return render(
        request,
        "chatbot/session_list.html",
        {"sessions": page_obj, "page_obj": page_obj, "paginator": paginator},
    )


@login_required
def session_detail(request: HttpRequest, pk: int) -> HttpResponse:
    session = get_object_or_404(ChatSession.objects.select_related("course"), pk=pk, student=request.user)
    return _conversation(request, session, session.course)


@login_required
def session_general(request: HttpRequest) -> HttpResponse:
    """Rozmowa ogólna (bez kursu) — powstaje dopiero przy pierwszym pytaniu."""
    return _conversation(request, find_session(request.user, None), None)


def _conversation(request: HttpRequest, session: ChatSession | None, course) -> HttpResponse:
    can_ask = course is None or can_discuss_course(request.user, course)
    form = ChatMessageForm(request.POST or None)
    if request.method == "POST":
        if not can_ask:
            messages.error(request, "Ten kurs nie jest już dostępny — możesz tylko przejrzeć historię.")
        elif form.is_valid():
            if is_busy(request.user):
                messages.warning(request, "Asystent odpowiada jeszcze na poprzednie pytanie. Poczekaj chwilę.")
            else:
                turn = start_turn(request.user, ChatContext(course=course), form.cleaned_data["message_text"])
                if turn.refused:
                    messages.warning(request, turn.lock["text"])
                    return redirect("chatbot:session_detail", pk=turn.session.pk)
                try:
                    finish_turn(turn, llm.complete_reply(turn.prompt))
                except llm.AssistantUnavailable as error:
                    cancel_turn(turn)
                    messages.error(request, str(error))
                else:
                    return redirect(f"{_detail_url(turn.session)}#najnowsza")

    page_obj = paginator = None
    if session is not None:
        conversation = session.messages.select_related("lesson").order_by("timestamp", "pk")
        if "page" not in request.GET:
            # Domyślnie ostatnia strona — najnowsze wiadomości.
            request.GET = request.GET.copy()
            request.GET["page"] = str(max(1, -(-conversation.count() // MESSAGES_PER_PAGE)))
        page_obj, paginator = paginate_queryset(request, conversation, per_page=MESSAGES_PER_PAGE)
    others = (
        ChatSession.objects.filter(student=request.user, messages__isnull=False)
        .select_related("course")
        .annotate(message_count=Count("messages"))
        .order_by("-updated_at")
    )
    return render(
        request,
        "chatbot/session_detail.html",
        {
            "conversations": others[:OTHER_CONVERSATIONS],
            "conversation_total": others.count(),
            "session": session,
            "course": course,
            "chat_messages": page_obj,
            "page_obj": page_obj,
            "paginator": paginator,
            "form": form,
            "can_ask": can_ask,
            "lock": lock_info(active_sitting(request.user)),
        },
    )


def _detail_url(session: ChatSession) -> str:
    return reverse("chatbot:session_detail", args=[session.pk])


@login_required
@require_POST
def session_clear(request: HttpRequest, pk: int) -> HttpResponse:
    session = get_object_or_404(ChatSession.objects.select_related("course"), pk=pk, student=request.user)
    title = session.title
    session.delete()
    messages.success(request, f"Usunięto rozmowę „{title}”.")
    return redirect("chatbot:session_list")
