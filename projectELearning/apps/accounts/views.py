from allauth.socialaccount.models import SocialAccount
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout, update_session_auth_hash
from django.contrib.auth.decorators import login_required
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils.http import url_has_allowed_host_and_scheme
from django.views.decorators.http import require_POST
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.tokens import AccessToken

from apps.accounts import lockout
from apps.accounts.adapter import PENDING_SOCIAL_KEY, get_pending
from apps.accounts.forms import (
    AccessiblePasswordChangeForm,
    AccessibleSetPasswordForm,
    CustomUserProfileForm,
    CustomUserRegistrationForm,
    LoginForm,
)
from apps.accounts.models import CustomUser
from apps.core.context_processors import SESSION_THEME_KEY
from apps.core.models import ColorTheme


@login_required
def profile_view(request: HttpRequest) -> HttpResponse:
    """Profil użytkownika — widok konta z informacjami, motywem i integracją kont społecznościowych."""
    social_qs = SocialAccount.objects.filter(user=request.user)
    connected_map = {}
    for acc in social_qs:
        connected_map[acc.provider] = acc
        if acc.provider == "bitbucket_oauth2":
            connected_map["bitbucket"] = acc

    providers_info = [
        {
            "id": "discord",
            "name": "Discord",
            "color": "#5865F2",
            "connected": "discord" in connected_map,
            "account": connected_map.get("discord"),
            "connect_url": reverse("discord_login") + "?process=connect",
        },
        {
            "id": "gitlab",
            "name": "GitLab",
            "color": "#FC6D26",
            "connected": "gitlab" in connected_map,
            "account": connected_map.get("gitlab"),
            "connect_url": reverse("gitlab_login") + "?process=connect",
        },
        {
            "id": "bitbucket",
            "name": "Bitbucket",
            "color": "#0052CC",
            "connected": "bitbucket" in connected_map or "bitbucket_oauth2" in connected_map,
            "account": connected_map.get("bitbucket") or connected_map.get("bitbucket_oauth2"),
            "connect_url": reverse("bitbucket_login") + "?process=connect",
        },
    ]

    return render(
        request,
        "accounts/profile.html",
        {
            "providers_info": providers_info,
            "connected_count": social_qs.count(),
            **_rank_ladder(request.user),
            **_certificates_page(request),
        },
    )


def _certificates_page(request: HttpRequest) -> dict:
    from apps.core.pagination import paginate_queryset

    certificates = request.user.certificates.select_related("course")
    page, _ = paginate_queryset(request, certificates, per_page=6, page_param="dyplomy")
    return {"certificates_page": page}


def _rank_ladder(user) -> dict:
    """Rangi z przedziałami punktów, bieżąca ranga i ile brakuje do następnej."""
    ladder, lower = [], None
    for upper, rank in CustomUser.RANK_THRESHOLDS:
        if lower is None:
            span = "poniżej 0 pkt"
        else:
            span = f"{lower}–{upper} pkt"
        ladder.append({"name": str(rank), "span": span, "current": user.rank_title == rank})
        lower = upper + 1
    ladder.append({"name": str(CustomUser.Rank.MASTER), "span": f"od {lower} pkt", "current": user.rank_title == CustomUser.Rank.MASTER})
    next_rank = next_points = None
    for upper, rank in CustomUser.RANK_THRESHOLDS:
        if user.reputation_points <= upper:
            index = [r for _, r in CustomUser.RANK_THRESHOLDS].index(rank)
            following = CustomUser.RANK_THRESHOLDS[index + 1][1] if index + 1 < len(CustomUser.RANK_THRESHOLDS) else CustomUser.Rank.MASTER
            next_rank, next_points = str(following), upper + 1 - user.reputation_points
            break
    return {"rank_ladder": ladder, "next_rank": next_rank, "next_rank_points": next_points}


@login_required
@require_POST
def disconnect_social_account_view(request: HttpRequest, account_id: int) -> HttpResponse:
    """Odłączenie konta społecznościowego z zachowaniem bezpieczeństwa logowania."""
    account = get_object_or_404(SocialAccount, pk=account_id, user=request.user)

    # Sprawdź, czy użytkownik ma inny sposób logowania (hasło lub inne powiązane konto)
    has_other_social = SocialAccount.objects.filter(user=request.user).exclude(pk=account.pk).exists()
    has_local_password = request.user.has_usable_password()

    if not has_other_social and not has_local_password:
        messages.error(
            request,
            "Nie możesz odłączyć tego konta, ponieważ jest to Twój jedyny sposób logowania. "
            "Najpierw połącz inne konto społecznościowe lub ustaw hasło do profilu.",
        )
        return redirect("accounts:profile")

    provider_name = account.get_provider().name
    account.delete()
    messages.success(request, f"Konto {provider_name} zostało odłączone od Twojego profilu.")
    return redirect("accounts:profile")


@login_required
def profile_edit_view(request: HttpRequest) -> HttpResponse:
    """Edycja danych profilu użytkownika."""
    if request.method == "POST":
        form = CustomUserProfileForm(request.POST, request.FILES, instance=request.user)
        if form.is_valid():
            form.save()
            messages.success(request, "Dane profilu zostały zaktualizowane.")
            return redirect("accounts:profile")
        messages.error(request, "Proszę poprawić błędy w formularzu.")
    else:
        form = CustomUserProfileForm(instance=request.user)
    return render(request, "accounts/profile_edit.html", {"form": form})


def login_view(request: HttpRequest) -> HttpResponse:
    """Logowanie użytkownika sesyjne z obsługą split-screen i zapamiętywaniem pól."""
    if request.user.is_authenticated:
        return redirect("core:home")

    next_url = _safe_redirect_target(request, request.GET.get("next") or request.POST.get("next"))
    pending = get_pending(request)

    if request.method == "POST":
        form = LoginForm(request.POST)
        if form.is_valid():
            login_val = form.cleaned_data["login"].strip()
            password = form.cleaned_data["password"]
            remember_me = form.cleaned_data.get("remember_me", True)

            remaining_lock = lockout.locked_for(login_val)
            if remaining_lock:
                form.add_error(
                    None,
                    "Za dużo nieudanych prób logowania na to konto. "
                    f"Spróbuj ponownie za {lockout.minutes_text(remaining_lock)}.",
                )
                messages.error(request, "Logowanie jest chwilowo zablokowane.")
                return render(request, "accounts/login.html", _login_context(form, next_url, pending))

            # Próba uwierzytelnienia po nazwie użytkownika
            user = authenticate(request, username=login_val, password=password)

            # Jeśli nie znaleziono po username, sprawdź czy podano e-mail
            if user is None and "@" in login_val:
                candidate = CustomUser.objects.filter(email__iexact=login_val).first()
                if candidate:
                    user = authenticate(request, username=candidate.username, password=password)

            if user is not None:
                if not user.is_active:
                    messages.error(request, "Twoje konto zostało zablokowane.")
                else:
                    lockout.reset(login_val)
                    login(request, user)
                    if not remember_me:
                        request.session.set_expiry(0)
                    else:
                        request.session.set_expiry(1209600)  # 2 tygodnie

                    # Jeśli w sesji gościa był wybrany motyw, przenieś go na profil
                    session_theme_id = request.session.pop(SESSION_THEME_KEY, None)
                    if session_theme_id and not user.selected_theme:
                        theme = ColorTheme.objects.filter(pk=session_theme_id).first()
                        if theme:
                            user.selected_theme = theme
                            user.save(update_fields=["selected_theme"])

                    messages.success(request, f"Witaj ponownie, {user.get_short_name() or user.username}!")
                    return redirect(next_url)
            else:
                left = lockout.register_failure(login_val)
                if left:
                    form.add_error(
                        None,
                        "Nieprawidłowa nazwa użytkownika / e-mail lub hasło. "
                        f"Pozostało prób przed czasową blokadą: {left}.",
                    )
                else:
                    form.add_error(
                        None,
                        f"Nieprawidłowe dane logowania po raz {lockout.MAX_FAILURES}. Logowanie na to konto jest "
                        f"zablokowane na {lockout.minutes_text(lockout.LOCK_SECONDS)}.",
                    )
                messages.error(request, "Niepoprawne dane logowania. Sprawdź login i hasło.")
        else:
            messages.error(request, "Proszę wypełnić wymagane pola formularza.")
    else:
        form = LoginForm(initial={"login": pending["email"]} if pending else None)
        if pending:
            # Adres jest już wpisany — fokus od razu w polu hasła
            form.fields["login"].widget.attrs.pop("autofocus", None)
            form.fields["password"].widget.attrs["autofocus"] = True

    return render(request, "accounts/login.html", _login_context(form, next_url, pending))


def _login_context(form, next_url: str, pending: dict | None) -> dict:
    return {"form": form, "next_url": next_url, "pending_social": pending, **_pending_owner_context(pending)}


def _pending_owner_context(pending: dict | None) -> dict:
    """Czy konto czekające na połączenie ma hasło, a jeśli nie — którymi portalami się logować."""
    if not pending:
        return {}
    owner = CustomUser.objects.filter(email__iexact=pending["email"]).first()
    if owner is None:
        return {}
    providers = sorted({account.get_provider().name for account in SocialAccount.objects.filter(user=owner)})
    return {"pending_has_password": owner.has_usable_password(), "pending_providers": providers}


@require_POST
def social_connect_cancel(request: HttpRequest) -> HttpResponse:
    """Rezygnacja z łączenia konta portalu czekającego w sesji."""
    if request.session.pop(PENDING_SOCIAL_KEY, None):
        messages.info(request, "Anulowano łączenie konta. Możesz zalogować się zwykłym kontem.")
    return redirect("accounts:login")


def check_availability(request: HttpRequest) -> JsonResponse:
    """Sprawdzenie na żywo w formularzu rejestracji: czy login / e-mail są wolne."""
    field = request.GET.get("field", "")
    value = request.GET.get("value", "").strip()
    if field not in {"email", "username"} or not value or len(value) > 254:
        return JsonResponse({"error": "Nieprawidłowe zapytanie."}, status=400)
    lookup = {f"{field}__iexact": value}
    taken = CustomUser.objects.filter(**lookup).exists()
    if not taken:
        message = "Ten adres jest wolny." if field == "email" else "Ta nazwa jest wolna."
    elif field == "email":
        message = "Konto z tym adresem już istnieje — zaloguj się na nie zamiast zakładać nowe."
    else:
        message = "Ta nazwa jest zajęta — wybierz inną."
    return JsonResponse({"available": not taken, "message": message})


def register_view(request: HttpRequest) -> HttpResponse:
    """Rejestracja nowego użytkownika z zapamiętywaniem pól przy błędzie."""
    if request.user.is_authenticated:
        return redirect("core:home")

    if request.method == "POST":
        form = CustomUserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user, backend="django.contrib.auth.backends.ModelBackend")
            messages.success(
                request,
                f"Konto zostało pomyślnie utworzone! Witaj na platformie, {user.username}.",
            )
            return redirect("core:home")
        messages.error(request, "Nie udało się utworzyć konta. Popraw błędy w formularzu.")
    else:
        form = CustomUserRegistrationForm()

    return render(request, "accounts/register.html", {"form": form})


@login_required
@require_POST
def api_session_token(request: HttpRequest) -> JsonResponse:
    """Krótkotrwały token JWT dla komponentów dynamicznych strony (np. koszyk) — z sesji + CSRF,
    żeby skrypt w przeglądarce nie potrzebował hasła."""
    token = AccessToken.for_user(request.user)
    return JsonResponse({"access": str(token), "expires_in": int(token.lifetime.total_seconds())})


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def api_me_view(request: HttpRequest) -> JsonResponse:
    """API endpoint zwracający profil zalogowanego użytkownika na podstawie tokenu JWT."""
    user: CustomUser = request.user
    return JsonResponse(
        {
            "id": user.pk,
            "username": user.username,
            "email": user.email,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "role": user.role,
            "avatar_url": user.avatar_url or (user.avatar.url if user.avatar else ""),
            "reputation_points": user.reputation_points,
            "rank_title": user.rank_title,
        }
    )


def _safe_redirect_target(request: HttpRequest, candidate: str | None) -> str:
    """Zwraca bezpieczny adres powrotu lub stronę wyboru motywu."""
    if candidate and url_has_allowed_host_and_scheme(
        candidate,
        allowed_hosts={request.get_host()},
        require_https=request.is_secure(),
    ):
        return candidate
    return reverse("accounts:select-theme")


def select_theme_view(request: HttpRequest) -> HttpResponse:
    """Publiczna strona wyboru motywu.

    Działa także dla niezalogowanych — wybór trafia wtedy do sesji,
    a po zalogowaniu można go zapisać na koncie (navbarem lub tutaj).
    """
    return render(
        request,
        "accounts/select_theme.html",
        {"next_url": _safe_redirect_target(request, request.GET.get("next"))},
    )


@require_POST
def set_theme_view(request: HttpRequest) -> HttpResponse:
    """Zapisuje wybrany motyw (konto lub sesja) i wraca na poprzednią stronę."""
    raw_theme = request.POST.get("theme", "").strip()
    theme: ColorTheme | None = None

    if raw_theme and raw_theme != "default":
        theme = (
            ColorTheme.objects.filter(pk=raw_theme).first()
            if raw_theme.isdigit()
            else None
        )
        if theme is None:
            messages.error(request, "Nie znaleziono wybranego motywu.")
            return redirect(_safe_redirect_target(request, request.POST.get("next")))

    if request.user.is_authenticated:
        request.user.selected_theme = theme
        request.user.save(update_fields=["selected_theme"])
        # Sesja nie powinna przykrywać wyboru zapisanego na koncie.
        request.session.pop(SESSION_THEME_KEY, None)
    elif theme is None:
        request.session.pop(SESSION_THEME_KEY, None)
    else:
        request.session[SESSION_THEME_KEY] = theme.pk

    if theme is None:
        messages.success(request, "Przywrócono domyślny motyw kolorystyczny.")
    else:
        messages.success(request, f"Wybrano motyw „{theme.name}”.")

    return redirect(_safe_redirect_target(request, request.POST.get("next")))


def logout_view(request: HttpRequest) -> HttpResponse:
    """Wylogowanie użytkownika."""
    logout(request)
    return redirect("core:home")


@login_required
def password_change_view(request: HttpRequest) -> HttpResponse:
    """Zmiana hasła (z obecnym hasłem). Konto tylko z portalu OAuth ustawia hasło po raz pierwszy."""
    has_password = request.user.has_usable_password()
    form_class = AccessiblePasswordChangeForm if has_password else AccessibleSetPasswordForm
    if request.method == "POST":
        form = form_class(request.user, request.POST)
        if form.is_valid():
            user = form.save()
            update_session_auth_hash(request, user)  # bez wylogowania po zmianie
            lockout.reset(user.username)
            messages.success(request, "Hasło zostało zmienione." if has_password else "Hasło zostało ustawione — możesz logować się też loginem i hasłem.")
            return redirect("accounts:profile")
        messages.error(request, "Popraw zaznaczone pola formularza.")
    else:
        form = form_class(request.user)
    return render(request, "accounts/password_change.html", {"form": form, "has_password": has_password})
