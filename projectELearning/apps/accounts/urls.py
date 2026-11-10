from django.urls import path
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

from apps.accounts import views

app_name = "accounts"

urlpatterns = [
    path("profile/", views.profile_view, name="profile"),
    path("profile/edit/", views.profile_edit_view, name="profile-edit"),
    path("haslo/", views.password_change_view, name="password_change"),
    path("social/disconnect/<int:account_id>/", views.disconnect_social_account_view, name="disconnect-social-account"),
    path("login/", views.login_view, name="login"),
    path("logout/", views.logout_view, name="logout"),
    path("polacz/anuluj/", views.social_connect_cancel, name="social_connect_cancel"),
    path("dostepnosc/", views.check_availability, name="check_availability"),
    path("register/", views.register_view, name="register"),
    path("select-theme/", views.select_theme_view, name="select-theme"),
    path("select-theme/set/", views.set_theme_view, name="set-theme"),
    # JWT Token API (Wymóg 4.0)
    path("api/token/", TokenObtainPairView.as_view(), name="token-obtain-pair"),
    path("api/token/refresh/", TokenRefreshView.as_view(), name="token-refresh"),
    path("api/token/sesja/", views.api_session_token, name="api_session_token"),
    path("api/me/", views.api_me_view, name="api-me"),
]
