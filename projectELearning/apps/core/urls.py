from django.urls import path

from apps.core import stats, views

app_name = "core"

urlpatterns = [
    path("", views.home, name="home"),
    path("statystyki/", stats.platform_stats, name="stats"),
    path("powiadomienia/", views.notification_list, name="notification_list"),
    path("powiadomienia/<int:pk>/otworz/", views.notification_open, name="notification_open"),
    path("powiadomienia/przeczytane/", views.notification_mark_all_read, name="notification_mark_all_read"),
    # Celowo niedostępna strona demonstracyjna (wymóg projektu) i jej opis dla prowadzącego
    path("wcag-anty-wzorzec/", views.anti_wcag, name="anti_wcag"),
    path("wcag-anty-wzorzec/opis/", views.anti_wcag_guide, name="anti_wcag_guide"),
]
