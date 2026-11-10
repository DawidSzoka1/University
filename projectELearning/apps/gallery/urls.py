from django.urls import path

from . import views

app_name = "gallery"

urlpatterns = [
    # 3. Slajder dla użytkownika
    path("", views.slider_view, name="slider_view"),
    # 1. Zdjęcia w bazie
    path("upload/", views.upload_image, name="upload_image"),
    path("zdjecie/<int:pk>/edytuj/", views.item_edit, name="item_edit"),
    path("zdjecie/<int:pk>/usun/", views.item_delete, name="item_delete"),
    # 2. Konfigurator układów (administrator)
    path("configurator/", views.slider_configurator, name="slider_configurator"),
    path("configurator/presets/", views.manage_presets, name="manage_presets"),
    path("configurator/uklad/nowy/", views.preset_create, name="preset_create"),
    path("configurator/uklad/<int:pk>/ustawienia/", views.preset_update, name="preset_update"),
    path("configurator/uklad/<int:pk>/wyswietlaj/", views.preset_activate, name="preset_activate"),
    path("configurator/uklad/<int:pk>/usun/", views.preset_delete, name="preset_delete"),
    path("configurator/uklad/<int:pk>/dodaj/", views.preset_add_item, name="preset_add_item"),
    path("configurator/uklad/<int:pk>/kolejnosc/", views.slider_reorder, name="slider_reorder"),
    path("configurator/uklad/<int:pk>/karta/<int:item_pk>/usun/", views.preset_remove_item, name="preset_remove_item"),
    path("configurator/uklad/<int:pk>/karta/<int:item_pk>/przesun/<str:direction>/", views.preset_move_item, name="preset_move_item"),
]
