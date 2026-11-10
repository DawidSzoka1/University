"""
Main URL configuration for the E-Learning Platform.
"""

from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.templatetags.static import static as static_url
from django.urls import include, path
from django.views.generic import RedirectView

from allauth.socialaccount.providers.bitbucket_oauth2 import views as bitbucket_views

from apps.learning import views as learning_views

urlpatterns = [
    # Ikona strony dla przeglądarek i panelu admina, które pytają o /favicon.ico
    path("favicon.ico", RedirectView.as_view(url=static_url("img/favicon.svg"), permanent=True)),

    # Admin
    path("admin/", admin.site.urls),

    # Project apps
    path("", include("apps.core.urls", namespace="core")),
    path("accounts/", include("apps.accounts.urls", namespace="accounts")),
    path("catalog/", include("apps.catalog.urls", namespace="catalog")),
    path("gallery/", include("apps.gallery.urls", namespace="gallery")),
    path("orders/", include("apps.orders.urls", namespace="orders")),
    path("learning/", include("apps.learning.urls", namespace="learning")),
    path("chatbot/", include("apps.chatbot.urls", namespace="chatbot")),

    # Bitbucket clean URLs (strictly /bitbucket/ without _oauth2)
    path("accounts/social/bitbucket/login/", bitbucket_views.oauth_login, name="bitbucket_login"),
    path("accounts/social/bitbucket/login/callback/", bitbucket_views.oauth_callback, name="bitbucket_callback"),

    # Allauth (social login)
    path("accounts/social/", include("allauth.urls")),

    # REST Framework browsable API login
    path("api-auth/", include("rest_framework.urls", namespace="rest_framework")),

    # CKEditor 5: własny upload zdjęć (uprawnienia + walidacja Pillow) przed domyślnym
    path("ckeditor5/image_upload/", learning_views.editor_image_upload, name="ck_editor_5_upload_file"),
    path("ckeditor5/", include("django_ckeditor_5.urls")),
]

# Serve media files in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

    # Django Debug Toolbar
    import debug_toolbar

    urlpatterns = [
        path("__debug__/", include(debug_toolbar.urls)),
    ] + urlpatterns
