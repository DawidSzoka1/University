"""
Django settings for the E-Learning Platform.

Reads configuration from environment variables via django-environ.
See .env for required variables.
"""

from datetime import timedelta
from pathlib import Path

import sys
import environ

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
env = environ.Env(
    DJANGO_DEBUG=(bool, False),
    DJANGO_ALLOWED_HOSTS=(list, ["localhost", "127.0.0.1"]),
    REDIS_URL=(str, "redis://localhost:6379/0"),
    EMAIL_BACKEND=(str, "django.core.mail.backends.console.EmailBackend"),
)
environ.Env.read_env(BASE_DIR / ".env")

# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
SECRET_KEY = env("DJANGO_SECRET_KEY")
DEBUG = env("DJANGO_DEBUG")
ALLOWED_HOSTS = env("DJANGO_ALLOWED_HOSTS")

# ---------------------------------------------------------------------------
# Applications
# ---------------------------------------------------------------------------
INSTALLED_APPS = [
    # Modern admin interface (must be before django.contrib.admin)
    "unfold",
    "unfold.contrib.filters",
    "unfold.contrib.forms",
    # Django defaults
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "django.contrib.sites",
    # Third-party
    "allauth",
    "allauth.account",
    "allauth.socialaccount",
    "allauth.socialaccount.providers.discord",
    "allauth.socialaccount.providers.gitlab",
    "allauth.socialaccount.providers.bitbucket_oauth2",
    "rest_framework",
    "rest_framework_simplejwt",
    "rest_framework_simplejwt.token_blacklist",
    "mptt",
    "corsheaders",
    "django_ckeditor_5",
    # Project apps
    "apps.core",
    "apps.accounts",
    "apps.catalog",
    "apps.gallery",
    "apps.orders",
    "apps.learning",
    "apps.chatbot",
]

if DEBUG:
    INSTALLED_APPS += ["debug_toolbar"]

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "corsheaders.middleware.CorsMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "allauth.account.middleware.AccountMiddleware",
]

if DEBUG:
    MIDDLEWARE.insert(0, "debug_toolbar.middleware.DebugToolbarMiddleware")

# ---------------------------------------------------------------------------
# URL / WSGI
# ---------------------------------------------------------------------------
ROOT_URLCONF = "config.urls"
WSGI_APPLICATION = "config.wsgi.application"

# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------
TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
                "apps.core.context_processors.theme_colors",
                "apps.core.context_processors.notifications_count",
                "apps.chatbot.context_processors.assistant",
                "apps.orders.context_processors.cart_count",
            ],
        },
    },
]

# ---------------------------------------------------------------------------
# Database – PostgreSQL
# ---------------------------------------------------------------------------
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": env("POSTGRES_DB"),
        "USER": env("POSTGRES_USER"),
        "PASSWORD": env("POSTGRES_PASSWORD"),
        "HOST": env("POSTGRES_HOST"),
        "PORT": env.int("POSTGRES_PORT", default=5432),
    },
}

# ---------------------------------------------------------------------------
# Cache – Redis
# ---------------------------------------------------------------------------
CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.redis.RedisCache",
        "LOCATION": env("REDIS_URL"),
    },
}

# Testy: pamięć podręczna w procesie — licznik blokady logowania nie przechodzi między uruchomieniami
if len(sys.argv) > 1 and sys.argv[1] == "test":
    CACHES = {"default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"}}

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
AUTH_USER_MODEL = "accounts.CustomUser"

AUTHENTICATION_BACKENDS = [
    "django.contrib.auth.backends.ModelBackend",
    "allauth.account.auth_backends.AuthenticationBackend",
]

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LOGIN_REDIRECT_URL = "/"
LOGOUT_REDIRECT_URL = "/"

# ---------------------------------------------------------------------------
# Sites framework (required by allauth)
# ---------------------------------------------------------------------------
SITE_ID = 1

# ---------------------------------------------------------------------------
# django-allauth (API zgodne z allauth >= 65)
# ---------------------------------------------------------------------------
ACCOUNT_LOGIN_METHODS = {"email", "username"}
ACCOUNT_SIGNUP_FIELDS = ["email*", "username*", "password1*", "password2*"]
ACCOUNT_EMAIL_VERIFICATION = "none"
SOCIALACCOUNT_AUTO_SIGNUP = True
SOCIALACCOUNT_STORE_TOKENS = True
# Zajęty e-mail z portalu obsługuje apps.accounts.adapter: logowanie na istniejące konto, potem
# automatyczne połączenie. Wbudowane logowanie po e-mailu allauth jest wyłączone, bo czyści hasło
# kont z niezweryfikowanym adresem (u nas rejestracja nie weryfikuje e-maila).
SOCIALACCOUNT_ADAPTER = "apps.accounts.adapter.SocialAccountAdapter"
SOCIALACCOUNT_EMAIL_AUTHENTICATION = False
SOCIALACCOUNT_EMAIL_AUTHENTICATION_AUTO_CONNECT = False

SOCIALACCOUNT_PROVIDERS = {
    "discord": {
        "APP": {
            "client_id": env("DISCORD_CLIENT_ID", default="sample_discord_client_id"),
            "secret": env("DISCORD_CLIENT_SECRET", default="sample_discord_secret"),
            "key": "",
        },
        "SCOPE": ["identify", "email"],
    },
    "gitlab": {
        "APP": {
            "client_id": env("GITLAB_CLIENT_ID", default="sample_gitlab_client_id"),
            "secret": env("GITLAB_CLIENT_SECRET", default="sample_gitlab_secret"),
            "key": "",
        },
        "SCOPE": ["read_user"],
    },
    "bitbucket_oauth2": {
        "APP": {
            "client_id": env("BITBUCKET_CLIENT_ID", default=env("BITBUCKET_KEY", default="sample_bitbucket_client_id")),
            "secret": env("BITBUCKET_CLIENT_SECRET", default=env("BITBUCKET_SECRET", default="sample_bitbucket_secret")),
            "key": "",
        },
        "SCOPE": ["account", "email"],
    },
}

# ---------------------------------------------------------------------------
# Django REST Framework
# ---------------------------------------------------------------------------
REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework_simplejwt.authentication.JWTAuthentication",
        "rest_framework.authentication.SessionAuthentication",
    ],
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.IsAuthenticated",
    ],
    "DEFAULT_PAGINATION_CLASS": "rest_framework.pagination.PageNumberPagination",
    "PAGE_SIZE": 20,
}

# ---------------------------------------------------------------------------
# Asystent AI (Bielik przez Ollama)
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = env("OLLAMA_BASE_URL", default="http://localhost:11434")
OLLAMA_MODEL = env("OLLAMA_MODEL", default="hf.co/speakleash/Bielik-Minitron-7B-v3.0-Instruct-GGUF:Q4_K_M")
OLLAMA_TIMEOUT = env.int("OLLAMA_TIMEOUT", default=180)  # sekundy na całą odpowiedź
OLLAMA_MODEL_LABEL = env("OLLAMA_MODEL_LABEL", default="")  # nazwa modelu dla ucznia; puste = z OLLAMA_MODEL
CHATBOT_HISTORY_LIMIT = 12  # ile ostatnich wiadomości trafia do modelu jako kontekst rozmowy
CHATBOT_QUIZ_LOCK_MINUTES = env.int("CHATBOT_QUIZ_LOCK_MINUTES", default=60)  # blokada po otwarciu testu

# ---------------------------------------------------------------------------
# Simple JWT
# ---------------------------------------------------------------------------
SIMPLE_JWT = {
    "ACCESS_TOKEN_LIFETIME": timedelta(minutes=60),
    "REFRESH_TOKEN_LIFETIME": timedelta(days=7),
    "ROTATE_REFRESH_TOKENS": True,
    "BLACKLIST_AFTER_ROTATION": True,
    "AUTH_HEADER_TYPES": ("Bearer",),
}

# ---------------------------------------------------------------------------
# CKEditor 5
# ---------------------------------------------------------------------------
CKEDITOR_5_CONFIGS = {
    "default": {
        "toolbar": [
            "heading",
            "|",
            "bold",
            "italic",
            "underline",
            "strikethrough",
            "|",
            "bulletedList",
            "numberedList",
            "blockQuote",
            "|",
            "link",
            "imageUpload",
            "mediaEmbed",
            "|",
            "undo",
            "redo",
        ],
    },
}
# Edytor kursów dla nauczycieli i uczniów: bez obrazków i mediów (upload CKEditora jest
# tylko dla obsługi, a bleach i tak usuwa osadzone media). Miniatura ma osobne pole.
CKEDITOR_5_CONFIGS["course"] = {
    "language": "pl",
    "toolbar": [
        "heading",
        "|",
        "bold",
        "italic",
        "underline",
        "strikethrough",
        "|",
        "bulletedList",
        "numberedList",
        "blockQuote",
        "|",
        "link",
        "|",
        "undo",
        "redo",
    ],
    "heading": {
        "options": [
            {"model": "paragraph", "title": "Akapit", "class": "ck-heading_paragraph"},
            {"model": "heading2", "view": "h2", "title": "Nagłówek", "class": "ck-heading_heading2"},
            {"model": "heading3", "view": "h3", "title": "Podtytuł", "class": "ck-heading_heading3"},
        ]
    },
}
# Edytor lekcji: zdjęcia z tekstem alternatywnym i podpisem, bloki kodu, tabele.
CKEDITOR_5_CONFIGS["lesson"] = {
    "language": "pl",
    "toolbar": [
        "heading", "|",
        "bold", "italic", "underline", "strikethrough", "code", "|",
        "bulletedList", "numberedList", "blockQuote", "|",
        "codeBlock", "insertTable", "imageUpload", "link", "|",
        "undo", "redo",
    ],
    "heading": CKEDITOR_5_CONFIGS["course"]["heading"],
    "image": {
        "toolbar": ["imageTextAlternative", "toggleImageCaption", "|", "imageStyle:inline", "imageStyle:block"],
    },
    "table": {"contentToolbar": ["tableColumn", "tableRow", "toggleTableCaption"]},
    "codeBlock": {
        "languages": [
            {"language": "python", "label": "Python"},
            {"language": "plaintext", "label": "Tekst"},
            {"language": "bash", "label": "Terminal"},
            {"language": "html", "label": "HTML"},
            {"language": "css", "label": "CSS"},
            {"language": "javascript", "label": "JavaScript"},
            {"language": "sql", "label": "SQL"},
        ]
    },
}
# Upload zdjęć z edytora: widok apps.learning.views.editor_image_upload sprawdza rolę
# (nauczyciel, admin, autor kursu) i waliduje obraz; tu tylko wymóg zalogowania.
CKEDITOR_5_FILE_UPLOAD_PERMISSION = "authenticated"
CKEDITOR_5_MAX_FILE_SIZE = 5  # MB
CKEDITOR_5_FILE_STORAGE = "django.core.files.storage.FileSystemStorage"

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
CORS_ALLOW_ALL_ORIGINS = DEBUG  # Restrict in production

# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------
EMAIL_BACKEND = env("EMAIL_BACKEND")
DEFAULT_FROM_EMAIL = env("DEFAULT_FROM_EMAIL", default="E-Learning <sklep@elearning.local>")
EMAIL_HOST = env("EMAIL_HOST", default="localhost")
EMAIL_PORT = env.int("EMAIL_PORT", default=25)
EMAIL_HOST_USER = env("EMAIL_HOST_USER", default="")
EMAIL_HOST_PASSWORD = env("EMAIL_HOST_PASSWORD", default="")
EMAIL_USE_TLS = env.bool("EMAIL_USE_TLS", default=False)

# Adres platformy w linkach z e-maili i powrotach z bramki płatności (bez ukośnika na końcu)
SITE_URL = env("SITE_URL", default="http://localhost:8000")

# ---------------------------------------------------------------------------
# Płatności — Stripe w trybie testowym (klucze sk_test_… / whsec_…)
# ---------------------------------------------------------------------------
STRIPE_SECRET_KEY = env("STRIPE_SECRET_KEY", default="")
STRIPE_PUBLISHABLE_KEY = env("STRIPE_PUBLISHABLE_KEY", default="")
STRIPE_WEBHOOK_SECRET = env("STRIPE_WEBHOOK_SECRET", default="")
# Gdy powyższe jest puste: plik zapisywany przez usługę stripe-cli z docker-compose
STRIPE_WEBHOOK_SECRET_FILE = BASE_DIR / ".stripe" / "webhook_secret"
STRIPE_CURRENCY = "pln"

# ---------------------------------------------------------------------------
# Internationalisation
# ---------------------------------------------------------------------------
LANGUAGE_CODE = "pl"
TIME_ZONE = "Europe/Warsaw"
USE_I18N = True
USE_TZ = True

# ---------------------------------------------------------------------------
# Static & Media files
# ---------------------------------------------------------------------------
STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STATICFILES_DIRS = [BASE_DIR / "static"]

MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"

# ---------------------------------------------------------------------------
# Debug Toolbar
# ---------------------------------------------------------------------------
INTERNAL_IPS = ["127.0.0.1"]

# ---------------------------------------------------------------------------
# Default primary key field type
# ---------------------------------------------------------------------------
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# ---------------------------------------------------------------------------
# Django Unfold Configuration
# ---------------------------------------------------------------------------
from django.templatetags.static import static
from django.urls import reverse_lazy

UNFOLD = {
    "SITE_TITLE": "E-Learning",
    "SITE_HEADER": "E-Learning Panel Zarządzania",
    "SITE_URL": "/",
    "SHOW_HISTORY": True,
    "SHOW_VIEW_ON_SITE": True,
    # Kolory, tryb jasny/ciemny i style panelu wynikają z aktywnego motywu platformy
    "COLORS": "apps.core.colors.get_unfold_theme_colors",
    "THEME": "apps.core.colors.get_unfold_theme_mode",
    "BORDER_RADIUS": "2px",
    "STYLES": [
        lambda request: static("css/fonts.css"),
        lambda request: static("css/admin.css"),
    ],
    "SCRIPTS": [
        lambda request: static("js/admin.js"),
    ],
    "SIDEBAR": {
        "show_search": True,
        "show_all_applications": True,
        "navigation": [
            {
                "title": "Nawigacja",
                "separator": True,
                "items": [
                    {
                        "title": "Strona główna platformy",
                        "icon": "home",
                        "link": lambda request: "/",
                    },
                ],
            },
            {
                "title": "Katalog & Nauka",
                "separator": True,
                "items": [
                    {
                        "title": "Kategorie",
                        "icon": "folder",
                        "link": lambda request: reverse_lazy("admin:catalog_category_changelist"),
                    },
                    {
                        "title": "Kursy",
                        "icon": "school",
                        "link": lambda request: reverse_lazy("admin:catalog_course_changelist"),
                    },
                    {
                        "title": "Zapisy na kursy",
                        "icon": "how_to_reg",
                        "link": lambda request: reverse_lazy("admin:catalog_enrollment_changelist"),
                    },
                    {
                        "title": "Komentarze i recenzje",
                        "icon": "rate_review",
                        "link": lambda request: reverse_lazy("admin:catalog_comment_changelist"),
                    },
                    {
                        "title": "Style uczenia się",
                        "icon": "psychology",
                        "link": lambda request: reverse_lazy("admin:learning_learningstyle_changelist"),
                    },
                ],
            },
            {
                "title": "Użytkownicy i System",
                "separator": True,
                "items": [
                    {
                        "title": "Użytkownicy",
                        "icon": "person",
                        "link": lambda request: reverse_lazy("apps_accounts_customuser_changelist" if False else "admin:accounts_customuser_changelist"),
                    },
                    {
                        "title": "Motywy kolorów",
                        "icon": "palette",
                        "link": lambda request: reverse_lazy("admin:core_colortheme_changelist"),
                    },
                ],
            },
        ],
    },
}

