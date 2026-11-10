from django.contrib import admin
from django.contrib.auth.admin import UserAdmin

from apps.accounts.models import CustomUser


@admin.register(CustomUser)
class CustomUserAdmin(UserAdmin):
    """Konfiguracja panelu admina dla rozszerzonego modelu użytkownika."""

    list_display = (
        'username',
        'email',
        'role',
        'rank_title',
        'reputation_points',
        'is_staff',
    )
    list_filter = ('role', 'rank_title')
    search_fields = ('username', 'email', 'first_name', 'last_name')

    # Rozszerzenie domyślnych fieldsetów UserAdmin o dodatkowe pola
    fieldsets = UserAdmin.fieldsets + (
        (
            'Rola i personalizacja',
            {
                'fields': ('role', 'avatar', 'selected_theme'),
            },
        ),
        (
            'Adres',
            {
                'fields': ('street_address', 'postal_code', 'city', 'phone_number'),
                'classes': ('collapse',),
            },
        ),
        (
            'Reputacja',
            {
                'fields': ('reputation_points', 'rank_title'),
            },
        ),
    )

    add_fieldsets = UserAdmin.add_fieldsets + (
        (
            'Rola',
            {
                'fields': ('role',),
            },
        ),
    )
