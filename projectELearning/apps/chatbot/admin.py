from django.contrib import admin
from unfold.admin import ModelAdmin, TabularInline

from apps.chatbot.models import ChatMessage, ChatSession


class ChatMessageInline(TabularInline):
    model = ChatMessage
    extra = 0
    fields = ('sender', 'message_text', 'lesson', 'is_refusal', 'timestamp')
    readonly_fields = fields

    def has_add_permission(self, request, obj=None) -> bool:  # noqa: D401
        return False

    def has_delete_permission(self, request, obj=None) -> bool:  # noqa: D401
        return False


@admin.register(ChatSession)
class ChatSessionAdmin(ModelAdmin):
    list_display = ('student', 'course', 'message_count', 'updated_at')
    list_filter = ('updated_at',)
    list_select_related = ('student', 'course')
    search_fields = ('student__username', 'course__title')
    inlines = [ChatMessageInline]

    @admin.display(description='Wiadomości')
    def message_count(self, obj: ChatSession) -> int:  # noqa: D401
        return obj.messages.count()
