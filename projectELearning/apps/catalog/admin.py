from __future__ import annotations

from django.contrib import admin
from mptt.admin import DraggableMPTTAdmin
from unfold.admin import ModelAdmin

from apps.catalog.models import (
    Category,
    Comment,
    CommentVote,
    Course,
    CourseSubmissionMessage,
    Enrollment,
)


@admin.register(Category)
class CategoryAdmin(DraggableMPTTAdmin):
    list_display = (
        "tree_actions",
        "indented_title",
        "slug",
        "is_active",
        "order",
    )
    list_display_links = ("indented_title",)
    list_filter = ("is_active",)
    search_fields = ("name", "description")
    prepopulated_fields = {"slug": ("name",)}
    filter_horizontal = ("moderators",)


@admin.register(Course)
class CourseAdmin(ModelAdmin):
    list_display = (
        "title",
        "category",
        "author",
        "price",
        "is_free",
        "students_count",
        "completed_count",
        "average_rating",
        "status",
        "is_published",
        "created_at",
    )
    list_filter = (
        "is_free",
        "status",
        "is_published",
        "difficulty_level",
        "category",
    )
    search_fields = (
        "title",
        "description",
        "author__username",
        "author__first_name",
        "author__last_name",
        "author__email",
    )
    prepopulated_fields = {"slug": ("title",)}
    raw_id_fields = ("author",)
    readonly_fields = (
        "students_count",
        "completed_count",
        "average_rating",
        "reviews_count",
    )
    fieldsets = (
        (
            "Informacje podstawowe",
            {
                "fields": (
                    "title",
                    "slug",
                    "category",
                    "author",
                    "difficulty_level",
                    "duration_hours",
                )
            },
        ),
        (
            "Statystyki i popularność",
            {
                "fields": (
                    "students_count",
                    "completed_count",
                    "average_rating",
                    "reviews_count",
                )
            },
        ),
        (
            "Cena i dostępność",
            {
                "fields": (
                    "is_free",
                    "price",
                )
            },
        ),
        (
            "Opis i multimedia",
            {
                "fields": (
                    "description",
                    "description_html",
                    "thumbnail",
                )
            },
        ),
        (
            "Moderacja i publikacja",
            {
                "fields": (
                    "status",
                    "is_published",
                    "rejection_reason",
                )
            },
        ),
    )


@admin.register(Enrollment)
class EnrollmentAdmin(ModelAdmin):
    list_display = (
        "user",
        "course",
        "is_completed",
        "progress_percent",
        "completed_at",
        "created_at",
    )
    list_filter = ("is_completed", "course", "created_at")
    search_fields = ("user__username", "user__email", "course__title")
    raw_id_fields = ("user", "course")


@admin.register(Comment)
class CommentAdmin(ModelAdmin):
    list_display = (
        "course",
        "author",
        "rating",
        "is_approved",
        "approved_by",
        "created_at",
    )
    list_filter = ("is_approved", "rating", "created_at")
    search_fields = ("content", "author__username", "course__title")
    raw_id_fields = ("course", "author", "approved_by")


@admin.register(CommentVote)
class CommentVoteAdmin(ModelAdmin):
    list_display = ("comment", "voter", "value", "created_at")
    list_filter = ("value", "created_at")
    raw_id_fields = ("comment", "voter")


@admin.register(CourseSubmissionMessage)
class CourseSubmissionMessageAdmin(ModelAdmin):
    list_display = ("course", "author", "created_at")
    search_fields = ("content", "course__title", "author__username")
    raw_id_fields = ("course", "author")
