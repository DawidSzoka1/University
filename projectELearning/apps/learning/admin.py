from django.contrib import admin
from unfold.admin import ModelAdmin, StackedInline, TabularInline

from apps.learning.models import (
    Answer,
    Attempt,
    Certificate,
    Choice,
    CourseSection,
    Lesson,
    LearningStyle,
    LessonAttachment,
    LessonVariant,
    SurveyAnswer,
    SurveyOption,
    SurveyQuestion,
    SurveyResponse,
    LessonProgress,
    Question,
    Quiz,
    QuizSitting,
)


class ChoiceInline(TabularInline):
    model = Choice
    extra = 1


@admin.register(Quiz)
class QuizAdmin(ModelAdmin):
    list_display = ("title", "course", "after_lesson", "is_required", "pass_percent", "max_attempts")
    list_filter = ("is_required",)
    search_fields = ("title", "course__title")
    raw_id_fields = ("course", "after_lesson")


class AnswerInline(TabularInline):
    model = Answer
    extra = 0
    fields = ("question", "text_answer", "file", "points", "feedback")
    raw_id_fields = ("question",)


@admin.register(Attempt)
class AttemptAdmin(ModelAdmin):
    list_display = ("quiz", "user", "status", "score", "max_score", "passed", "submitted_at")
    list_filter = ("status", "passed")
    search_fields = ("quiz__title", "user__username")
    raw_id_fields = ("quiz", "user", "graded_by")
    inlines = [AnswerInline]


@admin.register(QuizSitting)
class QuizSittingAdmin(ModelAdmin):
    list_display = ("quiz", "user", "started_at")
    search_fields = ("quiz__title", "user__username")
    raw_id_fields = ("quiz", "user")


@admin.register(Certificate)
class CertificateAdmin(ModelAdmin):
    list_display = ("code", "full_name", "course_title", "issued_at")
    search_fields = ("code", "full_name", "course_title", "user__username")
    raw_id_fields = ("user", "course")


@admin.register(Question)
class QuestionAdmin(ModelAdmin):
    list_display = ("text", "quiz", "question_type", "order")
    list_filter = ("question_type",)
    inlines = [ChoiceInline]


@admin.register(Choice)
class ChoiceAdmin(ModelAdmin):
    list_display = ("text", "question", "is_correct", "order")
    list_filter = ("is_correct",)


class LessonInline(StackedInline):
    model = Lesson
    extra = 0
    fields = ("title", "order", "duration_minutes", "is_preview")
    show_change_link = True


@admin.register(CourseSection)
class CourseSectionAdmin(ModelAdmin):
    list_display = ("title", "course", "order")
    list_filter = ("course",)
    search_fields = ("title", "course__title")
    inlines = [LessonInline]


class LessonAttachmentInline(TabularInline):
    model = LessonAttachment
    extra = 0
    fields = ("file", "original_name", "size")
    readonly_fields = ("size",)


class LessonVariantInline(StackedInline):
    model = LessonVariant
    extra = 0
    fields = ("style", "video_url")
    show_change_link = True


@admin.register(Lesson)
class LessonAdmin(ModelAdmin):
    list_display = ("title", "section", "order", "duration_minutes", "is_preview", "same_for_all_styles")
    list_filter = ("is_preview", "same_for_all_styles", "section__course")
    search_fields = ("title", "section__title", "section__course__title")
    inlines = [LessonAttachmentInline, LessonVariantInline]


@admin.register(LearningStyle)
class LearningStyleAdmin(ModelAdmin):
    """Słownik stylów uczenia się — administrator dodaje kolejne style i ustala ich kolejność."""

    list_display = ("name", "slug", "order", "is_active", "learner_count", "variant_count")
    search_fields = ("name", "description")
    prepopulated_fields = {"slug": ("name",)}
    fields = ("name", "slug", "description", "author_hint", "order", "is_active")

    @admin.display(description="Uczniowie z tym stylem")
    def learner_count(self, obj):
        return obj.learners.count()

    @admin.display(description="Wersje lekcji")
    def variant_count(self, obj):
        return obj.lesson_variants.count()


@admin.register(LessonVariant)
class LessonVariantAdmin(ModelAdmin):
    list_display = ("lesson", "style", "updated_at")
    list_filter = ("style",)
    search_fields = ("lesson__title", "lesson__section__course__title")
    raw_id_fields = ("lesson",)


@admin.register(LessonProgress)
class LessonProgressAdmin(ModelAdmin):
    list_display = ("user", "lesson", "style", "completed_at")
    raw_id_fields = ("user", "lesson")


class SurveyOptionInline(TabularInline):
    model = SurveyOption
    extra = 1
    fields = ("text", "style", "order")


@admin.register(SurveyQuestion)
class SurveyQuestionAdmin(ModelAdmin):
    """Pytania ankiety stylu — administrator zmienia treść i dodaje odpowiedzi dla nowych stylów."""

    list_display = ("text", "order", "is_active")
    inlines = [SurveyOptionInline]


class SurveyAnswerInline(TabularInline):
    model = SurveyAnswer
    extra = 0
    fields = ("question", "option")
    readonly_fields = ("question", "option")
    can_delete = False


@admin.register(SurveyResponse)
class SurveyResponseAdmin(ModelAdmin):
    list_display = ("user", "top_style", "created_at")
    list_filter = ("top_style",)
    search_fields = ("user__username",)
    raw_id_fields = ("user",)
    inlines = [SurveyAnswerInline]
