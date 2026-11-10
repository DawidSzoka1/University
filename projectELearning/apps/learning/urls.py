from django.urls import path

from apps.learning import progress_views, quiz_views, style_views, views

app_name = "learning"

urlpatterns = [
    path("my-courses/", views.my_courses, name="my_courses"),
    # Style uczenia się
    path("styl-uczenia/", style_views.learning_style, name="learning_style"),
    path("styl-uczenia/ankieta/", style_views.style_survey, name="style_survey"),
    path("kurs/<slug:slug>/lekcja/<int:pk>/style/", style_views.lesson_variants, name="lesson_variants"),
    path("kurs/<slug:slug>/lekcja/<int:pk>/style/przelacz/", style_views.lesson_styles_toggle, name="lesson_styles_toggle"),
    path("kurs/<slug:slug>/lekcja/<int:pk>/style/<slug:style_slug>/", style_views.lesson_variant_edit, name="lesson_variant_edit"),
    path("kurs/<slug:slug>/lekcja/<int:pk>/style/<slug:style_slug>/usun/", style_views.lesson_variant_delete, name="lesson_variant_delete"),
    # Kreator programu (autor kursu, moderatorzy kategorii)
    path("kurs/<slug:slug>/program/", views.curriculum_edit, name="curriculum_edit"),
    path("kurs/<slug:slug>/uczniowie/", progress_views.course_students, name="course_students"),
    path("kurs/<slug:slug>/program/sekcja/<int:pk>/nazwa/", views.section_rename, name="section_rename"),
    path("kurs/<slug:slug>/program/sekcja/<int:pk>/usun/", views.section_delete, name="section_delete"),
    path("kurs/<slug:slug>/program/sekcja/<int:pk>/przesun/<str:direction>/", views.section_move, name="section_move"),
    path("kurs/<slug:slug>/program/sekcja/<int:section_pk>/nowa-lekcja/", views.lesson_create, name="lesson_create"),
    path("kurs/<slug:slug>/lekcja/<int:pk>/edytuj/", views.lesson_edit, name="lesson_edit"),
    path("kurs/<slug:slug>/lekcja/<int:pk>/usun/", views.lesson_delete, name="lesson_delete"),
    path("kurs/<slug:slug>/lekcja/<int:pk>/przesun/<str:direction>/", views.lesson_move, name="lesson_move"),
    # Nauka
    path("kurs/<slug:slug>/lekcja/<int:pk>/", views.lesson_detail, name="lesson_detail"),
    path("kurs/<slug:slug>/tryb/", views.learning_mode, name="learning_mode"),
    path("kurs/<slug:slug>/lekcja/<int:pk>/ukonczona/", views.lesson_toggle_complete, name="lesson_complete"),
    path("plik/<int:pk>/", views.attachment_download, name="attachment_download"),
    # Testy: kreator (autor / nauczyciel kategorii)
    path("kurs/<slug:slug>/test/nowy/", quiz_views.quiz_create, name="quiz_create"),
    path("kurs/<slug:slug>/test/<int:pk>/edytuj/", quiz_views.quiz_edit, name="quiz_edit"),
    path("kurs/<slug:slug>/test/<int:pk>/usun/", quiz_views.quiz_delete, name="quiz_delete"),
    path("kurs/<slug:slug>/test/<int:quiz_pk>/pytanie/nowe/", quiz_views.question_create, name="question_create"),
    path("kurs/<slug:slug>/pytanie/<int:pk>/edytuj/", quiz_views.question_edit, name="question_edit"),
    path("kurs/<slug:slug>/pytanie/<int:pk>/usun/", quiz_views.question_delete, name="question_delete"),
    path("kurs/<slug:slug>/pytanie/<int:pk>/przesun/<str:direction>/", quiz_views.question_move, name="question_move"),
    # Testy: uczeń
    path("kurs/<slug:slug>/test/<int:pk>/", quiz_views.quiz_detail, name="quiz_detail"),
    path("podejscie/<int:pk>/", quiz_views.attempt_detail, name="attempt_detail"),
    path("odpowiedz/<int:pk>/plik/", quiz_views.answer_file, name="answer_file"),
    path("pytanie/<int:pk>/plik/", quiz_views.question_file, name="question_file"),
    # Ocenianie
    path("do-oceny/", quiz_views.grading_queue, name="grading_queue"),
    path("podejscie/<int:pk>/ocen/", quiz_views.attempt_grade, name="attempt_grade"),
    # Dyplomy
    path("dyplom/<str:code>/", quiz_views.certificate_verify, name="certificate_verify"),
    path("dyplom/<str:code>/pdf/", quiz_views.certificate_download, name="certificate_download"),
]
