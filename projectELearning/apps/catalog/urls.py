from django.urls import path

from apps.catalog import category_views, comment_views, moderation_views, views

app_name = "catalog"

urlpatterns = [
    path("", views.course_list, name="course_list"),
    path("create/", views.course_create, name="course_create"),
    # Moderacja i komentarze — przed wzorcami ze slugiem kursu
    path("moderacja/", moderation_views.moderation_submissions, name="moderation_submissions"),
    path("moderacja/komentarze/", moderation_views.moderation_comments, name="moderation_comments"),
    path("moderacja/kategorie/", category_views.category_manage, name="category_manage"),
    path("moderacja/kategorie/nowa/", category_views.category_create, name="category_create"),
    path("moderacja/kategorie/<int:pk>/edytuj/", category_views.category_edit, name="category_edit"),
    path("moderacja/kategorie/<int:pk>/przesun/<str:direction>/", category_views.category_move, name="category_move"),
    path("moderacja/kategorie/<int:pk>/widocznosc/", category_views.category_toggle, name="category_toggle"),
    path("moderacja/kategorie/<int:pk>/usun/", category_views.category_delete, name="category_delete"),
    path("komentarz/<int:pk>/edytuj/", comment_views.comment_edit, name="comment_edit"),
    path("komentarz/<int:pk>/usun/", comment_views.comment_delete, name="comment_delete"),
    path("komentarz/<int:pk>/zatwierdz/", comment_views.comment_approve, name="comment_approve"),
    path("komentarz/<int:pk>/glos/", comment_views.comment_vote, name="comment_vote"),
    path("<slug:slug>/", views.course_detail, name="course_detail"),
    path("<slug:slug>/zapisz/", views.course_enroll, name="course_enroll"),
    path("<slug:slug>/edytuj/", views.course_edit, name="course_edit"),
    path("<slug:slug>/wyslij/", views.course_submit, name="course_submit"),
    path("<slug:slug>/komentarze/", comment_views.comment_create, name="comment_create"),
    path("<slug:slug>/zgloszenie/", moderation_views.submission_thread, name="submission_thread"),
    path("<slug:slug>/zgloszenie/decyzja/", moderation_views.submission_decide, name="submission_decide"),
]
