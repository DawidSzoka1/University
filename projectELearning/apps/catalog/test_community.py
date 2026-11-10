"""Komentarze (wątki, edycja, usuwanie, głosy, akceptacja) oraz moderacja zgłoszeń kursów."""

from decimal import Decimal

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from apps.catalog.comments import MAX_DEPTH, build_comment_tree
from apps.catalog.models import Category, Comment, CommentVote, Course, CourseSubmissionMessage, Enrollment
from apps.core.models import Notification

User = get_user_model()


class CommunityFixture(TestCase):
    """Kategoria z nauczycielem, kurs, dwóch uczniów (jeden zapisany) i nauczyciel innej kategorii."""

    def setUp(self):
        self.teacher = User.objects.create_user(username="nauczyciel", password="Haslo12345!", role="moderator")
        self.other_teacher = User.objects.create_user(username="obcy_nauczyciel", password="Haslo12345!", role="moderator")
        self.student = User.objects.create_user(username="uczen", password="Haslo12345!", first_name="Ala")
        self.other = User.objects.create_user(username="inny", password="Haslo12345!")
        self.admin = User.objects.create_user(username="szef", password="Haslo12345!", role="admin")

        self.category = Category.objects.create(name="Python")
        self.category.moderators.add(self.teacher)
        self.other_category = Category.objects.create(name="Grafika")
        self.other_category.moderators.add(self.other_teacher)

        self.course = Course.objects.create(title="Python od zera", category=self.category, author=self.teacher)
        Enrollment.objects.create(user=self.student, course=self.course)

    def comment(self, author, content="Treść", *, parent=None, approved=True, rating=None, course=None):
        return Comment.objects.create(
            course=course or self.course, author=author, content=content, parent=parent,
            is_approved=approved, rating=rating,
        )

    def detail_url(self, course=None):
        return reverse("catalog:course_detail", kwargs={"slug": (course or self.course).slug})


class CommentTreeTests(CommunityFixture):
    def test_deep_replies_are_flattened_with_reply_to_label(self):
        root = self.comment(self.student, "Poziom 0")
        parent = root
        chain = []
        for level in range(1, MAX_DEPTH + 3):
            parent = self.comment(self.other if level % 2 else self.student, f"Poziom {level}", parent=parent)
            chain.append(parent)

        tree = build_comment_tree(self.course, self.other)
        self.assertEqual(len(tree), 1)
        node = tree[0]
        for _ in range(MAX_DEPTH - 1):
            self.assertEqual(len(node.children), 1)
            node = node.children[0]
        # Ostatni wcięty komentarz zbiera wszystkie głębsze odpowiedzi płasko
        self.assertEqual(node.depth, MAX_DEPTH - 1)
        flat = node.children
        self.assertEqual([n.comment.content for n in flat], [f"Poziom {i}" for i in range(MAX_DEPTH, MAX_DEPTH + 3)])
        self.assertTrue(all(n.depth == MAX_DEPTH for n in flat))
        self.assertEqual(flat[0].reply_to, "")  # bezpośrednia odpowiedź — bez dopisku
        self.assertEqual(flat[1].reply_to, flat[0].comment.author.username)

    def test_pending_comment_visible_only_to_author_and_moderator(self):
        self.comment(self.student, "Oczekujący", approved=False)
        self.assertEqual(len(build_comment_tree(self.course, self.student)), 1)
        self.assertEqual(len(build_comment_tree(self.course, self.teacher)), 1)
        self.assertEqual(len(build_comment_tree(self.course, self.other)), 0)
        self.assertEqual(len(build_comment_tree(self.course, self.other_teacher)), 0)

    def test_removed_comment_with_replies_stays_as_placeholder(self):
        root = self.comment(self.student, "Usunięty")
        self.comment(self.other, "Odpowiedź", parent=root)
        root.is_removed = True
        root.save()
        response = self.client.get(self.detail_url())
        self.assertContains(response, "Komentarz usunięty.")
        self.assertContains(response, "Odpowiedź")
        self.assertNotContains(response, "Usunięty</p>")

    def test_replies_do_not_change_course_rating(self):
        root = self.comment(self.student, "Recenzja", rating=4)
        self.comment(self.other, "Odpowiedź", parent=root, rating=1)
        self.course.refresh_from_db()
        self.assertEqual(self.course.reviews_count, 1)
        self.assertEqual(self.course.average_rating, Decimal("4.00"))

    def test_top_level_threads_are_paginated(self):
        for i in range(12):
            self.comment(self.other, f"Wątek {i:02d}")
        first = self.client.get(self.detail_url())
        self.assertContains(first, "Wątek 11")
        self.assertNotContains(first, "Wątek 01")
        second = self.client.get(self.detail_url() + "?page=2")
        self.assertContains(second, "Wątek 01")


class CommentCreateTests(CommunityFixture):
    def create_url(self):
        return reverse("catalog:comment_create", kwargs={"slug": self.course.slug})

    def test_requires_login_and_post(self):
        response = self.client.post(self.create_url(), {"content": "Hej"})
        self.assertEqual(response.status_code, 302)
        self.assertIn(reverse("accounts:login"), response.url)
        self.client.force_login(self.student)
        self.assertEqual(self.client.get(self.create_url()).status_code, 405)

    def test_student_comment_waits_for_approval_and_notifies_teacher(self):
        self.client.force_login(self.student)
        response = self.client.post(self.create_url(), {"content": "Świetny kurs", "rating": "5"})
        comment = Comment.objects.get(content="Świetny kurs")
        self.assertRedirects(response, self.detail_url() + f"#comment-{comment.pk}", fetch_redirect_response=False)
        self.assertFalse(comment.is_approved)
        self.assertEqual(comment.rating, 5)
        self.assertTrue(Notification.objects.filter(user=self.teacher, kind="comment_pending").exists())
        self.client.force_login(self.other)
        self.assertNotContains(self.client.get(self.detail_url()), "Świetny kurs")

    def test_teacher_of_category_publishes_immediately(self):
        root = self.comment(self.student, "Pytanie")
        self.client.force_login(self.teacher)
        self.client.post(self.create_url(), {"content": "Odpowiedź nauczyciela", "parent": root.pk})
        reply = Comment.objects.get(content="Odpowiedź nauczyciela")
        self.assertTrue(reply.is_approved)
        self.assertEqual(reply.parent, root)
        self.assertTrue(Notification.objects.filter(user=self.student, kind="comment_reply").exists())

    def test_rating_only_for_enrolled_and_never_in_reply(self):
        self.client.force_login(self.other)  # niezapisany
        self.client.post(self.create_url(), {"content": "Bez zapisu", "rating": "1"})
        self.assertIsNone(Comment.objects.get(content="Bez zapisu").rating)

        root = self.comment(self.other, "Wątek")
        self.client.force_login(self.student)
        self.client.post(self.create_url(), {"content": "Odpowiedź z oceną", "rating": "2", "parent": root.pk})
        self.assertIsNone(Comment.objects.get(content="Odpowiedź z oceną").rating)

    def test_second_rating_from_same_student_is_ignored(self):
        self.comment(self.student, "Pierwsza recenzja", rating=5)
        self.client.force_login(self.student)
        self.client.post(self.create_url(), {"content": "Drugi komentarz", "rating": "1"})
        self.assertIsNone(Comment.objects.get(content="Drugi komentarz").rating)

    def test_cannot_reply_to_pending_or_foreign_course_comment(self):
        pending = self.comment(self.other, "Oczekuje", approved=False)
        foreign_course = Course.objects.create(title="Inny kurs", category=self.other_category)
        foreign = self.comment(self.other, "Gdzie indziej", course=foreign_course)
        self.client.force_login(self.student)
        for parent in (pending, foreign):
            response = self.client.post(self.create_url(), {"content": "Odpowiedź", "parent": parent.pk})
            self.assertEqual(response.status_code, 404)

    def test_invalid_comment_keeps_text_and_describes_error(self):
        self.client.force_login(self.student)
        response = self.client.post(self.create_url(), {"content": " x "})
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "catalog/comment_form.html")
        self.assertContains(response, "Napisz co najmniej 2 znaki.")
        self.assertContains(response, 'aria-invalid="true"')
        self.assertContains(response, " x </textarea>")

    def test_unpublished_course_cannot_be_commented(self):
        self.course.status = Course.Status.PENDING
        self.course.is_published = False
        self.course.save()
        self.client.force_login(self.student)
        self.assertEqual(self.client.post(self.create_url(), {"content": "Hej"}).status_code, 404)


class CommentEditDeleteTests(CommunityFixture):
    def test_only_author_can_edit(self):
        comment = self.comment(self.student, "Moje")
        url = reverse("catalog:comment_edit", kwargs={"pk": comment.pk})
        for user in (self.other, self.teacher):
            self.client.force_login(user)
            self.assertEqual(self.client.get(url).status_code, 404)
        self.client.force_login(self.student)
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '<label for="id_content"')

    def test_edited_approved_comment_returns_to_review(self):
        comment = self.comment(self.student, "Przed", rating=3)
        self.client.force_login(self.student)
        self.client.post(reverse("catalog:comment_edit", kwargs={"pk": comment.pk}), {"content": "Po zmianie", "rating": "4"})
        comment.refresh_from_db()
        self.assertEqual(comment.content, "Po zmianie")
        self.assertEqual(comment.rating, 4)
        self.assertIsNotNone(comment.edited_at)
        self.assertFalse(comment.is_approved)
        self.assertTrue(Notification.objects.filter(user=self.teacher, kind="comment_pending").exists())

    def test_teacher_edit_of_own_comment_stays_public(self):
        comment = self.comment(self.teacher, "Nauczyciel")
        self.client.force_login(self.teacher)
        self.client.post(reverse("catalog:comment_edit", kwargs={"pk": comment.pk}), {"content": "Poprawione"})
        comment.refresh_from_db()
        self.assertTrue(comment.is_approved)

    def test_delete_without_replies_removes_row(self):
        comment = self.comment(self.student, "Do usunięcia")
        self.client.force_login(self.student)
        url = reverse("catalog:comment_delete", kwargs={"pk": comment.pk})
        self.assertEqual(self.client.get(url).status_code, 405)
        self.client.post(url)
        self.assertFalse(Comment.objects.filter(pk=comment.pk).exists())

    def test_delete_with_replies_keeps_marker_and_drops_votes(self):
        comment = self.comment(self.student, "Z odpowiedziami", rating=5)
        self.comment(self.other, "Odpowiedź", parent=comment)
        CommentVote.objects.create(comment=comment, voter=self.other, value=1)
        self.student.refresh_from_db()
        self.assertEqual(self.student.reputation_points, 1)

        self.client.force_login(self.student)
        self.client.post(reverse("catalog:comment_delete", kwargs={"pk": comment.pk}))
        comment.refresh_from_db()
        self.student.refresh_from_db()
        self.course.refresh_from_db()
        self.assertTrue(comment.is_removed)
        self.assertIsNone(comment.rating)
        self.assertEqual(self.student.reputation_points, 0)
        self.assertEqual(self.course.reviews_count, 0)

    def test_moderator_can_delete_others_but_student_cannot(self):
        comment = self.comment(self.student, "Spam")
        url = reverse("catalog:comment_delete", kwargs={"pk": comment.pk})
        self.client.force_login(self.other)
        self.assertEqual(self.client.post(url).status_code, 404)
        self.client.force_login(self.other_teacher)
        self.assertEqual(self.client.post(url).status_code, 404)
        self.client.force_login(self.teacher)
        self.client.post(url)
        self.assertFalse(Comment.objects.filter(pk=comment.pk).exists())


class CommentApproveTests(CommunityFixture):
    def test_teacher_approves_with_one_click_and_author_is_notified(self):
        comment = self.comment(self.student, "Czeka", approved=False)
        url = reverse("catalog:comment_approve", kwargs={"pk": comment.pk})
        self.client.force_login(self.teacher)
        self.assertEqual(self.client.get(url).status_code, 405)
        self.client.post(url)
        comment.refresh_from_db()
        self.assertTrue(comment.is_approved)
        self.assertEqual(comment.approved_by, self.teacher)
        self.assertTrue(Notification.objects.filter(user=self.student, kind="comment_approved").exists())

    def test_approve_button_shown_under_course_for_moderator_only(self):
        comment = self.comment(self.student, "Czeka", approved=False)
        url = reverse("catalog:comment_approve", kwargs={"pk": comment.pk})
        self.client.force_login(self.teacher)
        self.assertContains(self.client.get(self.detail_url()), url)
        self.client.force_login(self.student)
        response = self.client.get(self.detail_url())
        self.assertNotContains(response, url)
        self.assertContains(response, "Czeka na akceptację")

    def test_students_and_other_category_teachers_cannot_approve(self):
        comment = self.comment(self.student, "Czeka", approved=False)
        url = reverse("catalog:comment_approve", kwargs={"pk": comment.pk})
        for user in (self.student, self.other, self.other_teacher):
            self.client.force_login(user)
            self.assertEqual(self.client.post(url).status_code, 404)
        comment.refresh_from_db()
        self.assertFalse(comment.is_approved)

    def test_approve_redirects_back_to_queue_but_not_outside(self):
        comment = self.comment(self.student, "Czeka", approved=False)
        url = reverse("catalog:comment_approve", kwargs={"pk": comment.pk})
        self.client.force_login(self.teacher)
        queue = reverse("catalog:moderation_comments")
        self.assertRedirects(self.client.post(url, {"next": queue}), queue)
        comment.is_approved = False
        comment.save()
        response = self.client.post(url, {"next": "https://zly.example.com/"})
        self.assertFalse(response.url.startswith("https://"))


class CommentVoteTests(CommunityFixture):
    def setUp(self):
        super().setUp()
        self.target = self.comment(self.student, "Do oceny")
        self.url = reverse("catalog:comment_vote", kwargs={"pk": self.target.pk})

    def vote(self, value, **extra):
        return self.client.post(self.url, {"value": value}, **extra)

    def test_like_toggle_and_switch_update_reputation(self):
        self.client.force_login(self.other)
        self.vote("1")
        self.student.refresh_from_db()
        self.assertEqual(self.student.reputation_points, 1)
        self.vote("1")  # drugi raz to samo = cofnięcie
        self.assertFalse(CommentVote.objects.filter(comment=self.target).exists())
        self.vote("1")
        self.vote("-1")  # zmiana na „nie lubię”
        self.assertEqual(CommentVote.objects.get(comment=self.target).value, -1)
        self.student.refresh_from_db()
        self.assertEqual(self.student.reputation_points, -1)

    def test_json_response_for_fetch(self):
        self.client.force_login(self.other)
        response = self.vote("-1", HTTP_ACCEPT="application/json")
        self.assertEqual(response.json(), {"vote": -1, "likes": 0, "dislikes": 1})

    def test_cannot_vote_on_own_or_pending_comment(self):
        self.client.force_login(self.student)
        self.assertEqual(self.vote("1", HTTP_ACCEPT="application/json").status_code, 403)
        pending = self.comment(self.other, "Oczekuje", approved=False)
        self.client.force_login(self.teacher)
        response = self.client.post(
            reverse("catalog:comment_vote", kwargs={"pk": pending.pk}), {"value": "1"}, HTTP_ACCEPT="application/json"
        )
        self.assertEqual(response.status_code, 403)
        self.assertFalse(CommentVote.objects.exists())

    def test_invalid_value_and_anonymous(self):
        self.assertEqual(self.vote("1").status_code, 302)  # logowanie
        self.client.force_login(self.other)
        self.assertEqual(self.vote("5", HTTP_ACCEPT="application/json").status_code, 400)

    def test_buttons_expose_pressed_state(self):
        CommentVote.objects.create(comment=self.target, voter=self.other, value=1)
        self.client.force_login(self.other)
        self.assertContains(self.client.get(self.detail_url()), 'value="1" class="vote__btn" aria-pressed="true"')


class ModerationQueueTests(CommunityFixture):
    def test_queue_requires_moderator_role(self):
        for name in ("catalog:moderation_comments", "catalog:moderation_submissions"):
            self.client.force_login(self.student)
            self.assertEqual(self.client.get(reverse(name)).status_code, 404)
            self.client.force_login(self.teacher)
            self.assertEqual(self.client.get(reverse(name)).status_code, 200)

    def test_comment_queue_lists_only_own_categories_and_paginates(self):
        foreign_course = Course.objects.create(title="Grafika 101", category=self.other_category)
        self.comment(self.student, "Obcy komentarz", approved=False, course=foreign_course)
        for i in range(12):
            self.comment(self.student, f"Oczekujący {i:02d}", approved=False)
        self.client.force_login(self.teacher)
        response = self.client.get(reverse("catalog:moderation_comments"))
        self.assertContains(response, "Oczekujący 00")
        self.assertNotContains(response, "Oczekujący 11")
        self.assertNotContains(response, "Obcy komentarz")
        self.assertContains(self.client.get(reverse("catalog:moderation_comments") + "?page=2"), "Oczekujący 11")
        self.client.force_login(self.admin)
        self.assertContains(self.client.get(reverse("catalog:moderation_comments")), "Obcy komentarz")

    def test_submission_queue_shows_pending_courses(self):
        Course.objects.create(title="Zgłoszony kurs", category=self.category, author=self.student, status="pending", is_published=False)
        Course.objects.create(title="Obce zgłoszenie", category=self.other_category, author=self.student, status="pending", is_published=False)
        self.client.force_login(self.teacher)
        response = self.client.get(reverse("catalog:moderation_submissions"))
        self.assertContains(response, "Zgłoszony kurs")
        self.assertNotContains(response, "Obce zgłoszenie")

    def test_navbar_link_only_for_teachers_and_admins(self):
        url = reverse("catalog:moderation_submissions")
        self.client.force_login(self.student)
        self.assertNotContains(self.client.get(reverse("catalog:course_list")), f'href="{url}"')
        self.client.force_login(self.teacher)
        self.assertContains(self.client.get(reverse("catalog:course_list")), f'href="{url}"')


class SubmissionThreadTests(CommunityFixture):
    def setUp(self):
        super().setUp()
        self.submitted = Course.objects.create(
            title="Kurs ucznia", category=self.category, author=self.student, status="pending", is_published=False
        )
        self.thread_url = reverse("catalog:submission_thread", kwargs={"slug": self.submitted.slug})
        self.decide_url = reverse("catalog:submission_decide", kwargs={"slug": self.submitted.slug})

    def test_only_author_and_category_teachers_see_thread(self):
        for user, status in ((self.student, 200), (self.teacher, 200), (self.admin, 200), (self.other, 404), (self.other_teacher, 404)):
            self.client.force_login(user)
            self.assertEqual(self.client.get(self.thread_url).status_code, status, user.username)

    def test_messages_persist_and_notify_other_side(self):
        self.client.force_login(self.student)
        self.client.post(self.thread_url, {"content": "Proszę o sprawdzenie"})
        self.assertTrue(Notification.objects.filter(user=self.teacher, kind="submission_message").exists())

        self.client.force_login(self.teacher)
        self.client.post(self.thread_url, {"content": "Dodaj test końcowy"})
        self.assertTrue(Notification.objects.filter(user=self.student, kind="submission_message").exists())

        self.client.logout()
        self.client.force_login(self.student)  # „po ponownym zalogowaniu”
        response = self.client.get(self.thread_url)
        self.assertContains(response, "Proszę o sprawdzenie")
        self.assertContains(response, "Dodaj test końcowy")

    def test_empty_message_shows_error(self):
        self.client.force_login(self.student)
        response = self.client.post(self.thread_url, {"content": "   "})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Wpisz treść wiadomości.")
        self.assertFalse(CourseSubmissionMessage.objects.exists())

    def test_teacher_approves_submission(self):
        self.client.force_login(self.teacher)
        self.assertEqual(self.client.get(self.decide_url).status_code, 405)
        self.client.post(self.decide_url, {"action": "approve"})
        self.submitted.refresh_from_db()
        self.assertEqual(self.submitted.status, "approved")
        self.assertTrue(self.submitted.is_published)
        self.assertTrue(Notification.objects.filter(user=self.student, kind="submission_status").exists())

    def test_reject_requires_reason_and_records_it(self):
        self.client.force_login(self.teacher)
        response = self.client.post(self.decide_url, {"action": "reject", "reason": ""})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'aria-invalid="true"')
        self.submitted.refresh_from_db()
        self.assertEqual(self.submitted.status, "pending")

        self.client.post(self.decide_url, {"action": "reject", "reason": "Brakuje lekcji o testach."})
        self.submitted.refresh_from_db()
        self.assertEqual(self.submitted.status, "rejected")
        self.assertEqual(self.submitted.rejection_reason, "Brakuje lekcji o testach.")
        self.assertTrue(CourseSubmissionMessage.objects.filter(course=self.submitted, content__contains="Brakuje lekcji").exists())
        self.assertTrue(Notification.objects.filter(user=self.student, kind="submission_status").exists())

    def test_author_and_foreign_teacher_cannot_decide(self):
        for user in (self.student, self.other_teacher):
            self.client.force_login(user)
            self.assertEqual(self.client.post(self.decide_url, {"action": "approve"}).status_code, 404)
        self.submitted.refresh_from_db()
        self.assertEqual(self.submitted.status, "pending")

    def test_course_page_links_to_thread_for_author(self):
        self.client.force_login(self.student)
        self.assertContains(self.client.get(self.detail_url(self.submitted)), self.thread_url)


class CategoryManagementTests(CommunityFixture):
    def setUp(self):
        super().setUp()
        self.sub = Category.objects.create(name="Django", parent=self.category)
        self.manage_url = reverse("catalog:category_manage")

    def test_access_admin_and_teachers_only(self):
        self.client.force_login(self.student)
        self.assertEqual(self.client.get(self.manage_url).status_code, 404)
        self.client.force_login(self.teacher)
        response = self.client.get(self.manage_url)
        self.assertContains(response, "Django")
        self.assertNotContains(response, "Grafika")  # cudza gałąź
        self.client.force_login(self.admin)
        self.assertContains(self.client.get(self.manage_url), "Grafika")

    def test_teacher_adds_subcategory_only_in_own_branch(self):
        self.client.force_login(self.teacher)
        url = reverse("catalog:category_create")
        self.client.post(url, {"name": "FastAPI", "parent": self.category.pk, "is_active": "on"})
        created = Category.objects.get(name="FastAPI")
        self.assertEqual(created.parent, self.category)
        response = self.client.post(url, {"name": "Obca", "parent": self.other_category.pk, "is_active": "on"})
        self.assertEqual(response.status_code, 200)
        self.assertFalse(Category.objects.filter(name="Obca").exists())
        response = self.client.post(url, {"name": "Główna", "parent": "", "is_active": "on"})
        self.assertFalse(Category.objects.filter(name="Główna").exists())
        self.assertEqual(self.client.get(url + f"?rodzic={self.other_category.pk}").status_code, 404)

    def test_admin_creates_root_and_assigns_teacher(self):
        self.client.force_login(self.admin)
        self.client.post(reverse("catalog:category_create"), {"name": "Języki", "parent": "", "is_active": "on", "moderators": [self.other_teacher.pk]})
        created = Category.objects.get(name="Języki")
        self.assertIsNone(created.parent)
        self.assertEqual(list(created.moderators.all()), [self.other_teacher])

    def test_teacher_cannot_edit_assigned_root_or_change_moderators(self):
        self.client.force_login(self.teacher)
        self.assertEqual(self.client.get(reverse("catalog:category_edit", kwargs={"pk": self.category.pk})).status_code, 404)
        response = self.client.get(reverse("catalog:category_edit", kwargs={"pk": self.sub.pk}))
        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, 'name="moderators"')
        self.client.post(reverse("catalog:category_edit", kwargs={"pk": self.sub.pk}), {"name": "Django 5", "parent": self.category.pk, "is_active": "on"})
        self.sub.refresh_from_db()
        self.assertEqual(self.sub.name, "Django 5")

    def test_duplicate_name_and_cycle_are_rejected(self):
        self.client.force_login(self.admin)
        response = self.client.post(reverse("catalog:category_create"), {"name": "django", "parent": self.category.pk, "is_active": "on"})
        self.assertContains(response, "jest już kategoria o tej nazwie")
        response = self.client.post(
            reverse("catalog:category_edit", kwargs={"pk": self.category.pk}),
            {"name": "Python", "parent": self.sub.pk, "is_active": "on"},
        )
        self.assertEqual(response.status_code, 200)
        self.category.refresh_from_db()
        self.assertIsNone(self.category.parent)

    def test_move_toggle_and_delete(self):
        second = Category.objects.create(name="Flask", parent=self.category, order=20)
        self.sub.order = 10
        self.sub.save()
        self.client.force_login(self.teacher)
        move = reverse("catalog:category_move", kwargs={"pk": second.pk, "direction": "gora"})
        self.assertEqual(self.client.get(move).status_code, 405)
        self.client.post(move)
        self.assertEqual([c.name for c in self.category.get_children()], ["Flask", "Django"])

        self.client.post(reverse("catalog:category_toggle", kwargs={"pk": second.pk}))
        second.refresh_from_db()
        self.assertFalse(second.is_active)

        Course.objects.create(title="Kurs Django", category=self.sub)
        self.client.post(reverse("catalog:category_delete", kwargs={"pk": self.sub.pk}))
        self.assertTrue(Category.objects.filter(pk=self.sub.pk).exists())  # ma kurs
        self.client.post(reverse("catalog:category_delete", kwargs={"pk": second.pk}))
        self.assertFalse(Category.objects.filter(pk=second.pk).exists())

    def test_catalog_lists_subcategories_with_counts(self):
        Course.objects.create(title="Kurs Django", category=self.sub)
        response = self.client.get(reverse("catalog:course_list") + f"?category={self.category.slug}")
        self.assertContains(response, 'aria-label="Podkategorie: Python"')
        self.assertContains(response, f'href="?category={self.sub.slug}"')


class CommunityPolishTests(CommunityFixture):
    def test_moderator_removal_notifies_author(self):
        comment = self.comment(self.student, "Niestosowny", approved=False)
        self.client.force_login(self.teacher)
        self.client.post(reverse("catalog:comment_delete", kwargs={"pk": comment.pk}))
        note = Notification.objects.get(user=self.student, kind="comment_rejected")
        self.assertIn("Niestosowny", note.message)

    def test_own_removal_does_not_notify(self):
        comment = self.comment(self.student, "Moje")
        self.client.force_login(self.student)
        self.client.post(reverse("catalog:comment_delete", kwargs={"pk": comment.pk}))
        self.assertFalse(Notification.objects.filter(kind="comment_rejected").exists())

    def test_unread_notifications_announced_right_after_login(self):
        from apps.core.notifications import notify

        notify(self.student, Notification.Kind.SYSTEM, "Test")
        response = self.client.post(reverse("accounts:login"), {"login": "uczen", "password": "Haslo12345!"}, follow=True)
        self.assertContains(response, "Masz 1 nieprzeczytane powiadomienie")

    def test_comment_order_options(self):
        old = self.comment(self.other, "Stary wątek")
        best = self.comment(self.other, "Najlepszy wątek")
        newest = self.comment(self.other, "Nowy wątek")
        CommentVote.objects.create(comment=best, voter=self.student, value=1)
        Comment.objects.filter(pk=old.pk).update(created_at=old.created_at.replace(year=2020))
        tree = lambda order: [n.comment.content for n in build_comment_tree(self.course, self.student, order)]  # noqa: E731
        self.assertEqual(tree("najnowsze")[0], newest.content)
        self.assertEqual(tree("najstarsze")[0], "Stary wątek")
        self.assertEqual(tree("najlepsze")[0], "Najlepszy wątek")
        response = self.client.get(self.detail_url() + "?komentarze=najlepsze")
        self.assertContains(response, 'href="?komentarze=najlepsze#comments" class="segmented__option" aria-current="page"')

    def test_authored_list_links_to_submission_thread(self):
        course = Course.objects.create(title="Moja propozycja", category=self.category, author=self.student, status="rejected", is_published=False, rejection_reason="Za krótki")
        CourseSubmissionMessage.objects.create(course=course, author=self.teacher, content="Dodaj lekcje")
        self.client.force_login(self.student)
        response = self.client.get(reverse("learning:my_courses"))
        self.assertContains(response, reverse("catalog:submission_thread", kwargs={"slug": course.slug}))
        self.assertContains(response, "Powód odrzucenia: Za krótki")
        self.assertContains(response, "1 wiadomość w zgłoszeniu")

    def test_profile_shows_rank_ladder_and_points_to_next(self):
        self.client.force_login(self.student)
        response = self.client.get(reverse("accounts:profile"))
        self.assertContains(response, 'aria-current="step"')
        self.assertContains(response, "Do rangi <strong>Pomocnik</strong> brakuje Ci <strong>11 punktów</strong>")
        self.assertContains(response, "z 3")
