from decimal import Decimal
from django.contrib.auth import get_user_model
from django.core.exceptions import ValidationError
from django.test import TestCase

from apps.catalog.models import (
    Category,
    Comment,
    CommentVote,
    Course,
    CourseSubmissionMessage,
    Enrollment,
    sanitize_html,
)

User = get_user_model()


class CategoryModelTests(TestCase):
    def setUp(self):
        self.root_cat = Category.objects.create(name="Programowanie")
        self.sub_cat = Category.objects.create(name="Python", parent=self.root_cat)

    def test_category_slug_generated_automatically(self):
        self.assertEqual(self.root_cat.slug, "programowanie")
        self.assertEqual(self.sub_cat.slug, "python")

    def test_category_mptt_hierarchy(self):
        Category.objects.rebuild()
        self.root_cat.refresh_from_db()
        self.sub_cat.refresh_from_db()

        self.assertEqual(self.sub_cat.parent, self.root_cat)
        self.assertIn(self.sub_cat, self.root_cat.get_descendants())
        self.assertEqual(list(self.sub_cat.get_ancestors(include_self=True)), [self.root_cat, self.sub_cat])


class CourseModelTests(TestCase):
    def setUp(self):
        self.teacher = User.objects.create_user(
            username="nauczyciel", email="nauczyciel@example.com", password="Pass123!", role=User.Role.MODERATOR
        )
        self.category = Category.objects.create(name="Web Development")

    def test_free_course_logic(self):
        # 1. Course created with is_free=True has price 0.00
        course = Course.objects.create(
            title="Darmowy start w HTML",
            category=self.category,
            author=self.teacher,
            is_free=True,
            price=Decimal("50.00"),  # Should be reset to 0.00
        )
        self.assertTrue(course.is_course_free)
        self.assertEqual(course.price, Decimal("0.00"))
        self.assertEqual(course.get_price_display(), "Bezpłatny")

        # 2. Course created with price 0.00 automatically sets is_free=True
        course2 = Course.objects.create(
            title="Darmowy CSS",
            category=self.category,
            author=self.teacher,
            price=Decimal("0.00"),
        )
        self.assertTrue(course2.is_free)
        self.assertTrue(course2.is_course_free)
        self.assertEqual(course2.get_price_display(), "Bezpłatny")

    def test_paid_course_logic(self):
        course = Course.objects.create(
            title="Zaawansowany Django",
            category=self.category,
            author=self.teacher,
            price=Decimal("199.99"),
            is_free=False,
        )
        self.assertFalse(course.is_course_free)
        self.assertFalse(course.is_free)
        self.assertEqual(course.price, Decimal("199.99"))
        self.assertEqual(course.get_price_display(), "199,99\u00a0zł")

    def test_unique_slug_generation_on_collision(self):
        course1 = Course.objects.create(title="Podstawy Gita", category=self.category)
        course2 = Course.objects.create(title="Podstawy Gita", category=self.category)

        self.assertEqual(course1.slug, "podstawy-gita")
        self.assertEqual(course2.slug, "podstawy-gita-1")

    def test_html_description_sanitization_removes_xss(self):
        raw_html = '<p>Dzień dobry! <script>alert("hacked")</script> <a href="https://example.com" onclick="steal()">Link</a></p>'
        sanitized = sanitize_html(raw_html)

        self.assertNotIn("<script>", sanitized)
        self.assertNotIn("</script>", sanitized)
        self.assertNotIn("onclick", sanitized)
        self.assertIn("<p>Dzień dobry!", sanitized)
        self.assertIn('<a href="https://example.com">Link</a>', sanitized)


class EnrollmentModelTests(TestCase):
    def setUp(self):
        self.student = User.objects.create_user(
            username="student_test", email="student@example.com", password="Pass123!"
        )
        self.category = Category.objects.create(name="AI")
        self.course = Course.objects.create(title="Ollama i Bielik", category=self.category, is_free=True)

    def test_enrollment_creation(self):
        enrollment = Enrollment.objects.create(user=self.student, course=self.course)
        self.assertEqual(enrollment.user, self.student)
        self.assertEqual(enrollment.course, self.course)
        self.assertIn(enrollment, self.student.enrollments.all())

    def test_enrollment_unique_constraint(self):
        Enrollment.objects.create(user=self.student, course=self.course)
        with self.assertRaises(Exception):
            Enrollment.objects.create(user=self.student, course=self.course)

    def test_enrollment_and_completion_updates_course_stats(self):
        # 1. Enrolling student increases students_count
        enrollment = Enrollment.objects.create(user=self.student, course=self.course)
        self.course.refresh_from_db()
        self.assertEqual(self.course.students_count, 1)
        self.assertEqual(self.course.completed_count, 0)

        # 2. Marking course completed increases completed_count
        enrollment.mark_completed()
        self.course.refresh_from_db()
        self.assertEqual(self.course.students_count, 1)
        self.assertEqual(self.course.completed_count, 1)


class CourseSearchAndSortingTests(TestCase):
    def setUp(self):
        self.category = Category.objects.create(name="Programowanie")
        self.author1 = User.objects.create_user(
            username="jankowalski", first_name="Jan", last_name="Kowalski", email="jan@example.com", password="Pass123!"
        )
        self.author2 = User.objects.create_user(
            username="annakowalska", first_name="Anna", last_name="Kowalska", email="anna@example.com", password="Pass123!"
        )

        self.course_py = Course.objects.create(
            title="Python dla początkujących",
            category=self.category,
            author=self.author1,
            is_free=True,
            price=Decimal("0.00"),
        )
        self.course_django = Course.objects.create(
            title="Django Web Framework",
            category=self.category,
            author=self.author1,
            price=Decimal("120.00"),
        )
        self.course_fastapi = Course.objects.create(
            title="Mikroserwisy FastAPI",
            category=self.category,
            author=self.author2,
            price=Decimal("80.00"),
        )

        # Setup stats
        self.course_py.students_count = 100
        self.course_py.completed_count = 80
        self.course_py.average_rating = Decimal("4.90")
        self.course_py.save()

        self.course_django.students_count = 50
        self.course_django.completed_count = 20
        self.course_django.average_rating = Decimal("4.20")
        self.course_django.save()

        self.course_fastapi.students_count = 20
        self.course_fastapi.completed_count = 15
        self.course_fastapi.average_rating = Decimal("4.70")
        self.course_fastapi.save()

    def test_search_by_course_name_and_author_name(self):
        # Search by course title
        py_results = list(Course.objects.search("Python"))
        self.assertEqual(py_results, [self.course_py])

        # Search by author first/last name
        jan_results = list(Course.objects.search("Jan"))
        self.assertEqual(set(jan_results), {self.course_py, self.course_django})

        anna_results = list(Course.objects.search("Anna"))
        self.assertEqual(anna_results, [self.course_fastapi])

        # Search by author username
        user_results = list(Course.objects.search("annakowalska"))
        self.assertEqual(user_results, [self.course_fastapi])

    def test_sort_by_popularity(self):
        sorted_courses = list(Course.objects.sort_by("popularity"))
        self.assertEqual(sorted_courses, [self.course_py, self.course_django, self.course_fastapi])

    def test_sort_by_completions(self):
        sorted_courses = list(Course.objects.sort_by("completions"))
        self.assertEqual(sorted_courses, [self.course_py, self.course_django, self.course_fastapi])

    def test_sort_by_rating(self):
        sorted_courses = list(Course.objects.sort_by("rating"))
        self.assertEqual(sorted_courses, [self.course_py, self.course_fastapi, self.course_django])

    def test_sort_by_price(self):
        asc = list(Course.objects.sort_by("price_asc"))
        self.assertEqual(asc, [self.course_py, self.course_fastapi, self.course_django])

        desc = list(Course.objects.sort_by("price_desc"))
        self.assertEqual(desc, [self.course_django, self.course_fastapi, self.course_py])


class CommentRatingRecalculationTests(TestCase):
    def setUp(self):
        self.category = Category.objects.create(name="AI")
        self.course = Course.objects.create(title="Sztuczna inteligencja", category=self.category)
        self.student1 = User.objects.create_user(username="s1", email="s1@ex.com", password="P1!")
        self.student2 = User.objects.create_user(username="s2", email="s2@ex.com", password="P1!")

    def test_comment_updates_course_average_rating(self):
        Comment.objects.create(course=self.course, author=self.student1, content="Dobry", rating=4, is_approved=True)
        self.course.refresh_from_db()
        self.assertEqual(self.course.reviews_count, 1)
        self.assertEqual(self.course.average_rating, Decimal("4.00"))

        Comment.objects.create(course=self.course, author=self.student2, content="Super!", rating=5, is_approved=True)
        self.course.refresh_from_db()
        self.assertEqual(self.course.reviews_count, 2)
        self.assertEqual(self.course.average_rating, Decimal("4.50"))



class CommentAndReputationTests(TestCase):
    def setUp(self):
        self.author = User.objects.create_user(
            username="autor_recenzji", email="autor@example.com", password="Pass123!"
        )
        self.voter1 = User.objects.create_user(
            username="glosujacy1", email="v1@example.com", password="Pass123!"
        )
        self.voter2 = User.objects.create_user(
            username="glosujacy2", email="v2@example.com", password="Pass123!"
        )
        self.category = Category.objects.create(name="DevOps")
        self.course = Course.objects.create(title="Kubernetes w pigułce", category=self.category)
        self.comment = Comment.objects.create(
            course=self.course, author=self.author, content="Świetny kurs!", rating=5
        )

    def test_comment_votes_and_reputation_recalculation(self):
        # 1. Author cannot vote on their own comment
        with self.assertRaises(ValidationError):
            CommentVote.objects.create(comment=self.comment, voter=self.author, value=1)

        # 2. Voter1 upvotes (+1)
        CommentVote.objects.create(comment=self.comment, voter=self.voter1, value=1)
        self.author.refresh_from_db()
        self.assertEqual(self.author.reputation_points, 1)

        # 3. Voter2 upvotes (+1)
        CommentVote.objects.create(comment=self.comment, voter=self.voter2, value=1)
        self.author.refresh_from_db()
        self.assertEqual(self.author.reputation_points, 2)
        self.assertEqual(self.comment.score, 2)

    def test_changed_vote_recalculates_reputation(self):
        vote = CommentVote.objects.create(comment=self.comment, voter=self.voter1, value=1)
        vote.value = -1
        vote.save()
        self.author.refresh_from_db()
        self.assertEqual(self.author.reputation_points, -1)
        self.assertEqual(self.author.rank_title, "Obserwator")

    def test_deleted_vote_recalculates_reputation(self):
        vote = CommentVote.objects.create(comment=self.comment, voter=self.voter1, value=1)
        vote.delete()
        self.author.refresh_from_db()
        self.assertEqual(self.author.reputation_points, 0)

    def test_deleting_comment_removes_its_points(self):
        CommentVote.objects.create(comment=self.comment, voter=self.voter1, value=1)
        CommentVote.objects.create(comment=self.comment, voter=self.voter2, value=1)
        self.comment.delete()
        self.author.refresh_from_db()
        self.assertEqual(self.author.reputation_points, 0)


class CommentApprovalTests(TestCase):
    def setUp(self):
        self.author = User.objects.create_user(username="komentujacy", password="Pass123!")
        self.course = Course.objects.create(title="Git od zera", category=Category.objects.create(name="Narzędzia"))

    def test_new_comment_waits_for_approval(self):
        comment = Comment.objects.create(course=self.course, author=self.author, content="Pytanie", rating=4)
        self.assertFalse(comment.is_approved)

    def test_pending_comment_is_hidden_and_not_counted(self):
        Comment.objects.create(course=self.course, author=self.author, content="Ukryty komentarz", rating=1)
        self.course.refresh_from_db()
        self.assertEqual(self.course.reviews_count, 0)
        response = self.client.get(f"/catalog/{self.course.slug}/")
        self.assertNotContains(response, "Ukryty komentarz")

    def test_approved_comment_counts_in_course_stats(self):
        comment = Comment.objects.create(course=self.course, author=self.author, content="OK", rating=4)
        comment.is_approved = True
        comment.save()
        self.course.refresh_from_db()
        self.assertEqual(self.course.reviews_count, 1)
        self.assertEqual(self.course.average_rating, Decimal("4.00"))



class CourseListViewTests(TestCase):
    """Testy widoku listy kursów, filtrów, wyszukiwania, sortowania i paginacji."""

    def setUp(self):
        self.teacher1 = User.objects.create_user(
            username="autor_jan", email="jan@example.com", password="Pass123!", role=User.Role.MODERATOR
        )
        self.teacher2 = User.objects.create_user(
            username="autor_adam", email="adam@example.com", password="Pass123!", role=User.Role.MODERATOR
        )
        self.cat_dev = Category.objects.create(name="Programowanie", slug="programowanie")
        self.cat_ai = Category.objects.create(name="AI", slug="ai")

        self.c1 = Course.objects.create(
            title="Kurs Python Podstawy",
            category=self.cat_dev,
            author=self.teacher1,
            is_free=True,
            price=Decimal("0.00"),
            difficulty_level=Course.Level.BEGINNER,
            status=Course.Status.APPROVED,
            is_published=True,
            students_count=100,
            average_rating=Decimal("4.8"),
        )
        self.c2 = Course.objects.create(
            title="Kurs Django Zaawansowany",
            category=self.cat_dev,
            author=self.teacher1,
            is_free=False,
            price=Decimal("199.99"),
            difficulty_level=Course.Level.ADVANCED,
            status=Course.Status.APPROVED,
            is_published=True,
            students_count=50,
            average_rating=Decimal("4.5"),
        )
        self.c3 = Course.objects.create(
            title="Wprowadzenie do Sieci Neuronowych",
            category=self.cat_ai,
            author=self.teacher2,
            is_free=False,
            price=Decimal("299.99"),
            difficulty_level=Course.Level.INTERMEDIATE,
            status=Course.Status.APPROVED,
            is_published=True,
            students_count=20,
            average_rating=Decimal("4.9"),
        )
        self.c_draft = Course.objects.create(
            title="Kurs Nieopublikowany Szkic",
            category=self.cat_dev,
            author=self.teacher1,
            status=Course.Status.DRAFT,
            is_published=False,
        )

    def test_course_list_shows_only_published_courses(self):
        response = self.client.get("/catalog/")
        self.assertEqual(response.status_code, 200)
        content = response.content.decode()
        self.assertIn("Kurs Python Podstawy", content)
        self.assertIn("Kurs Django Zaawansowany", content)
        self.assertIn("Wprowadzenie do Sieci Neuronowych", content)
        self.assertNotIn("Kurs Nieopublikowany Szkic", content)

    def test_search_by_title_and_author(self):
        # 1. Search by title keyword
        resp1 = self.client.get("/catalog/?q=Django")
        self.assertIn("Kurs Django Zaawansowany", resp1.content.decode())
        self.assertNotIn("Kurs Python Podstawy", resp1.content.decode())

        # 2. Search by author username
        resp2 = self.client.get("/catalog/?q=autor_adam")
        self.assertIn("Wprowadzenie do Sieci Neuronowych", resp2.content.decode())
        self.assertNotIn("Kurs Python Podstawy", resp2.content.decode())

    def test_filter_by_price_type(self):
        # 1. Free only
        resp_free = self.client.get("/catalog/?type=free")
        self.assertIn("Kurs Python Podstawy", resp_free.content.decode())
        self.assertNotIn("Kurs Django Zaawansowany", resp_free.content.decode())

        # 2. Paid only
        resp_paid = self.client.get("/catalog/?type=paid")
        self.assertIn("Kurs Django Zaawansowany", resp_paid.content.decode())
        self.assertNotIn("Kurs Python Podstawy", resp_paid.content.decode())

    def test_filter_by_category(self):
        resp = self.client.get("/catalog/?category=ai")
        self.assertIn("Wprowadzenie do Sieci Neuronowych", resp.content.decode())
        self.assertNotIn("Kurs Python Podstawy", resp.content.decode())

    def test_filter_by_difficulty(self):
        resp = self.client.get("/catalog/?difficulty=beginner")
        self.assertIn("Kurs Python Podstawy", resp.content.decode())
        self.assertNotIn("Kurs Django Zaawansowany", resp.content.decode())

    def test_empty_state_rendered_when_no_matches(self):
        resp = self.client.get("/catalog/?q=ZupełnieNieistniejącyTytułXYZ")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("Nie znaleziono kursów", resp.content.decode())
        self.assertIn("empty-state", resp.content.decode())

    def test_pagination_and_query_retention(self):
        from apps.core.templatetags.core_tags import url_replace
        from django.test import RequestFactory
        factory = RequestFactory()
        req = factory.get("/catalog/?q=Python&category=programowanie&page=1")
        context = {"request": req}

        # Changing page retains q and category
        replaced = url_replace(context, page=2)
        self.assertIn("q=Python", replaced)
        self.assertIn("category=programowanie", replaced)
        self.assertIn("page=2", replaced)


class CourseEnrollAndPagesTests(TestCase):
    def setUp(self):
        self.student = User.objects.create_user(username="uczen", password="Haslo12345!")
        self.parent = Category.objects.create(name="Informatyka")
        self.category = Category.objects.create(name="Python", parent=self.parent)
        self.free = Course.objects.create(title="Darmowy Python", category=self.category, price=Decimal("0.00"))
        self.paid = Course.objects.create(title="Płatny Django", category=self.category, price=Decimal("99.00"))

    def test_enroll_requires_post(self):
        self.client.force_login(self.student)
        response = self.client.get(f"/catalog/{self.free.slug}/zapisz/")
        self.assertEqual(response.status_code, 405)

    def test_enroll_requires_login(self):
        response = self.client.post(f"/catalog/{self.free.slug}/zapisz/")
        self.assertEqual(response.status_code, 302)
        self.assertFalse(Enrollment.objects.exists())

    def test_free_course_enroll_is_idempotent(self):
        self.client.force_login(self.student)
        self.client.post(f"/catalog/{self.free.slug}/zapisz/")
        self.client.post(f"/catalog/{self.free.slug}/zapisz/")
        self.assertEqual(Enrollment.objects.filter(user=self.student, course=self.free).count(), 1)

    def test_paid_course_cannot_be_enrolled_directly(self):
        self.client.force_login(self.student)
        self.client.post(f"/catalog/{self.paid.slug}/zapisz/")
        self.assertFalse(Enrollment.objects.filter(course=self.paid).exists())

    def test_my_courses_lists_enrollments(self):
        Enrollment.objects.create(user=self.student, course=self.free)
        self.client.force_login(self.student)
        response = self.client.get("/learning/my-courses/")
        self.assertContains(response, "Darmowy Python")

    def test_detail_shows_category_breadcrumbs(self):
        response = self.client.get(f"/catalog/{self.free.slug}/")
        content = response.content.decode()
        self.assertIn('aria-label="Okruszki"', content)
        self.assertLess(content.index(">Informatyka<"), content.index(">Python<"))

    def test_course_cards_link_to_detail_page(self):
        response = self.client.get("/catalog/")
        self.assertContains(response, f'href="/catalog/{self.free.slug}/"')

    def test_filters_do_not_submit_on_change(self):
        response = self.client.get("/catalog/")
        self.assertNotContains(response, "onchange")
        self.assertContains(response, 'for="filter-category"')

    def test_filter_selects_are_keyboard_enhanced(self):
        content = self.client.get("/catalog/").content.decode()
        for field_id in ("filter-category", "filter-type", "filter-difficulty", "filter-sort"):
            with self.subTest(field=field_id):
                self.assertRegex(content, rf'id="{field_id}"[^>]*data-select-enhance')


class CourseCreateEditTests(TestCase):
    """Tworzenie/edycja kursu: kto publikuje od razu, kto wysyła zgłoszenie."""

    def setUp(self):
        self.root = Category.objects.create(name="Informatyka")
        self.python = Category.objects.create(name="Python", parent=self.root)
        self.design = Category.objects.create(name="Grafika")
        self.student = User.objects.create_user(username="uczen_k", password="Haslo12345!")
        self.other_student = User.objects.create_user(username="uczen_inny", password="Haslo12345!")
        self.teacher = User.objects.create_user(username="nauczyciel", password="Haslo12345!", role="moderator")
        self.root.moderators.add(self.teacher)  # przypisany do nadrzędnej — obejmuje podkategorie
        self.admin = User.objects.create_user(username="szef_k", password="Haslo12345!", role="admin")

    def payload(self, **overrides):
        data = {
            "title": "Testy w pytest",
            "category": self.python.pk,
            "description": "Krótko o testach.",
            "description_html": "<p>Program</p>",
            "difficulty_level": Course.Level.BEGINNER,
            "duration_hours": 4,
            "price": "0",
        }
        data.update(overrides)
        return data

    def test_anonymous_user_is_redirected_to_login(self):
        response = self.client.get("/catalog/create/")
        self.assertEqual(response.status_code, 302)
        self.assertIn("/accounts/login/", response["Location"])

    def create_and_submit(self, user, **overrides):
        """Krok 1 (dane) → lekcja w programie → krok 3 (wysłanie)."""
        from apps.learning.models import CourseSection, Lesson

        self.client.force_login(user)
        self.client.post("/catalog/create/", self.payload(**overrides))
        course = Course.objects.get(title=overrides.get("title", "Testy w pytest"))
        section = CourseSection.objects.create(course=course, title="Start", order=1)
        Lesson.objects.create(section=section, title="Pierwsza lekcja", order=1)
        self.client.post(f"/catalog/{course.slug}/wyslij/")
        course.refresh_from_db()
        return course

    def test_new_course_is_private_draft_and_opens_builder(self):
        self.client.force_login(self.teacher)
        response = self.client.post("/catalog/create/", self.payload())
        course = Course.objects.get(title="Testy w pytest")
        self.assertRedirects(response, f"/learning/kurs/{course.slug}/program/")
        self.assertEqual(course.status, Course.Status.DRAFT)
        self.assertFalse(course.is_published)
        self.assertEqual(course.author, self.teacher)
        self.client.force_login(self.other_student)
        self.assertEqual(self.client.get(f"/catalog/{course.slug}/").status_code, 404)

    def test_student_submission_becomes_pending(self):
        course = self.create_and_submit(self.student)
        self.assertEqual(course.status, Course.Status.PENDING)
        self.assertFalse(course.is_published)
        self.assertNotIn(course, Course.objects.published())

    def test_teacher_publishes_in_moderated_subcategory(self):
        course = self.create_and_submit(self.teacher)
        self.assertEqual(course.status, Course.Status.APPROVED)
        self.assertTrue(course.is_published)

    def test_teacher_outside_own_category_sends_submission(self):
        course = self.create_and_submit(self.teacher, category=self.design.pk)
        self.assertEqual(course.status, Course.Status.PENDING)

    def test_admin_publishes_anywhere(self):
        course = self.create_and_submit(self.admin, category=self.design.pk)
        self.assertEqual(course.status, Course.Status.APPROVED)

    def test_submit_requires_a_lesson_post_and_author(self):
        self.client.force_login(self.student)
        self.client.post("/catalog/create/", self.payload())
        course = Course.objects.get(title="Testy w pytest")
        url = f"/catalog/{course.slug}/wyslij/"
        self.assertEqual(self.client.get(url).status_code, 405)
        self.client.post(url)
        course.refresh_from_db()
        self.assertEqual(course.status, Course.Status.DRAFT)  # brak lekcji — nic się nie zmienia
        self.client.force_login(self.other_student)
        self.assertEqual(self.client.post(url).status_code, 404)

    def test_rejected_course_can_be_resubmitted(self):
        from apps.learning.models import CourseSection, Lesson

        course = Course.objects.create(title="Odrzucony", category=self.python, author=self.student,
                                       status=Course.Status.REJECTED, is_published=False, rejection_reason="Za krótki")
        Lesson.objects.create(section=CourseSection.objects.create(course=course, title="S", order=1), title="L", order=1)
        self.client.force_login(self.student)
        self.client.post(f"/catalog/{course.slug}/edytuj/", self.payload(title="Odrzucony"))
        course.refresh_from_db()
        self.assertEqual(course.status, Course.Status.REJECTED)  # sam zapis nie wysyła
        self.client.post(f"/catalog/{course.slug}/wyslij/")
        course.refresh_from_db()
        self.assertEqual((course.status, course.rejection_reason), (Course.Status.PENDING, ""))

    def test_builder_shows_publish_step_for_drafts(self):
        self.client.force_login(self.teacher)
        self.client.post("/catalog/create/", self.payload())
        course = Course.objects.get(title="Testy w pytest")
        response = self.client.get(f"/learning/kurs/{course.slug}/program/")
        self.assertContains(response, 'aria-current="step"')
        self.assertContains(response, "jeszcze brak")
        self.assertNotContains(response, "Opublikuj kurs")

    def test_html_description_is_sanitized(self):
        self.client.force_login(self.student)
        self.client.post("/catalog/create/", self.payload(
            description_html='<p onclick="x()">Treść</p><script>alert(1)</script><a href="javascript:x()" target="_blank">l</a><u>p</u>'
        ))
        html = Course.objects.get(title="Testy w pytest").description_html
        for forbidden in ("<script", "onclick", "javascript:", "target="):
            self.assertNotIn(forbidden, html)
        self.assertIn("<u>p</u>", html)

    def test_invalid_form_keeps_entered_data(self):
        self.client.force_login(self.student)
        response = self.client.post("/catalog/create/", self.payload(title="", description="Zostaje w polu"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Zostaje w polu")
        self.assertContains(response, 'aria-invalid="true"')
        self.assertFalse(Course.objects.exists())

    def test_pending_course_visible_only_to_author_and_moderators(self):
        course = Course.objects.create(title="Szkic", category=self.python, author=self.student,
                                       status=Course.Status.PENDING, is_published=False)
        url = f"/catalog/{course.slug}/"
        self.assertEqual(self.client.get(url).status_code, 404)
        self.client.force_login(self.other_student)
        self.assertEqual(self.client.get(url).status_code, 404)
        self.client.force_login(self.student)
        self.assertContains(self.client.get(url), "oczekuje na akceptację")
        self.client.force_login(self.teacher)
        self.assertEqual(self.client.get(url).status_code, 200)

    def test_only_author_or_moderator_can_edit(self):
        course = Course.objects.create(title="Cudzy", category=self.python, author=self.student)
        self.client.force_login(self.other_student)
        self.assertEqual(self.client.get(f"/catalog/{course.slug}/edytuj/").status_code, 404)
        self.client.force_login(self.teacher)
        self.assertEqual(self.client.get(f"/catalog/{course.slug}/edytuj/").status_code, 200)

    def test_student_edit_of_published_course_requires_reapproval(self):
        course = Course.objects.create(title="Opublikowany", category=self.python, author=self.student,
                                       status=Course.Status.APPROVED, is_published=True)
        self.client.force_login(self.student)
        self.client.post(f"/catalog/{course.slug}/edytuj/", self.payload(title="Opublikowany v2"))
        course.refresh_from_db()
        self.assertEqual(course.title, "Opublikowany v2")
        self.assertEqual(course.status, Course.Status.PENDING)
        self.assertFalse(course.is_published)

    def test_teacher_edit_keeps_publication(self):
        course = Course.objects.create(title="Mój kurs", category=self.python, author=self.teacher,
                                       status=Course.Status.APPROVED, is_published=True)
        self.client.force_login(self.teacher)
        self.client.post(f"/catalog/{course.slug}/edytuj/", self.payload(title="Mój kurs v2"))
        course.refresh_from_db()
        self.assertEqual(course.status, Course.Status.APPROVED)
        self.assertTrue(course.is_published)

    def test_free_course_can_get_a_price(self):
        course = Course.objects.create(title="Był darmowy", category=self.python, author=self.teacher, is_free=True)
        self.client.force_login(self.teacher)
        self.client.post(f"/catalog/{course.slug}/edytuj/", self.payload(title="Był darmowy", price="49.90"))
        course.refresh_from_db()
        self.assertEqual(course.price, Decimal("49.90"))
        self.assertFalse(course.is_free)

    def test_thumbnail_is_validated_and_resized(self):
        import shutil
        import tempfile
        from io import BytesIO

        from django.core.files.uploadedfile import SimpleUploadedFile
        from django.test import override_settings
        from PIL import Image

        media = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, media, ignore_errors=True)
        buffer = BytesIO()
        Image.new("RGB", (3000, 2000), "orange").save(buffer, "PNG")
        image = SimpleUploadedFile("duzy.png", buffer.getvalue(), content_type="image/png")
        fake = SimpleUploadedFile("wirus.png", b"to nie jest obraz", content_type="image/png")

        with override_settings(MEDIA_ROOT=media):
            self.client.force_login(self.teacher)
            response = self.client.post("/catalog/create/", self.payload(title="Zły plik", thumbnail=fake))
            self.assertContains(response, 'id="id_thumbnail-error"')
            self.assertFalse(Course.objects.filter(title="Zły plik").exists())
            self.client.post("/catalog/create/", self.payload(thumbnail=image))
            course = Course.objects.get(title="Testy w pytest")
            with Image.open(course.thumbnail.path) as saved:
                self.assertLessEqual(saved.width, 1280)
                self.assertLessEqual(saved.height, 720)


class ImageValidationTests(TestCase):
    def test_rejects_too_large_and_wrong_format(self):
        from io import BytesIO

        from django.core.exceptions import ValidationError
        from django.core.files.uploadedfile import SimpleUploadedFile
        from PIL import Image

        from apps.core.images import validate_image_upload

        gif = BytesIO()
        Image.new("RGB", (10, 10)).save(gif, "GIF")
        with self.assertRaisesMessage(ValidationError, "Dozwolone formaty"):
            validate_image_upload(SimpleUploadedFile("a.gif", gif.getvalue()))

        png = BytesIO()
        Image.new("RGB", (10, 10)).save(png, "PNG")
        with self.assertRaisesMessage(ValidationError, "za duży"):
            validate_image_upload(SimpleUploadedFile("a.png", png.getvalue()), max_mb=0)


class SubmissionNotificationTests(TestCase):
    def setUp(self):
        from apps.core.models import Notification

        self.Notification = Notification
        self.root = Category.objects.create(name="Technologie")
        self.child = Category.objects.create(name="Bazy danych", parent=self.root)
        self.orphan = Category.objects.create(name="Bez nauczyciela")
        self.teacher = User.objects.create_user(username="nauczyciel_n", password="Haslo12345!", role="moderator")
        self.root.moderators.add(self.teacher)
        self.admin = User.objects.create_user(username="admin_n", password="Haslo12345!", role="admin")
        self.student = User.objects.create_user(username="uczen_n", password="Haslo12345!")
        self.client.force_login(self.student)

    def submit(self, category):
        """Szkic → lekcja → wysłanie (powiadomienie pada dopiero przy wysłaniu)."""
        from apps.learning.models import CourseSection, Lesson

        self.client.post("/catalog/create/", {
            "title": "SQL od podstaw", "category": category.pk, "description": "Zapytania SELECT.",
            "description_html": "<p>Program</p>", "difficulty_level": "beginner", "duration_hours": 2, "price": "0",
        })
        course = Course.objects.get(title="SQL od podstaw")
        self.assertFalse(self.Notification.objects.exists())  # szkic nikogo nie powiadamia
        Lesson.objects.create(section=CourseSection.objects.create(course=course, title="S", order=1), title="L", order=1)
        return self.client.post(f"/catalog/{course.slug}/wyslij/")

    def test_submission_notifies_teachers_of_category_tree(self):
        self.submit(self.child)
        notification = self.Notification.objects.get(user=self.teacher)
        self.assertEqual(notification.kind, self.Notification.Kind.COURSE_SUBMITTED)
        self.assertIn("SQL od podstaw", notification.message)
        self.assertEqual(notification.url, "/catalog/sql-od-podstaw/")
        self.assertFalse(self.Notification.objects.filter(user=self.admin).exists())
        self.assertFalse(self.Notification.objects.filter(user=self.student).exists())

    def test_without_teachers_admins_are_notified(self):
        self.submit(self.orphan)
        self.assertTrue(self.Notification.objects.filter(user=self.admin, kind="course_submitted").exists())

    def test_teacher_publishing_in_own_category_notifies_nobody(self):
        self.client.force_login(self.teacher)
        self.submit(self.child)
        self.assertFalse(self.Notification.objects.exists())
