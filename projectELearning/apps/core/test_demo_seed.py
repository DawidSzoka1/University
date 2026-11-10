"""Dane demonstracyjne: kompletność, zgodność z regułami formularzy i powtarzalność komendy."""

import shutil
import tempfile
from io import StringIO

from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.management import call_command
from django.test import TestCase, override_settings

from apps.accounts.models import CustomUser
from apps.catalog.models import Comment, Course, Enrollment
from apps.core.demo import data
from apps.learning.media import validate_attachment, validate_video_file, validate_video_url
from apps.learning.models import Lesson, LessonAttachment, Quiz

MEDIA = tempfile.mkdtemp()


def tearDownModule():
    shutil.rmtree(MEDIA, ignore_errors=True)


class DemoDataRulesTests(TestCase):
    """Dane w data.py muszą przechodzić te same walidacje co formularz nauczyciela."""

    def test_videos_and_files_pass_platform_validation(self):
        from apps.core.management.commands.seed_demo import ASSETS

        for course in data.COURSES:
            for section in course["sections"]:
                for lesson in section["lessons"]:
                    with self.subTest(lesson=lesson["title"]):
                        video = lesson["video"] or ""
                        if video.startswith("youtube:"):
                            validate_video_url(f"https://www.youtube.com/watch?v={video[8:]}")
                        elif video.startswith("file:"):
                            name = video[5:]
                            validate_video_file(SimpleUploadedFile(name, (ASSETS / name).read_bytes()))
                        for filename, content in lesson["files"]:
                            validate_attachment(SimpleUploadedFile(filename, content.encode()))

    def test_ten_published_courses_with_program_and_quiz(self):
        published = [c for c in data.COURSES if c.get("status", "approved") == "approved"]
        self.assertEqual(len(published), 10)
        for course in published:
            with self.subTest(course=course["title"]):
                self.assertGreaterEqual(len(course["sections"]), 2)
                self.assertTrue(course["quiz"]["questions"])


@override_settings(MEDIA_ROOT=MEDIA)
class SeedDemoCommandTests(TestCase):
    def run_seed(self, *args):
        call_command("seed_demo", *args, stdout=StringIO())

    def test_builds_complete_platform_and_is_idempotent(self):
        self.run_seed()
        self.run_seed()  # drugi raz niczego nie dubluje

        self.assertEqual(Course.objects.filter(status="approved", is_published=True).count(), 10)
        self.assertEqual(Course.objects.filter(status="pending").count(), 1)
        self.assertEqual(Course.objects.filter(status="rejected").count(), 1)
        self.assertEqual(CustomUser.objects.filter(role="moderator").count(), len(data.TEACHERS))
        self.assertEqual(CustomUser.objects.filter(role="student").count(), len(data.STUDENTS))
        # Galeria: zdjęcia przeskalowane przy zapisie, kilka układów, dokładnie jeden wyświetlany
        from apps.gallery.models import GalleryItem, SliderPreset

        self.assertEqual(GalleryItem.objects.count(), len(data.GALLERY))
        self.assertTrue(all(i.was_resized and i.width <= 1600 for i in GalleryItem.objects.all()))
        self.assertEqual(SliderPreset.objects.count(), len(data.SLIDER_PRESETS))
        self.assertEqual(SliderPreset.objects.filter(is_active=True).count(), 1)
        expected_quizzes = sum(bool(c.get("quiz")) + len(c.get("quizzes", [])) for c in data.COURSES)
        self.assertEqual(Quiz.objects.count(), expected_quizzes)
        # Testy: test obowiązkowy w środku kursu, podejście do oceny, niezaliczone, dyplom i kurs bez dyplomu
        from apps.learning.models import Attempt, Certificate

        self.assertTrue(Quiz.objects.filter(is_required=True, after_lesson__isnull=False).exists())
        self.assertEqual(Attempt.objects.filter(status="pending_review").count(), len(data.PENDING_ATTEMPTS))
        self.assertTrue(Attempt.objects.filter(status="graded", passed=False).exists())
        self.assertTrue(Certificate.objects.filter(user__username="student2", course__title="Python dla początkujących").exists())
        linux = Enrollment.objects.get(user__username="michal.kaczmarek", course__title="Linux i terminal na co dzień")
        self.assertTrue(linux.is_completed)
        self.assertFalse(Certificate.objects.filter(course__title="Linux i terminal na co dzień").exists())
        self.assertEqual(Enrollment.objects.count(), len(data.ENROLLMENTS))
        self.assertEqual(Comment.objects.filter(is_approved=False).count(), 1)
        self.assertTrue(Lesson.objects.exclude(video_url="").exists())
        self.assertTrue(Lesson.objects.exclude(video_file="").exists())
        self.assertTrue(Lesson.objects.filter(content_html__contains="<figure").exists())
        self.assertGreater(LessonAttachment.objects.count(), 10)
        self.assertTrue(all(c.thumbnail for c in Course.objects.filter(status="approved")))
        # Sklep: zamówienia w każdym statusie, kody dla dodatkowych sztuk, koszyk z kumulacją
        from apps.orders.models import AccessCode, CartItem, Order

        self.assertEqual(Order.objects.count(), len(data.ORDERS))
        self.assertEqual(set(Order.objects.values_list("status", flat=True)), {"paid", "pending_payment", "completed", "cancelled"})
        self.assertEqual(AccessCode.objects.count(), 1)
        self.assertTrue(CartItem.objects.filter(quantity=2).exists())
        for order in Order.objects.filter(status__in=["paid", "completed"]):
            for item in order.items.all():
                self.assertTrue(Enrollment.objects.filter(user=order.user, course=item.course).exists())
        # Asystent AI: historia rozmów z lekcją i odmową w trakcie testu
        from apps.chatbot.models import ChatMessage, ChatSession

        self.assertEqual(ChatSession.objects.count(), len(data.CHAT_CONVERSATIONS))
        self.assertTrue(ChatMessage.objects.filter(is_refusal=True).exists())
        self.assertTrue(ChatMessage.objects.filter(lesson__title="Zmienne i typy danych").exists())

        admin = CustomUser.objects.get(username="admin")
        self.assertTrue(admin.is_superuser and admin.role == "admin")
        self.assertTrue(CustomUser.objects.get(username="student1").check_password(data.PASSWORD))
        self.assertEqual(CustomUser.objects.get(username="natalia.pawlak").rank_title, "Pomocnik")
        # Wątki odpowiedzi: głębiej niż wcięcia widoku i bez ocen w odpowiedziach
        from apps.catalog.comments import MAX_DEPTH
        deepest = Comment.objects.get(content="Dziękuję, czekam!")
        depth = 0
        while deepest.parent_id:
            deepest, depth = deepest.parent, depth + 1
        self.assertGreater(depth, MAX_DEPTH)
        self.assertFalse(Comment.objects.filter(parent__isnull=False, rating__isnull=False).exists())
        self.assertTrue(Comment.objects.filter(parent__isnull=True, rating__isnull=True, is_approved=True).exists())
        # Uczeń widzi ukończone lekcje jako postęp w zapisie
        enrollment = Enrollment.objects.get(user__username="student2", course__title="Python dla początkujących")
        self.assertEqual(enrollment.progress_percent, 100)
        self.assertTrue(enrollment.is_completed)

    def test_reset_rebuilds_from_scratch(self):
        self.run_seed()
        CustomUser.objects.filter(username="student1").update(first_name="Zmienione")
        self.run_seed("--reset")
        self.assertEqual(CustomUser.objects.get(username="student1").first_name, "Anna")
        self.assertEqual(Course.objects.filter(title="Python dla początkujących").count(), 1)
