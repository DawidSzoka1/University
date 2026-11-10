import shutil
import tempfile
from decimal import Decimal
from io import BytesIO

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings

from apps.catalog.models import Category, Course, Enrollment
from apps.learning.curriculum import build_outline
from apps.learning.media import video_embed_url
from apps.learning.models import CourseSection, Lesson, LessonAttachment, LessonProgress

User = get_user_model()

MEDIA = tempfile.mkdtemp()


def tearDownModule():
    shutil.rmtree(MEDIA, ignore_errors=True)


@override_settings(MEDIA_ROOT=MEDIA)
class CurriculumTestCase(TestCase):
    def setUp(self):
        self.category = Category.objects.create(name="Programowanie")
        self.teacher = User.objects.create_user(username="nauczyciel_p", password="Haslo12345!", role="moderator")
        self.category.moderators.add(self.teacher)
        self.student = User.objects.create_user(username="uczen_p", password="Haslo12345!")
        self.outsider = User.objects.create_user(username="obcy_p", password="Haslo12345!")
        self.course = Course.objects.create(
            title="Python od zera", category=self.category, author=self.teacher,
            status=Course.Status.APPROVED, is_published=True, price=Decimal("0.00"),
        )
        self.install = CourseSection.objects.create(course=self.course, title="Instalacja", order=1)
        self.first_code = CourseSection.objects.create(course=self.course, title="Pierwsze linie kodu", order=2)
        self.windows = Lesson.objects.create(section=self.install, title="Windows", order=1, is_preview=True)
        self.mac = Lesson.objects.create(section=self.install, title="macOS", order=2)
        self.hello = Lesson.objects.create(section=self.first_code, title="Hello world", order=1)

    def program_url(self):
        return f"/learning/kurs/{self.course.slug}/program/"

    def lesson_url(self, lesson):
        return f"/learning/kurs/{self.course.slug}/lekcja/{lesson.pk}/"


class BuilderTests(CurriculumTestCase):
    def test_only_author_and_moderators_see_builder(self):
        self.client.force_login(self.outsider)
        self.assertEqual(self.client.get(self.program_url()).status_code, 404)
        self.client.force_login(self.teacher)
        response = self.client.get(self.program_url())
        self.assertContains(response, "Instalacja")
        self.assertContains(response, "2.1")

    def test_add_section_appends_at_end(self):
        self.client.force_login(self.teacher)
        self.client.post(self.program_url(), {"title": "Zmienne"})
        self.assertEqual(list(self.course.sections.values_list("title", flat=True)),
                         ["Instalacja", "Pierwsze linie kodu", "Zmienne"])

    def test_moving_is_post_only_and_reorders(self):
        self.client.force_login(self.teacher)
        url = f"/learning/kurs/{self.course.slug}/lekcja/{self.mac.pk}/przesun/up/"
        self.assertEqual(self.client.get(url).status_code, 405)
        self.client.post(url)
        self.assertEqual(list(self.install.lessons.values_list("title", flat=True)), ["macOS", "Windows"])
        self.client.post(f"/learning/kurs/{self.course.slug}/program/sekcja/{self.first_code.pk}/przesun/up/")
        self.assertEqual(self.course.sections.first(), self.first_code)

    def test_lesson_with_text_video_link_and_files(self):
        self.client.force_login(self.teacher)
        response = self.client.post(
            f"/learning/kurs/{self.course.slug}/program/sekcja/{self.first_code.pk}/nowa-lekcja/",
            {
                "section": self.first_code.pk,
                "title": "Zadanie: popraw program",
                "content_html": '<p>Pobierz plik.</p><figure class="image"><img src="/media/a.png" alt="Zrzut"><figcaption>Wynik</figcaption></figure><script>x()</script>',
                "video_url": "https://youtu.be/dQw4w9WgXcQ",
                "duration_minutes": 10,
                "new_attachments": [
                    SimpleUploadedFile("start.py", b"print('hello')"),
                    SimpleUploadedFile("dane.csv", b"a,b\n1,2"),
                ],
            },
        )
        self.assertRedirects(response, self.program_url())
        lesson = Lesson.objects.get(title="Zadanie: popraw program")
        self.assertEqual(lesson.order, 2)
        self.assertIn("<figcaption>Wynik</figcaption>", lesson.content_html)
        self.assertIn('alt="Zrzut"', lesson.content_html)
        self.assertNotIn("<script", lesson.content_html)
        self.assertEqual(sorted(lesson.attachments.values_list("original_name", flat=True)), ["dane.csv", "start.py"])

    def test_rejects_both_video_sources_and_bad_files(self):
        self.client.force_login(self.teacher)
        mp4 = SimpleUploadedFile("film.mp4", b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 32, content_type="video/mp4")
        response = self.client.post(
            f"/learning/kurs/{self.course.slug}/lekcja/{self.hello.pk}/edytuj/",
            {"section": self.first_code.pk, "title": "Hello world", "duration_minutes": 5,
             "video_url": "https://vimeo.com/123456789", "video_file": mp4,
             "new_attachments": [SimpleUploadedFile("wirus.exe", b"MZ")]},
        )
        self.assertContains(response, "Wybierz jedno źródło filmu")
        self.assertContains(response, "niedozwolony typ")

    def test_fake_video_file_is_rejected(self):
        self.client.force_login(self.teacher)
        fake = SimpleUploadedFile("film.mp4", b"to nie jest film", content_type="video/mp4")
        response = self.client.post(
            f"/learning/kurs/{self.course.slug}/lekcja/{self.hello.pk}/edytuj/",
            {"section": self.first_code.pk, "title": "Hello world", "duration_minutes": 5, "video_file": fake},
        )
        self.assertContains(response, "Dozwolone formaty filmu")

    def test_student_author_change_sends_published_course_back_to_review(self):
        own = Course.objects.create(title="Mój kurs", category=self.category, author=self.student,
                                    status=Course.Status.APPROVED, is_published=True)
        self.client.force_login(self.student)
        self.client.post(f"/learning/kurs/{own.slug}/program/", {"title": "Nowa sekcja"})
        own.refresh_from_db()
        self.assertEqual(own.status, Course.Status.PENDING)
        self.assertFalse(own.is_published)

    def test_delete_section_requires_post_and_removes_lessons(self):
        self.client.force_login(self.teacher)
        url = f"/learning/kurs/{self.course.slug}/program/sekcja/{self.install.pk}/usun/"
        self.assertEqual(self.client.get(url).status_code, 405)
        self.client.post(url)
        self.assertFalse(Lesson.objects.filter(title__in=["Windows", "macOS"]).exists())


class LearningAccessTests(CurriculumTestCase):
    def test_preview_lesson_is_open_others_need_enrollment(self):
        self.assertEqual(self.client.get(self.lesson_url(self.windows)).status_code, 200)
        self.client.force_login(self.student)
        response = self.client.get(self.lesson_url(self.hello))
        self.assertRedirects(response, f"/catalog/{self.course.slug}/")
        Enrollment.objects.create(user=self.student, course=self.course)
        self.assertContains(self.client.get(self.lesson_url(self.hello)), "Hello world")

    def test_attachment_download_is_protected(self):
        attachment = LessonAttachment.objects.create(
            lesson=self.hello, file=SimpleUploadedFile("zadanie.txt", b"Zrob to"), original_name="zadanie.txt", size=7
        )
        url = f"/learning/plik/{attachment.pk}/"
        self.client.force_login(self.student)
        self.assertEqual(self.client.get(url).status_code, 404)
        Enrollment.objects.create(user=self.student, course=self.course)
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertIn('filename="zadanie.txt"', response["Content-Disposition"])

    def test_completion_updates_progress_and_suggests_next(self):
        enrollment = Enrollment.objects.create(user=self.student, course=self.course)
        self.client.force_login(self.student)
        response = self.client.get(self.lesson_url(self.windows))
        self.assertContains(response, "Dalej: 1.2 macOS")
        self.client.post(self.lesson_url(self.windows) + "ukonczona/")
        enrollment.refresh_from_db()
        self.assertEqual(enrollment.progress_percent, 33)
        self.assertEqual(build_outline(self.course, self.student).suggested_next().lesson, self.mac)
        # Kolejność to tylko podpowiedź: można od razu otworzyć ostatnią lekcję
        self.assertEqual(self.client.get(self.lesson_url(self.hello)).status_code, 200)
        self.client.post(self.lesson_url(self.windows) + "ukonczona/")
        self.assertFalse(LessonProgress.objects.exists())

    def test_finishing_all_lessons_completes_enrollment(self):
        enrollment = Enrollment.objects.create(user=self.student, course=self.course)
        self.client.force_login(self.student)
        for lesson in (self.windows, self.mac, self.hello):
            self.client.post(self.lesson_url(lesson) + "ukonczona/")
        enrollment.refresh_from_db()
        self.assertEqual(enrollment.progress_percent, 100)
        self.assertTrue(enrollment.is_completed)

    def test_course_page_shows_outline_with_locks(self):
        response = self.client.get(f"/catalog/{self.course.slug}/")
        self.assertContains(response, "Pierwsze linie kodu")
        self.assertContains(response, f'href="{self.lesson_url(self.windows)}"')
        self.assertNotContains(response, f'href="{self.lesson_url(self.hello)}"')
        self.assertContains(response, "po zapisie")


class EditorImageUploadTests(CurriculumTestCase):
    def image(self):
        from PIL import Image

        buffer = BytesIO()
        Image.new("RGB", (20, 20), "teal").save(buffer, "PNG")
        return SimpleUploadedFile("zrzut.png", buffer.getvalue(), content_type="image/png")

    def test_student_without_courses_cannot_upload(self):
        self.client.force_login(self.outsider)
        response = self.client.post("/ckeditor5/image_upload/", {"upload": self.image()})
        self.assertEqual(response.status_code, 403)

    def test_teacher_can_upload_valid_image_only(self):
        self.client.force_login(self.teacher)
        ok = self.client.post("/ckeditor5/image_upload/", {"upload": self.image()})
        self.assertEqual(ok.status_code, 200)
        self.assertIn("url", ok.json())
        bad = self.client.post("/ckeditor5/image_upload/", {"upload": SimpleUploadedFile("x.png", b"nie obraz")})
        self.assertEqual(bad.status_code, 400)


class VideoUrlTests(TestCase):
    def test_supported_links(self):
        cases = {
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ": "https://www.youtube-nocookie.com/embed/dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ": "https://www.youtube-nocookie.com/embed/dQw4w9WgXcQ",
            "https://vimeo.com/123456789": "https://player.vimeo.com/video/123456789",
            "https://example.com/film.mp4": None,
            "javascript:alert(1)": None,
        }
        for url, expected in cases.items():
            with self.subTest(url=url):
                self.assertEqual(video_embed_url(url), expected)
