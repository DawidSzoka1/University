"""Buduje dane demonstracyjne z ``apps/core/demo/data.py``.

    python manage.py seed_demo           # dopisuje brakujące dane (bezpieczne do powtarzania)
    python manage.py seed_demo --reset   # usuwa kursy i konta demo (z plikami) i tworzy je od nowa

Logika jest tutaj, dane w ``apps/core/demo/data.py``. Nowa funkcja platformy = nowa lista w
data.py + funkcja ``seed_<nazwa>`` poniżej, wywołana w ``handle``.
"""

from __future__ import annotations

import math
import re
import textwrap
import uuid
from decimal import Decimal
from io import BytesIO
from pathlib import Path

from django.contrib.auth import get_user_model
from django.core.files.base import ContentFile
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.files.storage import default_storage
from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils.text import slugify
from PIL import Image, ImageDraw, ImageFont

from apps.catalog.models import Category, Comment, CommentVote, Course, CourseSubmissionMessage, Enrollment
from apps.core.demo import data
from apps.core.models import ColorTheme, Notification
from apps.core.templatetags.core_tags import pl_plural
from apps.learning.curriculum import sync_enrollment_progress
from apps.learning.models import (
    Answer, Attempt, Choice, CourseSection, LearningStyle, Lesson, LessonAttachment, LessonProgress, LessonVariant,
    Question, Quiz,
)
from apps.learning.styles import style_for_attempt

User = get_user_model()
ASSETS = Path(__file__).resolve().parents[2] / "demo" / "assets"
FONTS = ASSETS / "fonts"  # fonty OFL z polskimi znakami (wbudowany font Pillow ich nie ma)


def font(size: int, display: bool = False) -> ImageFont.FreeTypeFont:
    if display:
        face = ImageFont.truetype(str(FONTS / "BricolageGrotesque.ttf"), size)
        face.set_variation_by_name("ExtraBold")
        return face
    return ImageFont.truetype(str(FONTS / "AtkinsonHyperlegible-Regular.ttf"), size)

# Kolory okładek wg kategorii głównej (dane obrazków, nie styl strony)
COVER_COLORS = {
    "Programowanie": ((180, 83, 9), (255, 251, 235)),
    "Data Science": ((15, 118, 110), (240, 253, 250)),
    "DevOps": ((30, 58, 138), (239, 246, 255)),
    "Projektowanie UX/UI": ((126, 34, 206), (250, 245, 255)),
    "Bezpieczeństwo IT": ((17, 24, 39), (249, 250, 251)),
}


class Command(BaseCommand):
    help = "Tworzy dane demonstracyjne: konta, kategorie, 10 kursów z programem, testy, zapisy, recenzje."

    def add_arguments(self, parser):
        parser.add_argument("--reset", action="store_true", help="Usuń kursy i konta demo, potem utwórz od nowa.")

    def handle(self, *args, reset: bool = False, **options):
        self.users: dict[str, object] = {}
        with transaction.atomic():
            if reset:
                self.reset()
            self.seed_users()
            self.seed_categories()
            self.seed_courses()
            self.seed_lesson_variants()
            self.seed_style_profiles()
            self.seed_quizzes()
            self.seed_enrollments()
            self.seed_quiz_attempts()
            self.seed_reviews()
            self.seed_notifications()
            self.seed_gallery()
            self.seed_chat()
            self.seed_orders()
        self.print_accounts()

    # ------------------------------------------------------------------ reset
    def reset(self) -> None:
        titles = [course["title"] for course in data.COURSES]
        from apps.orders.models import Order

        Order.objects.filter(items__course__title__in=titles).distinct().delete()
        courses = Course.objects.filter(title__in=titles)
        for course in courses:
            self._delete_course_files(course)
        deleted_courses = courses.count()
        courses.delete()
        from apps.gallery.models import GalleryItem, SliderPreset

        SliderPreset.objects.filter(name__in=[p[0] for p in data.SLIDER_PRESETS]).delete()
        for item in GalleryItem.objects.filter(title__in=[g[0] for g in data.GALLERY]):
            item.image.delete(save=False)
            item.delete()
        usernames = [u["username"] for u in data.TEACHERS + data.STUDENTS]
        from apps.orders.models import Order

        # Zamówienia chronią użytkownika i kurs (PROTECT) — usuwamy je przed kontami i kursami
        Order.objects.filter(user__username__in=usernames).delete()
        deleted_users, _ = User.objects.filter(username__in=usernames).delete()
        self.stdout.write(self.style.WARNING(
            f"Reset: usunięto {deleted_courses} {pl_plural(deleted_courses, 'kurs,kursy,kursów')} i konta demo."
        ))

    def _delete_course_files(self, course: Course) -> None:
        if course.thumbnail:
            course.thumbnail.delete(save=False)
        for lesson in Lesson.objects.filter(section__course=course):
            if lesson.video_file:
                lesson.video_file.delete(save=False)
            for attachment in lesson.attachments.all():
                attachment.file.delete(save=False)
            for src in re.findall(r'src="/media/(lessons/images/[^"]+)"', lesson.content_html):
                default_storage.delete(src)

    # ------------------------------------------------------------------ konta
    def seed_users(self) -> None:
        themes = {theme.name: theme for theme in ColorTheme.objects.all()}
        styles = {style.slug: style for style in LearningStyle.objects.all()}

        admin, created = User.objects.get_or_create(username=data.ADMIN["username"])
        for field in ("email", "first_name", "last_name"):
            setattr(admin, field, data.ADMIN[field])
        admin.role = User.Role.ADMIN
        admin.is_staff = admin.is_superuser = True
        if created:
            admin.set_password(data.ADMIN_PASSWORD)
        admin.save()
        self.users[admin.username] = admin

        for role, people in ((User.Role.MODERATOR, data.TEACHERS), (User.Role.STUDENT, data.STUDENTS)):
            for person in people:
                user, created = User.objects.get_or_create(username=person["username"])
                user.first_name = person["first_name"]
                user.last_name = person["last_name"]
                user.email = person["email"]
                user.role = role
                user.city = person.get("city", "")
                user.street_address = person.get("street", "")
                user.postal_code = person.get("postal", "")
                user.phone_number = person.get("phone", "")
                user.selected_theme = themes.get(person.get("theme"))
                user.learning_style = styles.get(data.STUDENT_STYLES.get(user.username))
                if created:
                    user.set_password(data.PASSWORD)
                user.save()
                self.users[user.username] = user
        self.stdout.write(f"✔ Konta: 1 admin, {len(data.TEACHERS)} nauczycieli, {len(data.STUDENTS)} uczniów")

    # ------------------------------------------------------------- kategorie
    def seed_categories(self) -> None:
        self.categories: dict[str, Category] = {}
        self.root_of: dict[str, str] = {}

        def upsert(spec, parent=None, root=None):
            category, _ = Category.objects.get_or_create(name=spec["name"], defaults={"parent": parent})
            if category.parent_id != (parent.pk if parent else None):
                category.parent = parent
                category.save()
            for username in spec.get("moderators", []):
                category.moderators.add(self.users[username])
            self.categories[category.name] = category
            self.root_of[category.name] = root or category.name
            for child in spec.get("children", []):
                upsert(child, category, root or category.name)

        for spec in data.CATEGORIES:
            upsert(spec)
        Category.objects.rebuild()
        self.categories = {c.name: c for c in Category.objects.filter(name__in=self.categories)}
        self.stdout.write(f"✔ Kategorie: {len(self.categories)} (z podkategoriami)")

    # ----------------------------------------------------------------- kursy
    def seed_courses(self) -> None:
        created_count = 0
        for spec in data.COURSES:
            status = spec.get("status", "approved")
            course, created = Course.objects.get_or_create(title=spec["title"])
            course.category = self.categories[spec["category"]]
            course.author = self.users[spec["author"]]
            course.description = spec["description"]
            course.description_html = spec["about"]
            course.price = Decimal(spec["price"])
            course.is_free = course.price == Decimal("0.00")
            course.difficulty_level = spec["level"]
            course.status = status
            course.is_published = status == "approved"
            course.rejection_reason = spec.get("rejection_reason", "")
            course.has_certificate = spec.get("certificate", True)
            minutes = sum(lesson["minutes"] for section in spec["sections"] for lesson in section["lessons"])
            course.duration_hours = max(1, math.ceil(minutes / 60))
            course.save()

            if not created:
                continue  # kurs już był — jego treść odtwarza tylko --reset
            created_count += 1
            self._cover(course, spec)
            self._curriculum(course, spec)
            for author, content in spec.get("messages", []):
                CourseSubmissionMessage.objects.create(course=course, author=self.users[author], content=content)
            course.recalculate_stats()
        total = Course.objects.filter(title__in=[c["title"] for c in data.COURSES])
        published = total.filter(status="approved").count()
        submitted = total.filter(status__in=["pending", "rejected"]).count()
        drafts = total.filter(status="draft").count()
        self.stdout.write(
            f"✔ Kursy: {published} {pl_plural(published, 'opublikowany,opublikowane,opublikowanych')}, "
            f"{submitted} {pl_plural(submitted, 'zgłoszenie,zgłoszenia,zgłoszeń')}, "
            f"{drafts} {pl_plural(drafts, 'szkic,szkice,szkiców')} (z nową treścią: {created_count})"
        )
        if created_count == 0:
            self.stdout.write("  Program istniejących kursów pominięty — pełne odtworzenie: seed_demo --reset")

    def _cover(self, course: Course, spec: dict) -> None:
        accent, paper = COVER_COLORS.get(self.root_of.get(spec["category"]), ((180, 83, 9), (255, 251, 235)))
        image = Image.new("RGB", (1280, 720), paper)
        draw = ImageDraw.Draw(image)
        for y in range(0, 720, 48):  # linie zeszytu
            draw.line([(0, y), (1280, y)], fill=tuple(int(c * 0.9) for c in paper), width=2)
        draw.rectangle([0, 0, 1280, 120], fill=accent)
        draw.text((64, 32), spec["category"].upper(), font=font(44), fill=paper)
        y = 200
        for line in textwrap.wrap(course.title, width=20):
            draw.text((64, y), line, font=font(92, display=True), fill=(17, 24, 39))
            y += 110
        buffer = BytesIO()
        image.save(buffer, "PNG", optimize=True)
        course.thumbnail.save(f"{course.slug}.png", ContentFile(buffer.getvalue()), save=False)
        course.save(update_fields=["thumbnail"])

    def _lesson_image(self, lesson_title: str, image_spec) -> str:
        alt, caption, lines = image_spec
        image = Image.new("RGB", (960, 540), (17, 24, 39))
        draw = ImageDraw.Draw(image)
        draw.text((48, 40), lesson_title, font=font(36, display=True), fill=(217, 119, 6))
        y = 130
        for line in lines:
            draw.text((48, y), line, font=font(40), fill=(255, 251, 235))
            y += 70
        buffer = BytesIO()
        image.save(buffer, "PNG", optimize=True)
        name = default_storage.save(f"lessons/images/{uuid.uuid4().hex}/{slugify(lesson_title)}.png", ContentFile(buffer.getvalue()))
        return (
            f'<figure class="image"><img src="{default_storage.url(name)}" alt="{alt}" width="960" height="540">'
            f"<figcaption>{caption}</figcaption></figure>"
        )

    def _curriculum(self, course: Course, spec: dict) -> None:
        for s_order, section_spec in enumerate(spec["sections"], start=1):
            section = CourseSection.objects.create(course=course, title=section_spec["title"], order=s_order)
            for l_order, lesson_spec in enumerate(section_spec["lessons"], start=1):
                html = lesson_spec["html"]
                if lesson_spec["image"]:
                    html += self._lesson_image(lesson_spec["title"], lesson_spec["image"])
                lesson = Lesson(
                    section=section, title=lesson_spec["title"], order=l_order,
                    duration_minutes=lesson_spec["minutes"], is_preview=lesson_spec["preview"], content_html=html,
                )
                video = lesson_spec["video"] or ""
                if video.startswith("youtube:"):
                    lesson.video_url = f"https://www.youtube.com/watch?v={video.split(':', 1)[1]}"
                elif video.startswith("file:"):
                    name = video.split(":", 1)[1]
                    lesson.video_file.save(name, ContentFile((ASSETS / name).read_bytes()), save=False)
                lesson.save()
                for filename, content in lesson_spec["files"]:
                    payload = content.encode("utf-8")
                    attachment = LessonAttachment(lesson=lesson, original_name=filename, size=len(payload))
                    attachment.file.save(filename, ContentFile(payload), save=True)

    # ------------------------------------------------- style uczenia się
    def seed_lesson_variants(self) -> None:
        """Wersje lekcji dla stylów (kurs Pythona). Lekcja z wersjami ma wyłączoną „tę samą treść”."""
        styles = {style.slug: style for style in LearningStyle.objects.all()}
        made = 0
        for course_title, lesson_title, slug, html, video in data.LESSON_VARIANTS:
            lesson = Lesson.objects.filter(section__course__title=course_title, title=lesson_title).first()
            if lesson is None or slug not in styles:
                continue
            _, created = LessonVariant.objects.update_or_create(
                lesson=lesson, style=styles[slug], defaults={"content_html": html, "video_url": video}
            )
            made += int(created)
            if lesson.same_for_all_styles:
                lesson.same_for_all_styles = False
                lesson.save(update_fields=["same_for_all_styles"])
        self.stdout.write(f"✔ Wersje lekcji dla stylów: {LessonVariant.objects.count()} (nowe: {made})")

    def seed_style_profiles(self) -> None:
        """Style polecane przez autorów kursów i wypełnione ankiety (raz na ucznia)."""
        from apps.learning.models import SurveyQuestion, SurveyResponse
        from apps.learning.styles import submit_survey

        styles = {style.slug: style for style in LearningStyle.objects.all()}
        for title, slugs in data.COURSE_STYLES.items():
            course = Course.objects.filter(title=title).first()
            if course:
                course.recommended_styles.set([styles[s] for s in slugs if s in styles])
        questions = list(SurveyQuestion.objects.filter(is_active=True).prefetch_related("options__style"))
        made = 0
        for username, answers in data.SURVEYS.items():
            user = self.users.get(username)
            if user is None or SurveyResponse.objects.filter(user=user).exists():
                continue
            picked = {}
            for question, slug in zip(questions, answers):
                option = next((o for o in question.options.all() if o.style.slug == slug), None)
                if option:
                    picked[question] = option
            if picked:
                submit_survey(user, picked)
                made += 1
        self.stdout.write(f"✔ Style w kursach: {len(data.COURSE_STYLES)}, ankiety stylu: {SurveyResponse.objects.count()} (nowe: {made})")

    def _apply_progress_styles(self) -> None:
        styles = {style.slug: style for style in LearningStyle.objects.all()}
        for username, course_title, slug in data.PROGRESS_STYLES:
            LessonProgress.objects.filter(
                user__username=username, lesson__section__course__title=course_title
            ).update(style=styles.get(slug))

    # ----------------------------------------------------------------- testy
    def seed_quizzes(self) -> None:
        """Testy kursów: końcowy („quiz”) i śródkursowe („quizzes”, po wskazanej lekcji).
        Dopisuje brakujące testy także do istniejących kursów (po tytule)."""
        created = 0
        for spec in data.COURSES:
            course = Course.objects.get(title=spec["title"])
            quizzes = ([spec["quiz"]] if spec.get("quiz") else []) + spec.get("quizzes", [])
            for quiz_spec in quizzes:
                created += self._quiz(course, quiz_spec)
        self.stdout.write(f"✔ Testy: {Quiz.objects.filter(course__title__in=[c['title'] for c in data.COURSES]).count()} (nowe: {created})")

    def _quiz(self, course: Course, spec: dict) -> int:
        after = None
        if spec.get("after"):
            after = Lesson.objects.filter(section__course=course, title=spec["after"]).first()
        quiz, created = Quiz.objects.get_or_create(
            course=course, title=spec["title"],
            defaults={
                "after_lesson": after,
                "is_required": spec.get("required", False),
                "pass_percent": spec.get("pass", 60),
                "max_attempts": spec.get("attempts", 0),
                "description": spec.get("description", ""),
            },
        )
        if not created:
            return 0
        for q_order, question_spec in enumerate(spec["questions"], start=1):
            question = Question.objects.create(
                quiz=quiz, question_type=question_spec["type"], text=question_spec["text"], order=q_order,
                points=question_spec.get("points", 1), answer_format=question_spec.get("format", "text"),
                code=question_spec.get("code", ""),
            )
            for c_order, (text, correct) in enumerate(question_spec["choices"], start=1):
                Choice.objects.create(question=question, text=text, is_correct=correct, order=c_order)
        return 1

    def seed_quiz_attempts(self) -> None:
        """Zaliczone testy po ukończonych lekcjach (i końcowe przy całym kursie) + wyjątki z data.py."""
        from apps.learning.curriculum import sync_enrollment_progress

        special = {(u, c, q): "pending" for u, c, q in data.PENDING_ATTEMPTS}
        special.update({(u, c, q): "failed" for u, c, q in data.FAILED_ATTEMPTS})
        made = 0
        for username, title, done in data.ENROLLMENTS:
            user, course = self.users[username], Course.objects.get(title=title)
            lessons = list(Lesson.objects.filter(section__course=course).order_by("section__order", "order"))
            finished = lessons if done == "all" else lessons[:done]
            finished_ids = {lesson.pk for lesson in finished}
            whole_course = len(finished) == len(lessons)
            for quiz in course.quizzes.prefetch_related("questions__choices"):
                mode = special.get((username, title, quiz.title))
                reached = quiz.after_lesson_id in finished_ids if quiz.after_lesson_id else whole_course
                if mode is None and not reached:
                    continue
                if quiz.attempts.filter(user=user).exists():
                    continue
                self._attempt(quiz, user, mode or "passed")
                made += 1
            sync_enrollment_progress(user, course)
        for username, title, quiz_title in data.FAILED_ATTEMPTS:
            quiz = Quiz.objects.filter(course__title=title, title=quiz_title).first()
            if quiz and not quiz.attempts.filter(user=self.users[username]).exists():
                self._attempt(quiz, self.users[username], "failed")
                made += 1
        # Starsze bazy demo: podejścia bez stylu dostają styl z wersji przerabianych lekcji
        for attempt in Attempt.objects.filter(user__username__in=self.users, learning_style__isnull=True).select_related("quiz__course"):
            style = style_for_attempt(attempt.user, attempt.quiz)
            if style is not None:
                attempt.learning_style = style
                attempt.save(update_fields=["learning_style"])
        from apps.learning.models import Certificate

        self.stdout.write(f"✔ Podejścia do testów: {made} nowych, dyplomy: {Certificate.objects.filter(user__username__in=self.users).count()}")

    def _attempt(self, quiz: Quiz, user, mode: str) -> None:
        """passed — wszystko dobrze; failed — błędne odpowiedzi ABC; pending — otwarte czekają na ocenę."""
        from django.utils import timezone

        attempt = Attempt.objects.create(
            quiz=quiz, user=user, max_score=quiz.max_score, learning_style=style_for_attempt(user, quiz)
        )
        for question in quiz.questions.all():
            answer = Answer.objects.create(attempt=attempt, question=question)
            if question.is_open:
                answer.text_answer = (
                    "def wieksza(a, b):\n    return a if a > b else b" if question.answer_format == "code"
                    else "Odpowiedź demonstracyjna ucznia."
                )
                if mode == "passed":
                    answer.points = question.points
                    answer.feedback = "Dobrze, dziękuję."
                answer.save()
                continue
            correct = [c for c in question.choices.all() if c.is_correct]
            wrong = [c for c in question.choices.all() if not c.is_correct]
            picked = wrong[:1] if mode == "failed" and wrong else correct
            answer.selected_choices.set(picked)
            answer.points = question.points if picked == correct else 0
            answer.save()
        attempt.score = sum(a.points or 0 for a in attempt.answers.all())
        if mode == "pending" and quiz.has_open_questions:
            attempt.status = Attempt.Status.PENDING_REVIEW
        else:
            attempt.status = Attempt.Status.GRADED
            attempt.passed = attempt.percent >= quiz.pass_percent
            attempt.graded_at = timezone.now()
            attempt.graded_by = quiz.course.author if quiz.has_open_questions else None
        attempt.save()

    # --------------------------------------------------- zapisy i postęp
    def seed_enrollments(self) -> None:
        for username, title, done in data.ENROLLMENTS:
            user, course = self.users[username], Course.objects.get(title=title)
            Enrollment.objects.get_or_create(user=user, course=course)
            lessons = list(Lesson.objects.filter(section__course=course).order_by("section__order", "order"))
            count = len(lessons) if done == "all" else done
            for lesson in lessons[:count]:
                LessonProgress.objects.get_or_create(user=user, lesson=lesson)
            sync_enrollment_progress(user, course)
        self._apply_progress_styles()
        self.stdout.write(f"✔ Zapisy na kursy z postępem: {len(data.ENROLLMENTS)}")

    # -------------------------------------------------- recenzje i głosy
    def seed_reviews(self) -> None:
        for username, title, rating, text, approved in data.REVIEWS:
            course = Course.objects.get(title=title)
            comment, _ = Comment.objects.get_or_create(
                course=course, author=self.users[username], content=text, parent=None,
                defaults={"rating": rating, "is_approved": approved},
            )
            if approved and comment.approved_by_id is None:
                comment.approved_by = course.author
                comment.save()
        for voter, author, title, value in data.REVIEW_VOTES:
            comment = Comment.objects.filter(course__title=title, author=self.users[author], parent=None).first()
            if comment:
                CommentVote.objects.update_or_create(comment=comment, voter=self.users[voter], defaults={"value": value})

        replies = 0
        for title, root_author, thread in data.COMMENT_THREADS:
            root = Comment.objects.filter(course__title=title, author=self.users[root_author], parent=None).first()
            if root:
                replies += self._replies(root, thread)
        for voter, title, author, prefix, value in data.REPLY_VOTES:
            comment = Comment.objects.filter(
                course__title=title, author=self.users[author], parent__isnull=False, content__startswith=prefix
            ).first()
            if comment:
                CommentVote.objects.update_or_create(comment=comment, voter=self.users[voter], defaults={"value": value})
        self.stdout.write(f"✔ Odpowiedzi w wątkach: {replies}, głosy na odpowiedzi: {len(data.REPLY_VOTES)}")
        ranked = User.objects.filter(username__in=self.users).exclude(rank_title="Nowicjusz").values_list("username", "rank_title")
        self.stdout.write(f"✔ Recenzje: {len(data.REVIEWS)}, głosy: {len(data.REVIEW_VOTES)}, rangi: {dict(ranked)}")

    def _replies(self, parent: Comment, thread: list) -> int:
        count = 0
        for username, text, children in thread:
            reply, _ = Comment.objects.get_or_create(
                course=parent.course, author=self.users[username], content=text, parent=parent,
                defaults={"is_approved": True, "approved_by": parent.course.author},
            )
            count += 1 + self._replies(reply, children)
        return count

    # ---------------------------------------------------------------- galeria
    def seed_gallery(self) -> None:
        from apps.gallery.models import GalleryItem, SliderPreset, SliderPresetItem

        admin = User.objects.get(username="admin")
        items = {}
        for title, description, alt, course_title, background, accent in data.GALLERY:
            item = GalleryItem.objects.filter(title=title).first()
            if item is None:
                image = Image.new("RGB", (2400, 1600), background)
                draw = ImageDraw.Draw(image)
                for y in range(0, 1600, 96):  # linie zeszytu
                    draw.line([(0, y), (2400, y)], fill=tuple(min(255, c + 18) for c in background), width=3)
                draw.rectangle([0, 1360, 2400, 1600], fill=accent)
                y = 220
                for line in textwrap.wrap(title, width=16):
                    draw.text((140, y), line, font=font(190, display=True), fill=accent)
                    y += 220
                draw.text((140, 1420), "E-Learning · galeria", font=font(90), fill=background)
                buffer = BytesIO()
                image.save(buffer, "PNG")
                # Jak przy zwykłym uploadzie: nowy plik → model zapisuje rozdzielczość i skaluje (Pillow)
                item = GalleryItem(
                    title=title, description=description, image_alt=alt, uploaded_by=admin,
                    related_course=Course.objects.filter(title=course_title).first() if course_title else None,
                    image=SimpleUploadedFile(f"{slugify(title)}.png", buffer.getvalue(), content_type="image/png"),
                )
                item.save()
            items[title] = item
        for name, active, autoplay, seconds, titles in data.SLIDER_PRESETS:
            preset, created = SliderPreset.objects.get_or_create(
                name=name, defaults={"autoplay": autoplay, "interval_seconds": seconds, "created_by": admin}
            )
            if active:
                preset.is_active = True
                preset.save()
            if created:
                for order, title in enumerate(titles, start=1):
                    SliderPresetItem.objects.create(preset=preset, gallery_item=items[title], order=order)
        self.stdout.write(f"✔ Galeria: {len(items)} zdjęć, układy: {len(data.SLIDER_PRESETS)} (wyświetlany: {SliderPreset.get_active()})")

    # --------------------------------------------------------- powiadomienia
    def seed_notifications(self) -> None:
        for username, kind, title, message, is_read in data.NOTIFICATIONS:
            Notification.objects.get_or_create(
                user=self.users[username], title=title,
                defaults={"kind": kind, "message": message, "is_read": is_read},
            )
        self.stdout.write(f"✔ Powiadomienia: {len(data.NOTIFICATIONS)}")

    # ----------------------------------------------------------- asystent AI
    def seed_chat(self) -> None:
        from apps.chatbot.models import ChatMessage, ChatSession

        made = 0
        for username, course_title, messages in data.CHAT_CONVERSATIONS:
            course = Course.objects.get(title=course_title) if course_title else None
            session, created = ChatSession.objects.get_or_create(student=self.users[username], course=course)
            if not created and session.messages.exists():
                continue
            lessons = {lesson.title: lesson for lesson in Lesson.objects.filter(section__course=course)} if course else {}
            for sender, text, lesson_title, refusal in messages:
                ChatMessage.objects.create(
                    session=session, sender=sender, message_text=text,
                    lesson=lessons.get(lesson_title), is_refusal=refusal,
                )
            made += 1
        self.stdout.write(f"✔ Rozmowy z asystentem: {made} nowych")

    # ------------------------------------------------------------------ sklep
    def seed_orders(self) -> None:
        from datetime import timedelta

        from django.utils import timezone

        from apps.orders.models import AccessCode, Cart, CartItem, Order, OrderItem, OrderStatusChange

        made = 0
        for index, (username, status, lines, note) in enumerate(data.ORDERS):
            user = self.users[username]
            first = Course.objects.get(title=lines[0][0])
            if Order.objects.filter(user=user, status=status, items__course=first).exists():
                continue
            created = timezone.now() - timedelta(days=len(data.ORDERS) - index, hours=3)
            order = Order.objects.create(
                number=Order.next_number(), user=user, status=status,
                email=user.email, first_name=user.first_name, last_name=user.last_name,
                street_address=user.street_address, postal_code=user.postal_code, city=user.city,
                phone_number=user.phone_number,
                payment_provider="manual" if "ręcznie" in note else ("stripe" if status != "cancelled" else ""),
                payment_reference=f"cs_test_demo_{index}" if "Stripe" in note else "",
            )
            total = Decimal("0")
            for title, quantity in lines:
                course = Course.objects.get(title=title)
                item = OrderItem.objects.create(
                    order=order, course=course, title=course.title, quantity=quantity,
                    unit_price=course.price, line_total=course.price * quantity,
                )
                total += item.line_total
                if status in ("paid", "completed"):
                    for _ in range(quantity - 1):  # zapis kupującego jest w ENROLLMENTS
                        AccessCode.objects.create(order_item=item)
            history = [("pending_payment", "Zamówienie złożone — oczekuje na płatność.")]
            if status in ("paid", "completed"):
                history.append(("paid", note))
            if status == "completed":
                history.append(("completed", "Zrealizowane."))
            if status == "cancelled":
                history.append(("cancelled", note))
            for step, (step_status, step_note) in enumerate(history):
                change = OrderStatusChange.objects.create(order=order, status=step_status, note=step_note, changed_by=user)
                OrderStatusChange.objects.filter(pk=change.pk).update(created_at=created + timedelta(minutes=5 * step))
            paid = created + timedelta(minutes=5) if status in ("paid", "completed") else None
            Order.objects.filter(pk=order.pk).update(total=total, created_at=created, paid_at=paid, fulfilled_at=paid)
            made += 1
        for username, lines in data.CARTS:
            cart, _ = Cart.objects.get_or_create(user=self.users[username])
            for title, quantity in lines:
                CartItem.objects.get_or_create(cart=cart, course=Course.objects.get(title=title), defaults={"quantity": quantity})
        self.stdout.write(f"✔ Zamówienia: {made} nowych, koszyki: {len(data.CARTS)}")

    # -------------------------------------------------------------- podsumowanie
    def print_accounts(self) -> None:
        self.stdout.write(self.style.SUCCESS("\nKonta testowe:"))
        self.stdout.write(f"  admin / {data.ADMIN_PASSWORD}  (administrator, panel /admin/)")
        for person in data.TEACHERS:
            self.stdout.write(f"  {person['username']} / {data.PASSWORD}  (nauczyciel)")
        for person in data.STUDENTS:
            self.stdout.write(f"  {person['username']} / {data.PASSWORD}  (uczeń)")
