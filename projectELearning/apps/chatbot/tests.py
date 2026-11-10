"""Asystent AI: JWT, kontekst lekcji, blokada w trakcie testu, wyjaśnianie po teście, historia."""

import json
from datetime import timedelta
from unittest import mock

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone
from rest_framework_simplejwt.tokens import AccessToken

from apps.catalog.models import Category, Course, Enrollment
from apps.chatbot import llm
from apps.chatbot.formatting import render_reply
from apps.chatbot.models import ChatMessage, ChatSession
from apps.chatbot.services import ChatContext
from apps.learning.models import Choice, CourseSection, Lesson, Question, Quiz, QuizSitting

User = get_user_model()


def fake_stream(*pieces):
    def stream(messages):
        stream.prompt = messages
        yield from pieces
    stream.prompt = None
    return stream


def failing_stream(messages):
    raise llm.AssistantUnavailable("Asystent jest chwilowo niedostępny — serwer modelu (Ollama) nie odpowiada.")
    yield  # pragma: no cover


class ChatFixture(TestCase):
    def setUp(self):
        self.teacher = User.objects.create_user(username="nauczyciel", password="Haslo12345!", role="moderator")
        self.student = User.objects.create_user(username="uczen", password="Haslo12345!")
        self.other = User.objects.create_user(username="inny", password="Haslo12345!")
        category = Category.objects.create(name="Python")
        self.course = Course.objects.create(
            title="Python od zera", category=category, author=self.teacher,
            status=Course.Status.APPROVED, is_published=True,
        )
        section = CourseSection.objects.create(course=self.course, title="Start", order=1)
        self.lesson = Lesson.objects.create(
            section=section, title="Pętle", order=1,
            content_html="<p>Pętla <code>for</code> przechodzi po elementach listy.</p>",
        )
        self.paid_lesson = Lesson.objects.create(section=section, title="Funkcje", order=2, content_html="<p>def</p>")
        Enrollment.objects.create(user=self.student, course=self.course)
        self.quiz = Quiz.objects.create(course=self.course, title="Sprawdzian z pętli", max_attempts=2)
        question = Question.objects.create(quiz=self.quiz, question_type="closed_single", text="Ile razy wykona się range(3)?", points=1, order=1)
        self.right = Choice.objects.create(question=question, text="3", is_correct=True, order=1)
        self.wrong = Choice.objects.create(question=question, text="4", is_correct=False, order=2)
        self.question = question

    def auth(self, user=None):
        return {"HTTP_AUTHORIZATION": f"Bearer {AccessToken.for_user(user or self.student)}"}

    def ask(self, text="Co to jest pętla?", user=None, **context):
        return self.client.post(
            reverse("chatbot:api_send_message"),
            data=json.dumps({"message": text, **context}),
            content_type="application/json",
            **self.auth(user),
        )

    @staticmethod
    def events(response):
        body = b"".join(response.streaming_content).decode()
        return [json.loads(line) for line in body.splitlines() if line.strip()]

    def quiz_url(self):
        return reverse("learning:quiz_detail", kwargs={"slug": self.course.slug, "pk": self.quiz.pk})


class JwtProtectionTests(ChatFixture):
    def test_token_endpoint_needs_login_and_post(self):
        url = reverse("chatbot:api_token")
        self.assertEqual(self.client.post(url).status_code, 302)
        self.client.force_login(self.student)
        self.assertEqual(self.client.get(url).status_code, 405)
        data = self.client.post(url).json()
        self.assertTrue(data["access"])
        self.assertGreater(data["expires_in"], 0)

    def test_api_rejects_session_without_token(self):
        self.client.force_login(self.student)
        response = self.client.get(reverse("chatbot:api_history"))
        self.assertEqual(response.status_code, 401)
        response = self.client.post(
            reverse("chatbot:api_send_message"), data=json.dumps({"message": "x"}), content_type="application/json"
        )
        self.assertEqual(response.status_code, 401)

    def test_api_rejects_invalid_token(self):
        response = self.client.get(reverse("chatbot:api_history"), HTTP_AUTHORIZATION="Bearer zly-token")
        self.assertEqual(response.status_code, 401)


@override_settings(CHATBOT_QUIZ_LOCK_MINUTES=60)
class ConversationTests(ChatFixture):
    def test_reply_is_streamed_and_both_messages_saved_with_lesson(self):
        stream = fake_stream("Pętla ", "**for** powtarza kod.")
        with mock.patch.object(llm, "stream_reply", stream):
            events = self.events(self.ask(course=self.course.pk, lesson=self.lesson.pk))
        self.assertEqual([e["type"] for e in events], ["start", "delta", "delta", "done"])
        self.assertIn("<strong>for</strong>", events[-1]["message"]["html"])
        session = ChatSession.objects.get(student=self.student, course=self.course)
        texts = list(session.messages.values_list("sender", "message_text", "lesson"))
        self.assertEqual(texts, [
            ("user", "Co to jest pętla?", self.lesson.pk),
            ("bot", "Pętla **for** powtarza kod.", self.lesson.pk),
        ])
        system = stream.prompt[0]["content"]
        self.assertIn("Python od zera", system)
        self.assertIn("przechodzi po elementach listy", system)
        self.assertNotIn("<code>", system)

    def test_history_goes_to_model_and_widget(self):
        with mock.patch.object(llm, "stream_reply", fake_stream("Pierwsza odpowiedź.")):
            self.events(self.ask("Pierwsze pytanie", course=self.course.pk))
        stream = fake_stream("Druga.")
        with mock.patch.object(llm, "stream_reply", stream):
            self.events(self.ask("Drugie pytanie", course=self.course.pk))
        roles = [(m["role"], m["content"]) for m in stream.prompt[1:]]
        self.assertEqual(roles, [
            ("user", "Pierwsze pytanie"), ("assistant", "Pierwsza odpowiedź."), ("user", "Drugie pytanie"),
        ])
        data = self.client.get(reverse("chatbot:api_history"), {"course": self.course.pk}, **self.auth()).json()
        self.assertEqual(len(data["messages"]), 4)
        self.assertIn("Python od zera", data["context"]["label"])
        self.assertIsNone(data["lock"])

    def test_conversations_are_separate_per_user_and_course(self):
        with mock.patch.object(llm, "stream_reply", fake_stream("Ok.")):
            self.events(self.ask("Ogólne"))
            self.events(self.ask("Kursowe", course=self.course.pk))
        self.assertEqual(ChatSession.objects.filter(student=self.student).count(), 2)
        other = self.client.get(reverse("chatbot:api_history"), {"course": self.course.pk}, **self.auth(self.other)).json()
        self.assertEqual(other["messages"], [])
        session = ChatSession.objects.get(student=self.student, course=self.course)
        self.client.force_login(self.other)
        self.assertEqual(self.client.get(reverse("chatbot:session_detail", args=[session.pk])).status_code, 404)
        self.assertEqual(self.client.post(reverse("chatbot:session_clear", args=[session.pk])).status_code, 404)

    def test_lesson_without_access_is_rejected(self):
        response = self.ask(lesson=self.lesson.pk, user=self.other)
        self.assertEqual(response.status_code, 404)
        self.assertFalse(ChatMessage.objects.exists())

    def test_unavailable_model_removes_question_and_reports_error(self):
        with mock.patch.object(llm, "stream_reply", failing_stream):
            events = self.events(self.ask(course=self.course.pk))
        self.assertEqual(events[-1]["type"], "error")
        self.assertIn("niedostępny", events[-1]["detail"])
        self.assertFalse(ChatMessage.objects.exists())

    def test_validation_and_busy_guard(self):
        self.assertEqual(self.ask("   ").status_code, 400)
        self.assertEqual(self.ask("x" * 2001).status_code, 400)
        session = ChatSession.objects.create(student=self.student)
        ChatMessage.objects.create(session=session, sender="user", message_text="czekam")
        self.assertEqual(self.ask("Kolejne").status_code, 429)

    def test_clear_removes_only_this_conversation(self):
        with mock.patch.object(llm, "stream_reply", fake_stream("Ok.")):
            self.events(self.ask("Ogólne"))
            self.events(self.ask("Kursowe", course=self.course.pk))
        response = self.client.post(
            reverse("chatbot:api_clear"), data=json.dumps({"course": self.course.pk}),
            content_type="application/json", **self.auth(),
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(list(ChatSession.objects.values_list("course", flat=True)), [None])


@override_settings(CHATBOT_QUIZ_LOCK_MINUTES=60)
class QuizLockTests(ChatFixture):
    def test_opening_quiz_blocks_assistant_until_submitted(self):
        self.client.force_login(self.student)
        self.client.get(self.quiz_url())
        self.assertTrue(QuizSitting.objects.filter(user=self.student, quiz=self.quiz).exists())

        stream = fake_stream("Nie powinno paść.")
        with mock.patch.object(llm, "stream_reply", stream):
            events = self.events(self.ask("Która odpowiedź jest dobra?", lesson=self.lesson.pk))
        self.assertIsNone(stream.prompt)  # model nie został zapytany
        self.assertEqual(events[0]["type"], "refusal")
        self.assertIn("Podczas testu nie mogę udzielać pomocy", events[0]["messages"][1]["text"])
        self.assertEqual(events[0]["lock"]["quiz_url"], self.quiz_url())
        history = self.client.get(reverse("chatbot:api_history"), **self.auth()).json()
        self.assertEqual(history["lock"]["quiz_title"], "Sprawdzian z pętli")

        # Odmowa nie trafia do kontekstu modelu po teście
        self.client.post(self.quiz_url(), {f"q{self.question.pk}": self.wrong.pk})
        self.assertFalse(QuizSitting.objects.exists())
        stream = fake_stream("Wyjaśniam.")
        with mock.patch.object(llm, "stream_reply", stream):
            events = self.events(self.ask("Teraz już mogę?", course=self.course.pk))
        self.assertEqual(events[-1]["type"], "done")
        self.assertNotIn("Która odpowiedź jest dobra?", [m["content"] for m in stream.prompt])

    def test_lock_expires(self):
        QuizSitting.objects.create(user=self.student, quiz=self.quiz, started_at=timezone.now() - timedelta(minutes=61))
        with mock.patch.object(llm, "stream_reply", fake_stream("Ok.")):
            events = self.events(self.ask())
        self.assertEqual(events[-1]["type"], "done")

    def test_editor_preview_does_not_lock(self):
        self.client.force_login(self.teacher)
        self.client.get(self.quiz_url())
        self.assertFalse(QuizSitting.objects.exists())

    def test_after_test_assistant_sees_answers_but_keeps_correct_one_secret_while_retry_left(self):
        self.client.force_login(self.student)
        self.client.get(self.quiz_url())
        self.client.post(self.quiz_url(), {f"q{self.question.pk}": self.wrong.pk})
        attempt = self.quiz.attempts.get()
        stream = fake_stream("Spójrz na range.")
        with mock.patch.object(llm, "stream_reply", stream):
            self.events(self.ask("Czemu źle?", attempt=attempt.pk))
        system = stream.prompt[0]["content"]
        self.assertIn("Ile razy wykona się range(3)?", system)
        self.assertIn("4 [zaznaczona przez ucznia]", system)
        self.assertNotIn("poprawna]", system)
        self.assertIn("NIE podawaj poprawnych odpowiedzi", system)

        # Ostatnie podejście wykorzystane — można omówić poprawne odpowiedzi
        self.client.get(self.quiz_url())
        self.client.post(self.quiz_url(), {f"q{self.question.pk}": self.wrong.pk})
        last = self.quiz.attempts.order_by("-pk").first()
        with mock.patch.object(llm, "stream_reply", stream):
            self.events(self.ask("A teraz?", attempt=last.pk))
        self.assertIn("3 [poprawna]", stream.prompt[0]["content"])

    def test_foreign_attempt_is_rejected(self):
        self.client.force_login(self.student)
        self.client.post(self.quiz_url(), {f"q{self.question.pk}": self.right.pk})
        attempt = self.quiz.attempts.get()
        self.assertEqual(self.ask(attempt=attempt.pk, user=self.other).status_code, 404)


class PagesTests(ChatFixture):
    def test_widget_only_for_logged_in_with_page_context(self):
        lesson_url = reverse("learning:lesson_detail", kwargs={"slug": self.course.slug, "pk": self.lesson.pk})
        self.assertNotContains(self.client.get(reverse("core:home")), "data-chat-widget")
        self.client.force_login(self.student)
        response = self.client.get(lesson_url)
        self.assertContains(response, "data-chat-widget")
        self.assertContains(response, f'data-lesson="{self.lesson.pk}"')
        self.assertContains(response, "Zapytaj asystenta o tę lekcję")
        self.assertContains(response, 'aria-controls="chat-panel"')
        self.assertContains(response, '<label for="chat-input"')
        self.assertContains(response, f"Model: {llm.model_label()}")

    def test_history_pages(self):
        self.client.force_login(self.student)
        response = self.client.get(reverse("chatbot:session_list"))
        self.assertContains(response, "Nie masz jeszcze rozmów")
        session = ChatSession.objects.create(student=self.student, course=self.course)
        for i in range(25):
            ChatMessage.objects.create(session=session, sender="user", message_text=f"pytanie {i}")
        ChatMessage.objects.create(session=session, sender="bot", message_text="<script>x</script> `kod`")
        response = self.client.get(reverse("chatbot:session_list"))
        self.assertContains(response, "Python od zera")
        response = self.client.get(reverse("chatbot:session_detail", args=[session.pk]))
        self.assertContains(response, "&lt;script&gt;")  # ostatnia strona — najnowsze wiadomości
        self.assertContains(response, "<code>kod</code>")
        self.assertNotContains(response, "<script>x")
        self.assertNotContains(response, "pytanie 0<")
        # Strona rozmowy: bez drugiej kopii czatu w rogu, boczna lista rozmów z aktywną pozycją
        self.assertNotContains(response, "data-chat-widget")
        self.assertContains(response, 'aria-current="page"')
        self.assertContains(response, 'placeholder="Twoje pytanie…"')
        self.assertContains(response, '<label for="id_message_text" class="sr-only">')

    def test_form_without_javascript(self):
        self.client.force_login(self.student)
        with mock.patch.object(llm, "stream_reply", fake_stream("Odpowiedź bez JS.")):
            response = self.client.post(reverse("chatbot:session_general"), {"message_text": "Pytanie"})
        session = ChatSession.objects.get(student=self.student, course=None)
        self.assertRedirects(response, reverse("chatbot:session_detail", args=[session.pk]) + "#najnowsza")
        self.assertEqual(session.messages.count(), 2)

        with mock.patch.object(llm, "stream_reply", failing_stream):
            response = self.client.post(reverse("chatbot:session_general"), {"message_text": "Drugie"})
        self.assertContains(response, "niedostępny")
        self.assertContains(response, "Drugie</textarea>")  # wpisane pytanie zostaje
        self.assertEqual(session.messages.count(), 2)

        response = self.client.post(reverse("chatbot:session_general"), {"message_text": " "})
        self.assertContains(response, 'aria-invalid="true"')

    def test_session_clear_is_post_only(self):
        self.client.force_login(self.student)
        session = ChatSession.objects.create(student=self.student)
        url = reverse("chatbot:session_clear", args=[session.pk])
        self.assertEqual(self.client.get(url).status_code, 405)
        self.assertRedirects(self.client.post(url), reverse("chatbot:session_list"))
        self.assertFalse(ChatSession.objects.exists())

    def test_pages_require_login(self):
        response = self.client.get(reverse("chatbot:session_list"))
        self.assertEqual(response.status_code, 302)


class FormattingTests(TestCase):
    def test_escapes_html_and_renders_basic_markdown(self):
        html = render_reply('<img src=x onerror=alert(1)>\n\n**Uwaga**: `x`\n\n- jeden\n- dwa\n\n```python\nprint("<b>")\n```')
        self.assertNotIn("<img", html)
        self.assertIn("&lt;img", html)
        self.assertIn("<strong>Uwaga</strong>", html)
        self.assertIn("<ul><li>jeden</li><li>dwa</li></ul>", html)
        self.assertIn('<pre data-language="python" tabindex="0"><code>print(&quot;&lt;b&gt;&quot;)</code></pre>', html)

    def test_model_label_is_readable(self):
        self.assertEqual(
            llm.model_label("hf.co/speakleash/Bielik-Minitron-7B-v3.0-Instruct-GGUF:Q4_K_M"),
            "Bielik-Minitron-7B-v3.0-Instruct (Q4_K_M)",
        )
        self.assertEqual(llm.model_label("bielik:latest"), "bielik")
        with override_settings(OLLAMA_MODEL_LABEL="Bielik 11B"):
            self.assertEqual(llm.model_label(), "Bielik 11B")

    def test_general_prompt_without_course(self):
        prompt = llm.system_prompt(ChatContext())
        self.assertIn("po polsku", prompt)
        self.assertIn("Nie rozwiązujesz za ucznia testów", prompt)
