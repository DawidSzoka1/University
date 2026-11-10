"""Pole zdjęcia z podglądem (ImagePreviewInput) i awatar: walidacja + skalowanie Pillow."""

import shutil
import tempfile
from io import BytesIO

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase, TestCase, override_settings
from django.urls import reverse
from PIL import Image

from apps.core.widgets import ImagePreviewInput

MEDIA = tempfile.mkdtemp()


def png(size=(1200, 900), name="zdjecie.png"):
    buffer = BytesIO()
    Image.new("RGB", size, (40, 120, 200)).save(buffer, "PNG")
    return SimpleUploadedFile(name, buffer.getvalue(), content_type="image/png")


class FakeFile:
    """Wartość początkowa pola jak FieldFile zapisanego zdjęcia."""

    url = "/media/courses/okladka.jpg"

    def __str__(self):
        return "courses/okladka.jpg"


class ImagePreviewInputTests(SimpleTestCase):
    def test_empty_field_has_hints_for_preview_script(self):
        html = ImagePreviewInput(max_mb=5, max_size=(1280, 720)).render("thumbnail", None, {"id": "id_thumbnail"})
        self.assertIn('data-image-input', html)
        self.assertIn('accept="image/jpeg,image/png,image/webp"', html)
        self.assertIn('data-max-mb="5"', html)
        self.assertIn('data-max-size="1280x720"', html)
        self.assertNotIn("data-image-current", html)

    def test_current_image_is_shown_with_labelled_clear_checkbox(self):
        widget = ImagePreviewInput()
        widget.is_required = False
        html = widget.render("thumbnail", FakeFile(), {"id": "id_thumbnail"})
        self.assertIn('<img src="/media/courses/okladka.jpg"', html)
        self.assertIn("Obecne zdjęcie", html)
        self.assertIn('<label class="checkbox-label image-input__clear" for="thumbnail-clear_id">', html)
        self.assertIn('id="thumbnail-clear_id"', html)

    def test_gallery_edit_has_no_clear_option(self):
        html = ImagePreviewInput(allow_clear=False, show_current=False).render("image", FakeFile(), {"id": "id_image"})
        self.assertNotIn("thumbnail-clear", html)
        self.assertNotIn("image-clear", html)
        self.assertNotIn("data-image-current", html)


@override_settings(MEDIA_ROOT=MEDIA)
class AvatarUploadTests(TestCase):
    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        shutil.rmtree(MEDIA, ignore_errors=True)

    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username="ala", password="Haslo12345!", email="ala@example.com", first_name="Ala", last_name="Kot"
        )
        self.client.force_login(self.user)
        self.data = {"first_name": "Ala", "last_name": "Kot", "email": "ala@example.com"}

    def test_profile_form_shows_preview_widget(self):
        response = self.client.get(reverse("accounts:profile-edit"))
        self.assertContains(response, "data-image-input")
        self.assertContains(response, 'data-max-size="512x512"')
        self.assertContains(response, "js/image_preview.js")

    def test_avatar_is_resized(self):
        response = self.client.post(reverse("accounts:profile-edit"), {**self.data, "avatar": png((1200, 900))})
        self.assertRedirects(response, reverse("accounts:profile"))
        self.user.refresh_from_db()
        with Image.open(self.user.avatar.path) as image:
            self.assertEqual(image.size, (512, 384))
        page = self.client.get(reverse("accounts:profile-edit"))
        self.assertContains(page, "Obecne zdjęcie")

    def test_non_image_avatar_is_rejected(self):
        fake = SimpleUploadedFile("wirus.png", b"to nie obraz", content_type="image/png")
        response = self.client.post(reverse("accounts:profile-edit"), {**self.data, "avatar": fake})
        self.assertEqual(response.status_code, 200)
        self.user.refresh_from_db()
        self.assertFalse(self.user.avatar)
