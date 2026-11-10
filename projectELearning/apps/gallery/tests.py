"""Galeria: skalowanie przy dodawaniu, uprawnienia, układy, kolejność (przeciąganie i ↑↓), slajder."""

import io
import json
import shutil
import tempfile

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse
from PIL import Image

from apps.gallery import slider
from apps.gallery.models import GalleryItem, SliderPreset, SliderPresetItem

User = get_user_model()
MEDIA = tempfile.mkdtemp()


def tearDownModule():
    shutil.rmtree(MEDIA, ignore_errors=True)


def image_upload(size=(3000, 2000), name="zdjecie.jpg", fmt="JPEG"):
    buffer = io.BytesIO()
    Image.new("RGB", size, (120, 80, 40)).save(buffer, fmt)
    return SimpleUploadedFile(name, buffer.getvalue(), content_type=f"image/{fmt.lower()}")


@override_settings(MEDIA_ROOT=MEDIA)
class GalleryFixture(TestCase):
    def setUp(self):
        self.admin = User.objects.create_user(username="szef", password="Haslo12345!", role="admin")
        self.teacher = User.objects.create_user(username="nauczyciel", password="Haslo12345!", role="moderator")
        self.student = User.objects.create_user(username="uczen", password="Haslo12345!")

    def make_item(self, title, size=(800, 600)):
        return GalleryItem.objects.create(title=title, description=f"Opis {title}", image_alt=f"Alt {title}", image=image_upload(size, f"{title}.jpg"))

    def make_preset(self, name="Układ", titles=("A", "B", "C"), active=False):
        preset = SliderPreset.objects.create(name=name, is_active=active)
        for order, title in enumerate(titles, start=1):
            item = GalleryItem.objects.filter(title=title).first() or self.make_item(title)
            SliderPresetItem.objects.create(preset=preset, gallery_item=item, order=order)
        return preset

    def order_of(self, preset):
        return [entry.gallery_item.title for entry in slider.ordered_items(preset)]


class UploadTests(GalleryFixture):
    def test_large_image_is_scaled_on_upload(self):
        self.client.force_login(self.teacher)
        response = self.client.post(reverse("gallery:upload_image"), {
            "title": "Duże", "description": "Bardzo długi opis " * 30, "image_alt": "Pomarańczowe tło",
            "image": image_upload((4000, 3000)),
        }, follow=True)
        item = GalleryItem.objects.get()
        self.assertEqual((item.original_width, item.original_height), (4000, 3000))
        self.assertEqual((item.width, item.height), (1600, 1200))
        with Image.open(item.image.path) as stored:
            self.assertEqual(stored.size, (1600, 1200))
        self.assertContains(response, "przeskalowano z 4000×3000 do 1600×1200")
        self.assertEqual(item.uploaded_by, self.teacher)

    def test_small_image_keeps_size(self):
        item = self.make_item("Małe", (640, 480))
        self.assertEqual((item.width, item.height), (640, 480))
        self.assertFalse(item.was_resized)

    def test_validation_alt_and_file_type(self):
        self.client.force_login(self.admin)
        response = self.client.post(reverse("gallery:upload_image"), {
            "title": "Bez opisu", "description": "x", "image_alt": "",
            "image": SimpleUploadedFile("plik.txt", b"to nie obraz", content_type="text/plain"),
        })
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'aria-invalid="true"')
        self.assertFalse(GalleryItem.objects.exists())

    def test_students_cannot_upload_or_configure(self):
        self.client.force_login(self.student)
        self.assertEqual(self.client.get(reverse("gallery:upload_image")).status_code, 404)
        self.assertEqual(self.client.get(reverse("gallery:slider_configurator")).status_code, 404)
        self.client.force_login(self.teacher)
        self.assertEqual(self.client.get(reverse("gallery:slider_configurator")).status_code, 404)  # układy: admin

    def test_edit_and_delete_item(self):
        preset = self.make_preset(titles=("A", "B", "C"))
        item = GalleryItem.objects.get(title="B")
        self.client.force_login(self.admin)
        self.client.post(reverse("gallery:item_edit", kwargs={"pk": item.pk}), {"title": "B2", "description": "Nowy", "image_alt": "Alt"})
        item.refresh_from_db()
        self.assertEqual(item.title, "B2")
        self.assertTrue(item.image)  # bez nowego pliku zdjęcie zostaje
        delete_url = reverse("gallery:item_delete", kwargs={"pk": item.pk})
        self.assertEqual(self.client.get(delete_url).status_code, 405)
        self.client.post(delete_url)
        self.assertEqual(self.order_of(preset), ["A", "C"])
        self.assertEqual([e.order for e in slider.ordered_items(preset)], [1, 2])


class PresetTests(GalleryFixture):
    def setUp(self):
        super().setUp()
        self.client.force_login(self.admin)

    def test_create_activate_and_only_one_active(self):
        self.client.post(reverse("gallery:preset_create"), {"new-name": "Jesień", "new-autoplay": "on", "new-interval_seconds": 5})
        first = SliderPreset.objects.get(name="Jesień")
        self.assertTrue(first.is_active)  # pierwszy układ od razu wyświetlany
        self.client.post(reverse("gallery:preset_create"), {"new-name": "Programowanie", "new-interval_seconds": 8})
        second = SliderPreset.objects.get(name="Programowanie")
        self.assertFalse(second.is_active)
        self.client.post(reverse("gallery:preset_activate", kwargs={"pk": second.pk}))
        first.refresh_from_db()
        second.refresh_from_db()
        self.assertEqual((first.is_active, second.is_active), (False, True))

    def test_update_options_and_delete(self):
        preset = self.make_preset()
        self.client.post(reverse("gallery:preset_update", kwargs={"pk": preset.pk}), {"edit-name": "Nowa nazwa", "edit-interval_seconds": 10})
        preset.refresh_from_db()
        self.assertEqual((preset.name, preset.autoplay, preset.interval_seconds), ("Nowa nazwa", False, 10))
        self.client.post(reverse("gallery:preset_update", kwargs={"pk": preset.pk}), {"edit-name": "X", "edit-interval_seconds": 1})
        preset.refresh_from_db()
        self.assertEqual(preset.interval_seconds, 10)  # poniżej minimum — odrzucone
        self.client.post(reverse("gallery:preset_delete", kwargs={"pk": preset.pk}))
        self.assertFalse(SliderPreset.objects.exists())
        self.assertEqual(GalleryItem.objects.count(), 3)  # zdjęcia zostają w bazie

    def test_configurator_switches_edited_preset(self):
        self.make_preset("Pierwszy", ("A", "B"), active=True)
        second = self.make_preset("Drugi", ("C",))
        response = self.client.get(reverse("gallery:slider_configurator") + f"?uklad={second.pk}")
        self.assertContains(response, "Układ „Drugi”")
        self.assertContains(response, 'data-item-id="', count=3)  # 1 karta + 2 zdjęcia z bazy


class OrderTests(GalleryFixture):
    def setUp(self):
        super().setUp()
        self.client.force_login(self.admin)
        self.preset = self.make_preset(titles=("A", "B", "C", "D"))
        self.ids = {item.title: item.pk for item in GalleryItem.objects.all()}
        self.record_ids = sorted(SliderPresetItem.objects.values_list("pk", flat=True))

    def reorder(self, titles):
        return self.client.post(
            reverse("gallery:slider_reorder", kwargs={"pk": self.preset.pk}),
            data=json.dumps({"items": [self.ids[t] for t in titles]}),
            content_type="application/json",
            HTTP_ACCEPT="application/json",
        )

    def test_drag_and_drop_reorder_updates_order_without_recreating(self):
        response = self.reorder(["C", "A", "D", "B"])
        self.assertEqual(response.json(), {"ok": True})
        self.assertEqual(self.order_of(self.preset), ["C", "A", "D", "B"])
        self.assertEqual(sorted(SliderPresetItem.objects.values_list("pk", flat=True)), self.record_ids)

    def test_reorder_rejects_mismatched_list_and_get(self):
        self.assertEqual(self.reorder(["A", "B", "C"]).status_code, 409)
        self.assertEqual(self.reorder(["A", "A", "B", "C"]).status_code, 409)
        self.assertEqual(self.client.get(reverse("gallery:slider_reorder", kwargs={"pk": self.preset.pk})).status_code, 405)
        self.assertEqual(self.order_of(self.preset), ["A", "B", "C", "D"])

    def test_keyboard_alternative_up_down(self):
        url = reverse("gallery:preset_move_item", kwargs={"pk": self.preset.pk, "item_pk": self.ids["C"], "direction": "up"})
        self.client.post(url)
        self.assertEqual(self.order_of(self.preset), ["A", "C", "B", "D"])
        first_up = reverse("gallery:preset_move_item", kwargs={"pk": self.preset.pk, "item_pk": self.ids["A"], "direction": "up"})
        self.client.post(first_up)
        self.assertEqual(self.order_of(self.preset), ["A", "C", "B", "D"])

    def test_drag_from_library_inserts_at_position(self):
        new = self.make_item("E")
        response = self.client.post(
            reverse("gallery:preset_add_item", kwargs={"pk": self.preset.pk}),
            data=json.dumps({"item": new.pk, "position": 1}), content_type="application/json", HTTP_ACCEPT="application/json",
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.order_of(self.preset), ["A", "E", "B", "C", "D"])
        duplicate = self.client.post(
            reverse("gallery:preset_add_item", kwargs={"pk": self.preset.pk}),
            data=json.dumps({"item": new.pk}), content_type="application/json", HTTP_ACCEPT="application/json",
        )
        self.assertEqual(duplicate.status_code, 409)

    def test_add_button_and_remove(self):
        new = self.make_item("E")
        self.client.post(reverse("gallery:preset_add_item", kwargs={"pk": self.preset.pk}), {"item": new.pk})
        self.assertEqual(self.order_of(self.preset)[-1], "E")
        self.client.post(reverse("gallery:preset_remove_item", kwargs={"pk": self.preset.pk, "item_pk": self.ids["B"]}))
        self.assertEqual(self.order_of(self.preset), ["A", "C", "D", "E"])
        self.assertTrue(GalleryItem.objects.filter(title="B").exists())

    def test_teacher_cannot_reorder(self):
        self.client.force_login(self.teacher)
        self.assertEqual(self.reorder(["D", "C", "B", "A"]).status_code, 404)


class PublicSliderTests(GalleryFixture):
    def test_shows_active_preset_in_order_with_full_descriptions(self):
        self.make_preset("Inny", ("X",))
        preset = self.make_preset("Wyświetlany", ("C", "A", "B"), active=True)
        long_text = "Pełny opis bez ucinania. " * 40
        GalleryItem.objects.filter(title="A").update(description=long_text)
        response = self.client.get(reverse("gallery:slider_view"))
        html = response.content.decode()
        self.assertContains(response, 'aria-roledescription="karuzela"')
        self.assertLess(html.index('aria-label="1 z 3: C"'), html.index('aria-label="2 z 3: A"'))
        self.assertIn(long_text.strip(), html)
        self.assertNotIn("X</h3>", html)
        self.assertContains(response, f'data-interval="{preset.interval_seconds}"')

    def test_empty_state_without_active_preset(self):
        response = self.client.get(reverse("gallery:slider_view"))
        self.assertContains(response, "Galeria jest pusta")
