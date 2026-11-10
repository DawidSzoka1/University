"""Strona Anty-WCAG: celowe naruszenia są na miejscu, a strona jest odizolowana od stylów platformy."""

from pathlib import Path

from django.conf import settings
from django.test import TestCase
from django.urls import reverse

STATIC = Path(settings.BASE_DIR) / "static" / "anti_wcag"


class AntiWcagPageTests(TestCase):
    def setUp(self):
        self.response = self.client.get(reverse("core:anti_wcag"))
        self.html = self.response.content.decode()

    def test_page_is_isolated_from_platform_styles(self):
        self.assertEqual(self.response.status_code, 200)
        self.assertTemplateNotUsed(self.response, "base.html")
        self.assertNotIn("css/main.css", self.html)
        self.assertIn("anti_wcag/anty-wcag.css", self.html)
        self.assertIn("anti_wcag/anty-wcag.js", self.html)
        self.assertNotIn("{#", self.html)  # komentarz szablonu nie może wyciec na stronę

    def test_basic_violations(self):
        css = (STATIC / "anty-wcag.css").read_text()
        self.assertIn("outline: none !important", css)  # brak fokusu
        self.assertIn("#fff59d", css)  # jasnożółty na białym
        self.assertIn("overflow: hidden", css)  # ucinanie po powiększeniu
        self.assertIn("user-scalable=no", self.html)
        self.assertNotIn('lang="pl"', self.html)
        self.assertLess(self.html.index("<h4"), self.html.index("<h1"))  # zła kolejność nagłówków
        self.assertIn('class="aw-dot aw-dot--green"></span>', self.html)  # legenda tylko kolorem

    def test_keyboard_violations(self):
        for value in ('tabindex="9"', 'tabindex="3"', 'tabindex="12"', 'tabindex="1"'):
            self.assertIn(value, self.html)
        self.assertIn('<div class="aw-btn" id="aw-mouse-only">', self.html)
        self.assertIn('tabindex="-1" id="aw-mouse-only-2"', self.html)
        self.assertIn('id="aw-trap"', self.html)
        js = (STATIC / "anty-wcag.js").read_text()
        self.assertIn("event.key === 'Tab'", js)

    def test_menu_form_and_popup_behaviour(self):
        css = (STATIC / "anty-wcag.css").read_text()
        js = (STATIC / "anty-wcag.js").read_text()
        self.assertIn(".aw-menu__item:hover .aw-menu__sub", css)
        self.assertIn("addEventListener('resize'", js)
        self.assertIn("jumpy.addEventListener('focus'", js)
        self.assertIn("form.reset()", js)
        self.assertIn("15000", js)
        self.assertIn("closeButton.addEventListener('mouseenter'", js)
        self.assertIn("setTimeout(showAuto, 5000)", js)
        self.assertIn('id="aw-popup-auto"', self.html)
        self.assertNotIn("alert(", js)


class AntiWcagGuideTests(TestCase):
    def test_guide_is_accessible_page_linked_from_footer(self):
        guide_url = reverse("core:anti_wcag_guide")
        response = self.client.get(guide_url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "base.html")
        self.assertContains(response, reverse("core:anti_wcag"))
        for heading in ("Nawigacja klawiaturą", "Nieprzewidywalne menu", "Problematyczny formularz", "Uporczywe popupy"):
            self.assertContains(response, heading)
        self.assertContains(self.client.get(reverse("core:home")), f'href="{guide_url}"')
