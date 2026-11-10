/* ==========================================================================
   ANTY-WCAG — skrypt CELOWO łamie dostępność (wymóg projektu, punkt 8). NIE POPRAWIAĆ.
   Opis naruszeń: /wcag-anty-wzorzec/opis/
   ========================================================================== */
(() => {
    const random = (min, max) => Math.round(min + Math.random() * (max - min));

    /* --- Nieprzewidywalne menu ------------------------------------------------
       1) rozwijanie tylko :hover (CSS), 2) zmiana szerokości okna → menu skacze
       w przeciwległy róg, 3) fokus na „Promocje” zmienia napis i położenie. */
    const menu = document.getElementById('aw-menu');
    let lastWidth = window.innerWidth;
    let resizeTimer;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(() => {
            if (window.innerWidth === lastWidth) return;
            lastWidth = window.innerWidth;
            menu.classList.toggle('aw-menu--tl');
            menu.classList.toggle('aw-menu--br');
        }, 120);
    });

    const jumpy = document.getElementById('aw-jumpy');
    const labels = ['Promocje', 'Okazje!', 'Kup teraz', 'Zniżki', 'Nowości', 'Kliknij!'];
    jumpy.addEventListener('focus', () => {
        jumpy.textContent = labels[random(0, labels.length - 1)];
        jumpy.style.transform = `translate(${random(-60, 120)}px, ${random(-10, 50)}px)`;
    });

    /* --- Przyciski działające tylko myszą (brak obsługi klawiatury) ---------- */
    const ctaResult = document.getElementById('aw-cta-result');
    ['aw-mouse-only', 'aw-mouse-only-2'].forEach((id) => {
        document.getElementById(id).addEventListener('click', () => {
            ctaResult.textContent = 'Dziękujemy! Zapisano.';
        });
    });

    /* --- Problematyczny formularz -------------------------------------------
       Błędy tylko kolorem ramki, ogólny komunikat bez wskazania pól,
       po 15 s od pierwszego znaku formularz czyści się bez ostrzeżenia. */
    const form = document.getElementById('formularz');
    const message = document.getElementById('aw-form-msg');
    const rules = {
        imie: (v) => v.trim().length >= 3,
        email: (v) => /^[^@\s]+@[^@\s]+\.[a-z]{2,}$/i.test(v),
        telefon: (v) => /^\d{9}$/.test(v),
        kod: (v) => /^\d{2}-\d{3}$/.test(v),
        rabat: (v) => v === '' || /^[A-Z]{4}\d{2}$/.test(v),
        kurs: (v) => v !== 'Wybierz',
    };
    let resetTimer = null;
    form.addEventListener('input', () => {
        if (resetTimer) return;
        resetTimer = setTimeout(() => {
            form.reset();
            form.querySelectorAll('.aw-invalid').forEach((el) => el.classList.remove('aw-invalid'));
            message.textContent = '';
            resetTimer = null;
        }, 15000);
    });
    form.addEventListener('submit', (event) => {
        event.preventDefault();
        let ok = true;
        Object.entries(rules).forEach(([name, check]) => {
            const field = form.elements[name];
            const valid = check(field.value);
            field.classList.toggle('aw-invalid', !valid);
            ok = ok && valid;
        });
        message.textContent = ok ? 'OK.' : 'Błąd.';
    });

    /* --- Pułapka klawiaturowa: Tab i Shift+Tab krążą między dwoma polami ---- */
    const trapFields = Array.from(document.querySelectorAll('.aw-trap__field'));
    trapFields.forEach((field, index) => {
        field.addEventListener('keydown', (event) => {
            if (event.key === 'Tab') {
                event.preventDefault();
                trapFields[(index + 1) % trapFields.length].focus();
            }
            if (event.key === 'Escape') {
                event.preventDefault();
            }
        });
    });

    /* --- Popup 1: otwiera się, gdy fokus trafi na „Kod rabatowy”;
       przycisk × ucieka przed kursorem (po kilku ucieczkach daje się kliknąć). */
    const promoPopup = document.getElementById('aw-popup-promo');
    const closeButton = document.getElementById('aw-popup-close');
    let escapes = 0;
    document.getElementById('aw-promo-field').addEventListener('focus', () => {
        escapes = 0;
        closeButton.style.top = '30%';
        closeButton.style.left = '60%';
        promoPopup.hidden = false;
    });
    closeButton.addEventListener('mouseenter', () => {
        if (escapes >= 6) return;
        escapes += 1;
        closeButton.style.top = `${random(5, 90)}%`;
        closeButton.style.left = `${random(5, 90)}%`;
    });
    closeButton.addEventListener('click', () => {
        promoPopup.hidden = true;
    });

    /* --- Popup 2: pojawia się sam po 5 s, bez zamykania (ani ×, ani Esc),
       znika sam po 8 s i wraca co 40 s. */
    const autoPopup = document.getElementById('aw-popup-auto');
    const showAuto = () => {
        autoPopup.hidden = false;
        setTimeout(() => { autoPopup.hidden = true; }, 8000);
    };
    setTimeout(showAuto, 5000);
    setInterval(showAuto, 40000);
})();
