/* Slajder galerii: zapętlony dla dowolnej liczby kart, 1/2/3 karty zależnie od szerokości
   (zmienna --per-view z CSS), przesuwa się zawsze o jedną kartę. Pętla bez klonów: po
   przesunięciu pierwsza karta przechodzi na koniec (wstecz — ostatnia na początek).
   Dostępność: przyciski i strzałki ←/→, pauza (przycisk, najechanie, fokus w środku),
   brak automatu przy prefers-reduced-motion, karty poza ekranem są inert. */
(() => {
    const root = document.querySelector('[data-slider]');
    if (!root) return;
    const track = root.querySelector('[data-slider-track]');
    const controls = root.querySelector('[data-slider-controls]');
    const toggle = root.querySelector('[data-slider-toggle]');
    const status = root.querySelector('[data-slider-status]');
    const live = root.querySelector('[data-slider-live]');
    const total = track.children.length;
    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
    const intervalMs = Math.max(3, Number(root.dataset.interval) || 6) * 1000;

    let busy = false;
    let timer = null;
    let userPaused = root.dataset.autoplay !== '1' || reduceMotion.matches;
    let hoverPaused = false;

    const perView = () => Number(getComputedStyle(track).getPropertyValue('--per-view')) || 1;
    const canSlide = () => total > perView();
    const slides = () => Array.from(track.children);

    const update = (announce = false) => {
        const visible = perView();
        slides().forEach((slide, index) => {
            const hidden = index >= visible;
            slide.inert = hidden;
            slide.setAttribute('aria-hidden', String(hidden));
        });
        const first = Number(track.firstElementChild.dataset.index);
        const last = Number(slides()[Math.min(visible, total) - 1].dataset.index);
        const range = visible === 1 || total === 1 ? `Karta ${first}` : `Karty ${first}–${last}`;
        status.textContent = `${range} z ${total}`;
        if (announce) live.textContent = status.textContent;
        controls.hidden = !canSlide();
        toggle.textContent = userPaused ? 'Wznów przewijanie' : 'Zatrzymaj';
    };

    const step = () => track.firstElementChild.getBoundingClientRect().width;

    const animate = (transform, done) => {
        if (reduceMotion.matches) {
            done();
            return;
        }
        track.style.transition = 'transform 0.45s ease';
        track.style.transform = transform;
        track.addEventListener('transitionend', function finish() {
            track.removeEventListener('transitionend', finish);
            done();
        });
    };

    const next = (announce = false) => {
        if (busy || !canSlide()) return;
        busy = true;
        animate(`translateX(-${step()}px)`, () => {
            track.style.transition = 'none';
            track.appendChild(track.firstElementChild);
            track.style.transform = 'translateX(0)';
            busy = false;
            update(announce);
        });
    };

    const prev = (announce = false) => {
        if (busy || !canSlide()) return;
        busy = true;
        track.style.transition = 'none';
        track.prepend(track.lastElementChild);
        track.style.transform = `translateX(-${step()}px)`;
        void track.offsetWidth; // wymuszenie przeliczenia przed animacją
        animate('translateX(0)', () => {
            track.style.transition = 'none';
            track.style.transform = 'translateX(0)';
            busy = false;
            update(announce);
        });
        if (reduceMotion.matches) track.style.transform = 'translateX(0)';
    };

    const schedule = () => {
        window.clearInterval(timer);
        timer = null;
        if (!userPaused && !hoverPaused && canSlide()) {
            timer = window.setInterval(() => next(false), intervalMs);
        }
    };

    root.querySelector('[data-slider-next]').addEventListener('click', () => next(true));
    root.querySelector('[data-slider-prev]').addEventListener('click', () => prev(true));
    toggle.addEventListener('click', () => {
        userPaused = !userPaused;
        update();
        schedule();
        live.textContent = userPaused ? 'Przewijanie zatrzymane.' : 'Przewijanie wznowione.';
    });
    root.addEventListener('keydown', (event) => {
        if (event.key === 'ArrowRight') { event.preventDefault(); next(true); }
        if (event.key === 'ArrowLeft') { event.preventDefault(); prev(true); }
    });
    // Pauza, gdy użytkownik czyta (kursor nad slajderem) albo jest w nim fokus
    root.addEventListener('mouseenter', () => { hoverPaused = true; schedule(); });
    root.addEventListener('mouseleave', () => { hoverPaused = false; schedule(); });
    root.addEventListener('focusin', () => { hoverPaused = true; schedule(); });
    root.addEventListener('focusout', (event) => {
        if (!root.contains(event.relatedTarget)) { hoverPaused = false; schedule(); }
    });
    window.addEventListener('resize', () => { update(); schedule(); });
    reduceMotion.addEventListener('change', () => {
        if (reduceMotion.matches) userPaused = true;
        update();
        schedule();
    });

    root.classList.add('is-ready');
    update();
    schedule();
})();
