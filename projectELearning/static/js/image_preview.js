/* ==========================================================================
   Podgląd zdjęć przed wysłaniem formularza.

   Działa dla każdego <input type="file"> przyjmującego obrazy (accept="image/…"):
   miniatura kursu, galeria, zdjęcie w pytaniu, awatar, zdjęcie z rozwiązaniem.
   - Pod polem pokazuje wybrane zdjęcie z nazwą, wymiarami i rozmiarem pliku.
   - data-max-mb / data-max-size (z ImagePreviewInput) → podpowiedź o limicie
     i o zmniejszeniu zdjęcia po zapisie. Decyduje i tak walidacja na serwerze.
   - Obecne zdjęcie (figura [data-image-current]) dostaje dopisek, że zostanie
     zastąpione albo usunięte.
   - „Usuń wybór” czyści pole i oddaje mu fokus. Region aria-live ogłasza zmianę.
   ========================================================================== */

(() => {
    const formatMb = (bytes) => `${(bytes / 1024 / 1024).toLocaleString('pl-PL', { maximumFractionDigits: 1 })} MB`;
    const formatKb = (bytes) => (bytes < 1024 * 1024 ? `${Math.max(1, Math.round(bytes / 1024))} kB` : formatMb(bytes));

    function parseSize(value) {
        const match = /^(\d+)x(\d+)$/.exec(value || '');
        return match ? [Number(match[1]), Number(match[2])] : null;
    }

    function enhance(input) {
        if (input.dataset.previewReady) return;
        input.dataset.previewReady = 'true';

        const wrapper = input.closest('[data-image-input]');
        const current = wrapper?.querySelector('[data-image-current]');
        const note = current?.querySelector('[data-image-note]');
        const clear = wrapper?.querySelector('[data-image-clear]');
        const maxMb = Number(input.dataset.maxMb) || 0;
        const maxSize = parseSize(input.dataset.maxSize);
        let urls = [];

        const preview = document.createElement('div');
        preview.className = 'image-input__preview';
        preview.setAttribute('aria-live', 'polite');
        preview.hidden = true;
        input.insertAdjacentElement('afterend', preview);

        function updateCurrent() {
            if (!current) return;
            let text = '';
            if (input.files.length) text = '— po zapisie zastąpi je nowe zdjęcie.';
            else if (clear?.checked) text = '— zostanie usunięte po zapisie.';
            current.classList.toggle('image-input__figure--outgoing', Boolean(text));
            if (note) {
                note.textContent = text;
                note.hidden = !text;
            }
        }

        function figureFor(file) {
            const figure = document.createElement('figure');
            figure.className = 'image-input__figure image-input__figure--new';
            const caption = document.createElement('figcaption');
            caption.className = 'image-input__caption';
            const label = document.createElement('strong');
            label.textContent = 'Wybrane zdjęcie';
            const details = document.createElement('span');
            details.textContent = ` · ${file.name} · ${formatKb(file.size)}`;
            caption.append(label, details);

            const warnings = [];
            if (!file.type.startsWith('image/')) {
                warnings.push(`To nie jest zdjęcie — wybierz plik JPEG, PNG lub WebP.`);
            } else if (maxMb && file.size > maxMb * 1024 * 1024) {
                warnings.push(`Plik ma ${formatMb(file.size)}, a limit to ${maxMb} MB — wybierz mniejszy.`);
            }

            if (file.type.startsWith('image/')) {
                const img = document.createElement('img');
                img.className = 'image-input__img';
                img.alt = `Podgląd: ${file.name}`;
                const url = URL.createObjectURL(file);
                urls.push(url);
                img.addEventListener('load', () => {
                    let text = ` · ${img.naturalWidth}×${img.naturalHeight} px`;
                    if (maxSize && (img.naturalWidth > maxSize[0] || img.naturalHeight > maxSize[1])) {
                        text += ` — po zapisie zostanie zmniejszone do maks. ${maxSize[0]}×${maxSize[1]} px`;
                    }
                    details.textContent += text;
                }, { once: true });
                img.addEventListener('error', () => {
                    img.remove();
                    const warning = document.createElement('p');
                    warning.className = 'image-input__warning';
                    warning.textContent = 'Przeglądarka nie potrafi wyświetlić tego pliku — sprawdź, czy to na pewno zdjęcie.';
                    figure.append(warning);
                }, { once: true });
                img.src = url;
                figure.append(img);
            }
            figure.append(caption);
            warnings.forEach((text) => {
                const warning = document.createElement('p');
                warning.className = 'image-input__warning';
                warning.textContent = text;
                figure.append(warning);
            });
            return figure;
        }

        function render() {
            urls.forEach((url) => URL.revokeObjectURL(url));
            urls = [];
            preview.replaceChildren();
            const files = Array.from(input.files);
            preview.hidden = files.length === 0;
            updateCurrent();
            if (!files.length) return;

            files.forEach((file) => preview.append(figureFor(file)));
            const reset = document.createElement('button');
            reset.type = 'button';
            reset.className = 'btn btn--ghost btn--sm';
            reset.textContent = files.length > 1 ? 'Usuń wybrane pliki' : 'Usuń wybór';
            reset.addEventListener('click', () => {
                input.value = '';
                render();
                input.focus();
            });
            preview.append(reset);
        }

        input.addEventListener('change', render);
        clear?.addEventListener('change', updateCurrent);
        // Po „Wstecz” przeglądarka potrafi przywrócić wybrany plik — pokaż go od razu.
        if (input.files.length) render();
    }

    function init(root = document) {
        root.querySelectorAll('input[type="file"][accept*="image"]').forEach(enhance);
    }

    document.addEventListener('DOMContentLoaded', () => init());
    window.ImagePreview = { init };
})();
