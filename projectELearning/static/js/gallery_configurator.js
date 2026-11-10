/* Konfigurator slajdera: przeciąganie (HTML5 Drag and Drop) zmienia kolejność kart i dodaje
   zdjęcia z bazy na wybraną pozycję. Zapis w tle (fetch + CSRF), wynik ogłasza aria-live.
   Alternatywa bez myszy: przyciski ↑ ↓ / „Dodaj” / „Usuń” (zwykłe formularze POST). */
(() => {
    const list = document.querySelector('[data-dnd-list]');
    if (!list) return;
    const live = document.querySelector('[data-dnd-live]');
    const announce = (text) => {
        live.textContent = '';
        window.setTimeout(() => { live.textContent = text; }, 50);
    };
    const marker = document.createElement('li');
    marker.className = 'dnd-marker';
    marker.setAttribute('aria-hidden', 'true');
    let dragged = null; // { id, title, fromList, element }

    const cards = () => Array.from(list.querySelectorAll('.dnd-card'));

    const refreshNumbers = () => {
        const items = cards();
        items.forEach((card, index) => {
            card.querySelector('[data-dnd-num]').textContent = index + 1;
            const up = card.querySelector('[data-dnd-up]');
            const down = card.querySelector('[data-dnd-down]');
            if (up) up.disabled = index === 0;
            if (down) down.disabled = index === items.length - 1;
        });
    };

    const post = (url, payload) => apiFetch(url, {
        method: 'POST',
        body: JSON.stringify(payload),
        headers: { Accept: 'application/json' },
    });

    const dropIndex = (clientY) => {
        // Pozycja wstawienia: przed pierwszą kartą, której środek jest poniżej kursora
        const items = cards().filter((card) => card !== dragged?.element);
        const index = items.findIndex((card) => {
            const box = card.getBoundingClientRect();
            return clientY < box.top + box.height / 2;
        });
        return { items, index: index === -1 ? items.length : index };
    };

    document.addEventListener('dragstart', (event) => {
        const card = event.target.closest?.('.dnd-card[draggable="true"]');
        if (!card) return;
        dragged = {
            id: Number(card.dataset.itemId),
            title: card.dataset.title,
            fromList: list.contains(card),
            element: card,
        };
        card.classList.add('is-dragging');
        event.dataTransfer.effectAllowed = 'move';
        event.dataTransfer.setData('text/plain', String(dragged.id));
    });

    document.addEventListener('dragend', () => {
        dragged?.element.classList.remove('is-dragging');
        marker.remove();
        list.classList.remove('is-over');
        dragged = null;
    });

    list.addEventListener('dragover', (event) => {
        if (!dragged) return;
        event.preventDefault();
        list.classList.add('is-over');
        const { items, index } = dropIndex(event.clientY);
        const before = items[index];
        if (before) list.insertBefore(marker, before); else list.appendChild(marker);
    });

    list.addEventListener('dragleave', (event) => {
        if (!list.contains(event.relatedTarget)) {
            marker.remove();
            list.classList.remove('is-over');
        }
    });

    list.addEventListener('drop', async (event) => {
        if (!dragged) return;
        event.preventDefault();
        const { index } = dropIndex(event.clientY);
        const { id, title, fromList, element } = dragged;
        marker.remove();
        list.classList.remove('is-over');

        if (fromList) {
            const items = cards().filter((card) => card !== element);
            const before = items[index];
            if (before) list.insertBefore(element, before); else list.appendChild(element);
            refreshNumbers();
            const order = cards().map((card) => Number(card.dataset.itemId));
            try {
                const response = await post(list.dataset.reorderUrl, { items: order });
                if (!response.ok) throw new Error((await response.json()).error);
                announce(`Zapisano kolejność: „${title}” na pozycji ${order.indexOf(id) + 1}.`);
            } catch (error) {
                showToast(error.message || 'Nie udało się zapisać kolejności — odświeżam stronę.', 'error');
                window.setTimeout(() => window.location.reload(), 1500);
            }
            return;
        }

        // Zdjęcie z bazy → nowa karta na wskazanej pozycji (strona odświeża się z nową kartą)
        try {
            const response = await post(list.dataset.addUrl, { item: id, position: index });
            if (!response.ok) throw new Error((await response.json()).error);
            window.location.hash = `card-${id}`;
            window.location.reload();
        } catch (error) {
            showToast(error.message || 'Nie udało się dodać zdjęcia.', 'error');
        }
    });
})();
