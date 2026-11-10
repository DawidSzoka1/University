/* ==========================================================================
   Koszyk — zmiana ilości bez przeładowania strony przez API chronione JWT
   (/orders/api/koszyk/…, apps/orders/api.py).

   - Token JWT z sesji (POST + CSRF na accounts:api_session_token) przez wspólny
     klient ChatClient.create (chat_client.js): Bearer, odświeżenie po 401.
   - Ulepsza istniejące formularze −/+/„Zmień”; bez JS albo przy błędzie sieci
     działają one zwykłym POST-em. Usuwanie pozycji (z potwierdzeniem) zostaje
     formularzem — strona i tak musi się wtedy przebudować.
   - Fokus zostaje na klikniętym przycisku, wynik ogłasza toast (aria-live).
   ========================================================================== */

(() => {
    const root = document.querySelector('[data-cart]');
    if (!root || !window.ChatClient) return;

    const client = window.ChatClient.create(root.dataset.tokenUrl);
    const itemUrl = (id) => root.dataset.itemApi.replace(/0\/$/, `${id}/`);
    const cartUrl = root.dataset.cartUrl;

    function setMinusConfirm(button, quantity, title) {
        if (quantity === 1) {
            button.dataset.confirm = `Usunąć „${title}” z koszyka?`;
            button.dataset.confirmBtn = 'Usuń';
            button.dataset.confirmDanger = 'true';
        } else {
            delete button.dataset.confirm;
            delete button.dataset.confirmBtn;
            delete button.dataset.confirmDanger;
        }
    }

    function renderItem(row, item, maxQuantity) {
        row.querySelector('input[name="quantity"]').value = item.quantity;
        row.querySelector('[data-line-total]').textContent = item.line_total_display;
        const note = row.querySelector('[data-cart-note]');
        note.textContent = item.note;
        note.hidden = !item.note;
        const plus = row.querySelector('[data-qty-plus]');
        const atMax = item.quantity >= maxQuantity;
        if (atMax && document.activeElement === plus) {
            row.querySelector('input[name="quantity"]').focus();
        }
        plus.disabled = atMax;
        setMinusConfirm(row.querySelector('[data-qty-minus]'), item.quantity, item.course.title);
    }

    function renderSummary(cart) {
        document.querySelectorAll('[data-cart-count]').forEach((el) => { el.textContent = cart.count; });
        document.querySelectorAll('[data-cart-total]').forEach((el) => { el.textContent = cart.total_display; });
        document.querySelectorAll('[data-cart-count-text]').forEach((el) => { el.textContent = cart.count_display; });
        document.querySelectorAll(`a.navbar__link[href="${cartUrl}"]`).forEach((link) => {
            link.setAttribute('aria-label', `Koszyk, ${cart.count_display}`);
            const tag = link.querySelector('.navbar__tag');
            if (tag) tag.textContent = cart.count;
        });
    }

    function requestedQuantity(form, row) {
        const current = parseInt(row.querySelector('input[name="quantity"]').value, 10) || 0;
        const step = form.querySelector('input[name="step"]');
        if (step) return current + parseInt(step.value, 10);
        return parseInt(form.querySelector('input[name="quantity"]').value, 10);
    }

    root.addEventListener('submit', async (event) => {
        const form = event.target;
        const row = form.closest('[data-cart-item]');
        if (!row || form.hasAttribute('data-confirm')) return;
        const quantity = requestedQuantity(form, row);
        // 0 i mniej = usunięcie: zostawiamy formularzowi (pełne przeładowanie listy).
        if (!Number.isInteger(quantity) || quantity < 1) return;

        event.preventDefault();
        const controls = row.querySelectorAll('button, input');
        row.setAttribute('aria-busy', 'true');
        controls.forEach((el) => { el.dataset.wasDisabled = el.disabled; el.disabled = true; });
        const focused = document.activeElement;

        let response;
        try {
            response = await client.api(itemUrl(row.dataset.cartItem), {
                method: 'PATCH',
                body: JSON.stringify({ quantity }),
            });
        } catch {
            form.submit(); // brak sieci / tokenu — zwykły POST
            return;
        }

        controls.forEach((el) => { el.disabled = el.dataset.wasDisabled === 'true'; delete el.dataset.wasDisabled; });
        row.removeAttribute('aria-busy');

        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
            window.showToast(data.detail || 'Nie udało się zmienić ilości.', 'error');
            if (focused) focused.focus();
            return;
        }
        if (data.item_id === null) {
            window.location.reload();
            return;
        }
        const item = data.cart.items.find((entry) => entry.id === data.item_id);
        renderItem(row, item, data.cart.max_quantity);
        renderSummary(data.cart);
        if (focused && !focused.disabled) focused.focus();
        window.showToast(data.detail, 'success');
    });
})();
