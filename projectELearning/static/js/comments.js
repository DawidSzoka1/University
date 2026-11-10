/* Komentarze pod kursem: głosy „lubię / nie lubię” bez przeładowania strony.
   Bez JS formularz wysyła zwykły POST i wraca do komentarza. Wynik ogłasza region aria-live. */
(() => {
    const live = document.querySelector('[data-vote-live]');

    const announce = (text) => {
        if (!live) return;
        live.textContent = '';
        window.setTimeout(() => { live.textContent = text; }, 50);
    };

    document.addEventListener('submit', async (event) => {
        const form = event.target.closest('[data-vote-form]');
        if (!form) return;
        const button = event.submitter;
        if (!button || !button.value) return;
        event.preventDefault();

        const buttons = form.querySelectorAll('button[name="value"]');
        buttons.forEach((b) => { b.disabled = true; });
        try {
            const body = new URLSearchParams(new FormData(form));
            body.set('value', button.value);
            const response = await apiFetch(form.action, {
                method: 'POST',
                body,
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                    Accept: 'application/json',
                },
            });
            const data = await response.json().catch(() => ({}));
            if (!response.ok) {
                showToast(data.error || 'Nie udało się zapisać głosu. Spróbuj ponownie.', 'error');
                return;
            }
            form.querySelector('[data-vote-count="likes"]').textContent = data.likes;
            form.querySelector('[data-vote-count="dislikes"]').textContent = data.dislikes;
            buttons.forEach((b) => {
                b.setAttribute('aria-pressed', String(Number(b.value) === data.vote));
            });
            const state = data.vote === 1 ? 'Polubiono komentarz.'
                : data.vote === -1 ? 'Oznaczono: nie lubię.'
                : 'Cofnięto głos.';
            announce(`${state} Polubienia: ${data.likes}, na nie: ${data.dislikes}.`);
        } catch (error) {
            showToast('Brak połączenia — głos nie został zapisany.', 'error');
        } finally {
            buttons.forEach((b) => { b.disabled = false; });
            button.focus();
        }
    });
})();
