/* Rejestracja: sprawdza na bieżąco, czy e-mail i nazwa użytkownika są wolne.
   Komunikat tekstowy pod polem (aria-live + aria-describedby), nie tylko kolorem.
   Serwer i tak waliduje formularz — to tylko wcześniejsza podpowiedź. */
(() => {
    const form = document.querySelector('[data-availability-form]');
    if (!form) return;
    const endpoint = form.dataset.availabilityUrl;
    const loginUrl = form.dataset.loginUrl;

    const setup = (input, field) => {
        if (!input) return;
        const note = document.createElement('p');
        note.id = `${input.id}-availability`;
        note.className = 'form-availability';
        note.setAttribute('aria-live', 'polite');
        const group = input.closest('.form-group');
        if (group) group.append(note); else input.after(note);
        const describedBy = (input.getAttribute('aria-describedby') || '').split(' ').filter(Boolean);
        describedBy.push(note.id);
        input.setAttribute('aria-describedby', describedBy.join(' '));

        let timer;
        let lastValue = '';
        const looksValid = (value) => (field === 'email' ? /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value) : value.length >= 3);

        const check = async () => {
            const value = input.value.trim();
            if (value === lastValue) return;
            lastValue = value;
            note.textContent = '';
            note.className = 'form-availability';
            if (!looksValid(value)) return;
            try {
                const url = `${endpoint}?field=${field}&value=${encodeURIComponent(value)}`;
                const response = await fetch(url, { credentials: 'same-origin', headers: { Accept: 'application/json' } });
                if (!response.ok || input.value.trim() !== value) return;
                const data = await response.json();
                note.className = `form-availability form-availability--${data.available ? 'ok' : 'taken'}`;
                note.textContent = data.message;
                if (!data.available && field === 'email' && loginUrl) {
                    const link = document.createElement('a');
                    link.href = loginUrl;
                    link.textContent = 'Przejdź do logowania';
                    note.append(' ', link);
                }
            } catch {
                /* brak sieci — walidacja serwera i tak zadziała po wysłaniu */
            }
        };

        input.addEventListener('input', () => {
            window.clearTimeout(timer);
            timer = window.setTimeout(check, 500);
        });
        input.addEventListener('blur', check);
        if (input.value.trim()) check();
    };

    setup(form.querySelector('input[name="email"]'), 'email');
    setup(form.querySelector('input[name="username"]'), 'username');
})();
