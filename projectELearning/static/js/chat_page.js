/* ==========================================================================
   Strona rozmowy z asystentem (chatbot/session_detail.html).

   Z JavaScriptem formularz wysyła pytanie do API (JWT) i pokazuje odpowiedź na
   bieżąco: pytanie od razu w rozmowie, potem dymek „Asystent pisze…” i tekst
   napływający fragmentami. Bez JavaScriptu działa zwykły POST formularza.
   ========================================================================== */

(() => {
    const form = document.querySelector('[data-chat-page-form]');
    if (!form || !window.ChatClient) return;

    const { errorDetail, readLines, messageItem, typingItem } = window.ChatClient;
    const { api } = window.ChatClient.create(form.dataset.tokenUrl);
    const input = form.querySelector('textarea');
    const send = form.querySelector('[data-chat-page-send]');
    const stop = form.querySelector('[data-chat-page-stop]');
    const status = form.querySelector('[data-chat-page-status]');
    const errorBox = form.querySelector('[data-chat-page-error]');
    const log = document.querySelector('[data-chat-page-log]');
    const streaming = form.dataset.stream === 'true';
    const dayFormat = new Intl.DateTimeFormat('pl-PL', { day: 'numeric', month: 'long', year: 'numeric' });
    const baseDescribedBy = (input.getAttribute('aria-describedby') || '')
        .split(' ').filter((id) => id && id !== errorBox.id).join(' ');
    let controller = null;

    const autosize = () => {
        input.style.height = 'auto';
        input.style.height = `${input.scrollHeight + 4}px`;
    };
    input.addEventListener('input', autosize);
    autosize();

    input.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
            event.preventDefault();
            form.requestSubmit();
        }
    });

    function transcript() {
        let list = log.querySelector('.chat-transcript');
        if (!list) {
            log.querySelector('[data-chat-page-empty]')?.remove();
            list = document.createElement('ol');
            list.className = 'chat-transcript';
            list.setAttribute('aria-label', 'Wiadomości');
            log.append(list);
        }
        return list;
    }

    function ensureToday(list) {
        const today = dayFormat.format(new Date());
        const days = list.querySelectorAll('.chat-transcript__day');
        if (days.length && days[days.length - 1].textContent.trim() === today) return;
        const day = document.createElement('li');
        day.className = 'chat-transcript__day';
        day.setAttribute('aria-hidden', 'true');
        const label = document.createElement('span');
        label.textContent = today;
        day.append(label);
        list.append(day);
    }

    // Przewija tylko listę wiadomości, nie stronę.
    function reveal() {
        log.scrollTop = log.scrollHeight;
    }

    function showError(message) {
        errorBox.textContent = message || '';
        errorBox.hidden = !message;
        const ids = [baseDescribedBy, message ? errorBox.id : ''].filter(Boolean).join(' ');
        input.setAttribute('aria-describedby', ids);
        input.toggleAttribute('aria-invalid', Boolean(message));
    }

    function setBusy(value) {
        send.hidden = value;
        stop.hidden = !value;
        log.setAttribute('aria-busy', String(value));
    }

    async function ask(text) {
        showError('');
        setBusy(true);
        status.textContent = 'Asystent przygotowuje odpowiedź…';
        controller = new AbortController();
        const list = transcript();
        let draft = null;
        let question = null;
        let draftText = '';
        try {
            const response = await api(form.dataset.messageUrl, {
                method: 'POST',
                body: JSON.stringify({ course: form.dataset.course || '', message: text }),
                signal: controller.signal,
            });
            if (!response.ok) {
                showError(await errorDetail(response));
                status.textContent = '';
                return;
            }
            input.value = '';
            autosize();
            for await (const event of readLines(response)) {
                if (event.type === 'refusal') {
                    ensureToday(list);
                    event.messages.forEach((message) => list.append(messageItem(message)));
                    reveal(list.lastElementChild);
                    status.textContent = event.messages[1]?.text || '';
                } else if (event.type === 'start') {
                    ensureToday(list);
                    question = messageItem(event.question);
                    draft = typingItem();
                    list.append(question, draft);
                    reveal(draft);
                } else if (event.type === 'delta') {
                    draftText += event.text;
                    draft.setText(draftText);
                    reveal(draft);
                } else if (event.type === 'done') {
                    const finished = messageItem(event.message);
                    draft.replaceWith(finished);
                    draft = null;
                    reveal(finished);
                    status.textContent = `Asystent: ${finished.querySelector('.chat-msg__body').textContent}`;
                } else if (event.type === 'error') {
                    question?.remove();
                    draft?.remove();
                    draft = null;
                    input.value = text;
                    autosize();
                    showError(event.detail);
                    status.textContent = '';
                }
            }
        } catch (error) {
            if (error.name === 'AbortError') {
                if (draft) draft.setText(draftText || '(przerwano)');
                status.textContent = 'Przerwano odpowiedź.';
            } else {
                input.value = input.value || text;
                showError(error.message || 'Nie udało się wysłać pytania.');
                status.textContent = '';
            }
        } finally {
            controller = null;
            setBusy(false);
            if (document.activeElement === document.body || form.contains(document.activeElement)) input.focus();
        }
    }

    form.addEventListener('submit', (event) => {
        if (!streaming) {
            // Starsza strona rozmowy — zwykły POST, ale z widocznym stanem wysyłania.
            if (!input.value.trim()) return;
            send.disabled = true;
            send.textContent = 'Wysyłanie…';
            status.textContent = 'Asystent przygotowuje odpowiedź…';
            return;
        }
        event.preventDefault();
        if (controller) return;
        const text = input.value.trim();
        if (!text) {
            showError('Wpisz pytanie.');
            input.focus();
            return;
        }
        ask(text);
    });

    stop.addEventListener('click', () => controller?.abort());

    // Najnowsza wiadomość widoczna od razu — w ramce rozmowy, bez przewijania strony.
    reveal();
})();
