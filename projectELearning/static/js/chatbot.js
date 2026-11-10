/* ==========================================================================
   Asystent AI (Bielik) — okno czatu w prawym dolnym rogu.

   - Panel niemodalny: Tab wychodzi z niego na stronę (brak pułapki), Esc zamyka
     i oddaje fokus przyciskowi, który go otworzył.
   - Desktop (od 64rem): panel boczny po prawej, treść strony zwęża się w lewo;
     stan otwarcia pamiętany w sessionStorage między stronami (np. kolejne lekcje).
   - Kontekst strony (kurs / lekcja / wynik testu) z atrybutów [data-chat-page].
   - API chronione JWT — klient, strumień i dymki wiadomości: chat_client.js.
   - Odpowiedź przychodzi strumieniem NDJSON; czytnik ekranu dostaje całą
     odpowiedź jednym komunikatem (role=status) po jej zakończeniu.
   ========================================================================== */

(() => {
    const root = document.querySelector('[data-chat-widget]');
    if (!root) return;

    const page = document.querySelector('[data-chat-page]');
    const context = {
        course: page?.dataset.course || '',
        lesson: page?.dataset.lesson || '',
        attempt: page?.dataset.attempt || '',
    };

    const toggle = root.querySelector('[data-chat-toggle]');
    const panel = root.querySelector('#chat-panel');
    const closeBtn = root.querySelector('[data-chat-close]');
    const contextLabel = root.querySelector('[data-chat-context]');
    const historyLink = root.querySelector('[data-chat-history-link]');
    const clearBtn = root.querySelector('[data-chat-clear]');
    const lockBox = root.querySelector('[data-chat-lock]');
    const lockText = root.querySelector('[data-chat-lock-text]');
    const lockLink = root.querySelector('[data-chat-lock-link]');
    const log = root.querySelector('[data-chat-log]');
    const list = root.querySelector('[data-chat-list]');
    const empty = root.querySelector('[data-chat-empty]');
    const status = root.querySelector('[data-chat-status]');
    const errorBox = root.querySelector('[data-chat-error]');
    const form = root.querySelector('[data-chat-form]');
    const input = root.querySelector('[data-chat-input]');
    const sendBtn = root.querySelector('[data-chat-send]');
    const stopBtn = root.querySelector('[data-chat-stop]');

    const { errorDetail, readLines, messageItem, typingItem } = window.ChatClient;
    const { api } = window.ChatClient.create(root.dataset.tokenUrl);

    let loaded = false;
    let busy = false;
    let controller = null;
    let returnFocusTo = toggle;

    function query() {
        const params = new URLSearchParams();
        Object.entries(context).forEach(([key, value]) => value && params.set(key, value));
        return params.toString();
    }

    /* ------------------------------------------------------ rendering ---- */

    function showError(message) {
        errorBox.textContent = message || '';
        errorBox.hidden = !message;
        input.setAttribute('aria-describedby', message ? 'chat-input-hint chat-error' : 'chat-input-hint');
    }

    function setLock(lock) {
        lockBox.hidden = !lock;
        if (!lock) return;
        lockText.textContent = lock.notice;
        lockLink.href = lock.quiz_url;
        lockLink.textContent = 'Wróć do testu';
    }

    function updateEmpty() {
        empty.hidden = list.children.length > 0;
        clearBtn.disabled = list.children.length === 0 || busy;
    }

    function scrollToEnd() {
        log.scrollTop = log.scrollHeight;
    }

    function appendMessage(message) {
        const item = messageItem(message);
        list.append(item);
        updateEmpty();
        scrollToEnd();
        return item;
    }

    function setBusy(value) {
        busy = value;
        sendBtn.disabled = value;
        sendBtn.hidden = value;
        stopBtn.hidden = !value;
        log.setAttribute('aria-busy', String(value));
        updateEmpty();
    }

    /* ---------------------------------------------------------- data ---- */

    async function loadHistory() {
        showError('');
        status.textContent = 'Wczytywanie rozmowy…';
        try {
            const response = await api(`${root.dataset.historyUrl}?${query()}`);
            if (!response.ok) {
                showError(await errorDetail(response));
                status.textContent = '';
                return;
            }
            const data = await response.json();
            contextLabel.textContent = data.context.label;
            historyLink.href = data.context.session_url || root.dataset.listUrl;
            setLock(data.lock);
            list.replaceChildren(...data.messages.map(messageItem));
            updateEmpty();
            scrollToEnd();
            loaded = true;
            status.textContent = data.messages.length
                ? `Wczytano ${data.messages.length} ostatnich wiadomości.`
                : 'Brak wcześniejszych wiadomości.';
        } catch (error) {
            showError(error.message || 'Nie udało się wczytać rozmowy.');
            status.textContent = '';
        }
    }

    async function send(text) {
        showError('');
        setBusy(true);
        status.textContent = 'Asystent przygotowuje odpowiedź…';
        controller = new AbortController();
        let draft = null;
        let draftText = '';
        try {
            const response = await api(root.dataset.messageUrl, {
                method: 'POST',
                body: JSON.stringify({ ...context, message: text }),
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
                if (event.session_url) {
                    historyLink.href = event.session_url;
                }
                if (event.type === 'refusal') {
                    event.messages.forEach(appendMessage);
                    setLock(event.lock);
                    status.textContent = event.messages[1]?.text || '';
                } else if (event.type === 'start') {
                    appendMessage(event.question);
                    draft = typingItem();
                    list.append(draft);
                    updateEmpty();
                    scrollToEnd();
                } else if (event.type === 'delta') {
                    draftText += event.text;
                    draft.setText(draftText);
                    scrollToEnd();
                } else if (event.type === 'done') {
                    const finished = messageItem(event.message);
                    draft.replaceWith(finished);
                    draft = null;
                    scrollToEnd();
                    status.textContent = `Asystent: ${finished.querySelector('.chat-msg__body').textContent}`;
                } else if (event.type === 'error') {
                    if (draft) {
                        draft.previousElementSibling?.remove();
                        draft.remove();
                        draft = null;
                    }
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
            updateEmpty();
            if (!panel.hidden && (document.activeElement === document.body || panel.contains(document.activeElement))) input.focus();
        }
    }

    async function clearConversation() {
        const confirmed = await window.showConfirmDialog({
            title: 'Wyczyścić rozmowę?',
            message: 'Wiadomości z tej rozmowy zostaną trwale usunięte. Innych rozmów to nie dotyczy.',
            confirmText: 'Wyczyść',
            isDanger: true,
            triggerEl: clearBtn,
        });
        if (!confirmed) return;
        try {
            const response = await api(root.dataset.clearUrl, {
                method: 'POST',
                body: JSON.stringify(context),
            });
            if (!response.ok) {
                showError(await errorDetail(response));
                return;
            }
            list.replaceChildren();
            historyLink.href = root.dataset.listUrl;
            updateEmpty();
            status.textContent = 'Rozmowa została wyczyszczona.';
            input.focus();
        } catch (error) {
            showError(error.message || 'Nie udało się wyczyścić rozmowy.');
        }
    }

    /* ---------------------------------------------------- open/close ---- */

    const STORAGE_KEY = 'chat-panel-open';
    const docked = window.matchMedia('(min-width: 64rem)');
    const header = document.querySelector('.site-header');

    function syncLayout() {
        // Panel boczny zaczyna się pod nagłówkiem; treść strony zwęża się tylko na desktopie.
        if (header) document.documentElement.style.setProperty('--chat-top', `${header.offsetHeight}px`);
        document.body.classList.toggle('chat-docked', !panel.hidden && docked.matches);
    }

    function remember(isOpen) {
        try {
            sessionStorage.setItem(STORAGE_KEY, isOpen ? '1' : '0');
        } catch {
            /* tryb prywatny bez sessionStorage — panel po prostu się nie odtworzy */
        }
    }

    function open(trigger = toggle, { focus = true } = {}) {
        returnFocusTo = trigger || toggle;
        panel.hidden = false;
        root.classList.add('is-open');
        toggle.setAttribute('aria-expanded', 'true');
        document.querySelectorAll('[data-chat-open]').forEach((el) => el.setAttribute('aria-expanded', 'true'));
        syncLayout();
        remember(true);
        if (!loaded) loadHistory();
        if (focus) input.focus();
    }

    function close() {
        panel.hidden = true;
        root.classList.remove('is-open');
        toggle.setAttribute('aria-expanded', 'false');
        document.querySelectorAll('[data-chat-open]').forEach((el) => el.setAttribute('aria-expanded', 'false'));
        syncLayout();
        remember(false);
        (returnFocusTo && document.contains(returnFocusTo) ? returnFocusTo : toggle).focus();
    }

    docked.addEventListener('change', syncLayout);
    window.addEventListener('resize', syncLayout);

    toggle.addEventListener('click', () => (panel.hidden ? open(toggle) : close()));
    closeBtn.addEventListener('click', close);
    document.querySelectorAll('[data-chat-open]').forEach((button) => {
        button.setAttribute('aria-expanded', 'false');
        button.addEventListener('click', () => (panel.hidden ? open(button) : input.focus()));
    });

    panel.addEventListener('keydown', (event) => {
        if (event.key === 'Escape') {
            event.preventDefault();
            event.stopPropagation();
            close();
        }
    });

    // Pole rośnie razem z tekstem (do max-height z CSS), przycisk zostaje obok.
    function autosize() {
        input.style.height = 'auto';
        input.style.height = `${input.scrollHeight + 4}px`;
    }
    input.addEventListener('input', autosize);

    input.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
            event.preventDefault();
            form.requestSubmit();
        }
    });

    form.addEventListener('submit', (event) => {
        event.preventDefault();
        if (busy) return;
        const text = input.value.trim();
        if (!text) {
            showError('Wpisz pytanie.');
            input.setAttribute('aria-invalid', 'true');
            input.focus();
            return;
        }
        input.removeAttribute('aria-invalid');
        send(text);
    });

    stopBtn.addEventListener('click', () => controller?.abort());
    clearBtn.addEventListener('click', clearConversation);

    // Panel otwarty na poprzedniej stronie (np. przejście do następnej lekcji) zostaje otwarty
    // na desktopie — bez przenoszenia fokusu, żeby nie zabierać go ze strony.
    let restore = false;
    try {
        restore = sessionStorage.getItem(STORAGE_KEY) === '1';
    } catch {
        restore = false;
    }
    if (restore && docked.matches) open(toggle, { focus: false });
})();
