/* ==========================================================================
   Wspólny klient asystenta AI — używany przez okno czatu (chatbot.js)
   i stronę rozmowy (chat_page.js).

   - Token JWT z sesji (POST + CSRF), odświeżany po 401; API przyjmuje tylko Bearer.
   - readLines: strumień NDJSON z /chatbot/api/message/.
   - messageItem / typingItem: elementy listy wiadomości (bez innerHTML z tekstu
     użytkownika; HTML odpowiedzi budowany po stronie serwera po escape).
   ========================================================================== */

(() => {
    const FALLBACK_ERROR = 'Coś poszło nie tak. Spróbuj ponownie.';

    function create(tokenUrl) {
        let token = null;
        let tokenExpires = 0;

        async function getToken(force = false) {
            if (!force && token && Date.now() < tokenExpires) return token;
            const response = await fetch(tokenUrl, {
                method: 'POST',
                credentials: 'same-origin',
                headers: { 'X-CSRFToken': getCsrfToken() },
            });
            if (!response.ok) throw new Error('Sesja wygasła — zaloguj się ponownie.');
            const data = await response.json();
            token = data.access;
            tokenExpires = Date.now() + (data.expires_in - 30) * 1000;
            return token;
        }

        async function api(url, options = {}, retry = true) {
            const response = await fetch(url, {
                ...options,
                credentials: 'omit',
                headers: {
                    'Content-Type': 'application/json',
                    Authorization: `Bearer ${await getToken()}`,
                    ...(options.headers || {}),
                },
            });
            if (response.status === 401 && retry) {
                await getToken(true);
                return api(url, options, false);
            }
            return response;
        }

        return { api };
    }

    async function errorDetail(response) {
        try {
            const data = await response.json();
            return data.detail || FALLBACK_ERROR;
        } catch {
            return FALLBACK_ERROR;
        }
    }

    async function* readLines(response) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            let newline;
            while ((newline = buffer.indexOf('\n')) >= 0) {
                const line = buffer.slice(0, newline).trim();
                buffer = buffer.slice(newline + 1);
                if (line) yield JSON.parse(line);
            }
        }
        if (buffer.trim()) yield JSON.parse(buffer);
    }

    function messageItem(message) {
        const item = document.createElement('li');
        const isBot = message.sender === 'bot';
        item.className = `chat-msg chat-msg--${isBot ? 'bot' : 'user'}${message.is_refusal ? ' chat-msg--refusal' : ''}`;

        const meta = document.createElement('p');
        meta.className = 'chat-msg__meta';
        const who = document.createElement('span');
        who.className = 'chat-msg__who';
        who.textContent = isBot ? 'Asystent' : 'Ty';
        meta.append(who);
        if (message.time) {
            const time = document.createElement('time');
            time.dateTime = message.timestamp || '';
            time.textContent = message.time;
            meta.append(' · ', time);
        }
        if (!isBot && message.lesson) {
            const lesson = document.createElement('span');
            lesson.className = 'chat-msg__lesson';
            lesson.textContent = `lekcja „${message.lesson}”`;
            meta.append(' · ', lesson);
        }

        const body = document.createElement('div');
        body.className = 'chat-msg__body';
        if (isBot && message.html) {
            // HTML budowany po stronie serwera z tekstu po escape (apps/chatbot/formatting.py).
            body.innerHTML = message.html;
        } else {
            body.textContent = message.text;
        }
        item.append(meta, body);
        return item;
    }

    /** Dymek „Asystent pisze…” z animowanymi kropkami; tekst odpowiedzi wpisuje setText. */
    function typingItem() {
        const item = messageItem({ sender: 'bot', text: '' });
        item.classList.add('chat-msg--typing');
        const body = item.querySelector('.chat-msg__body');
        const dots = document.createElement('span');
        dots.className = 'chat-typing';
        dots.setAttribute('aria-hidden', 'true');
        dots.append(document.createElement('span'), document.createElement('span'), document.createElement('span'));
        const label = document.createElement('span');
        label.className = 'chat-typing__label';
        label.textContent = 'Asystent pisze…';
        body.append(dots, label);
        item.setText = (text) => {
            item.classList.remove('chat-msg--typing');
            body.textContent = text;
        };
        return item;
    }

    window.ChatClient = { create, errorDetail, readLines, messageItem, typingItem };
})();
