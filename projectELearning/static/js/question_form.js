/* Formularz pytania: pokazuje sekcję „Odpowiedzi do wyboru” albo „Odpowiedź otwarta”
   zależnie od typu pytania. Bez JS obie sekcje są widoczne (serwer i tak waliduje). */
(() => {
    const form = document.querySelector('[data-question-form]');
    if (!form) return;
    const choices = form.querySelector('[data-section="choices"]');
    const open = form.querySelector('[data-section="open"]');
    const sync = () => {
        const type = form.querySelector('input[name="question_type"]:checked')?.value;
        const isOpen = type === 'open';
        choices.hidden = isOpen;
        open.hidden = !isOpen;
    };
    form.querySelectorAll('input[name="question_type"]').forEach((input) => input.addEventListener('change', sync));
    sync();
})();
