/* Panel admina (django-unfold): tryb jasny/ciemny zawsze zgodny z motywem platformy.
   Unfold zapamiętuje tryb w localStorage („adminTheme”, Alpine $persist) i ta wartość
   wygrywa z ustawieniem serwera. Serwer wpisuje tryb motywu jako klasę <html>
   (UNFOLD["THEME"] → apps.core.colors.get_unfold_theme_mode), więc przed startem
   Alpine nadpisujemy nim zapamiętaną wartość. Skrypt ładuje się w <head>, a Alpine
   ma `defer`, więc kolejność jest gwarantowana. */
(function () {
    var root = document.documentElement;
    var mode = root.classList.contains('dark') ? 'dark' : root.classList.contains('light') ? 'light' : null;
    if (!mode) return;
    try {
        localStorage.setItem('adminTheme', JSON.stringify(mode));
    } catch (error) {
        /* localStorage niedostępny (tryb prywatny) — klasa z serwera i tak obowiązuje */
    }
}());

/* Dostępność panelu (axe/Lighthouse): Unfold nie oznacza głównego obszaru landmarkiem <main>,
   a SimpleBar tworzy kilka regionów o tej samej nazwie „scrollable content”. */
(function () {
    function fixLandmarks() {
        var main = document.getElementById('main');
        if (main && !main.hasAttribute('role')) {
            main.setAttribute('role', 'main');
        }
        var regions = document.querySelectorAll('.simplebar-content-wrapper[role="region"]');
        regions.forEach(function (region, index) {
            var owner = region.closest('[id]');
            var name = owner && owner.id === 'nav-sidebar-apps' ? 'Menu aplikacji' : 'Lista przewijana ' + (index + 1);
            region.setAttribute('aria-label', name);
        });
        /* Pola w tabelach (lista z list_editable, formularze inline) nie mają <label>:
           nazwa = nagłówek kolumny + nazwa wiersza. Puste nagłówki dostają tekst dla czytników. */
        document.querySelectorAll('#content table').forEach(function (table) {
            var headerCells = Array.from(table.querySelectorAll('thead th'));
            headerCells.forEach(function (th) {
                if (!th.textContent.trim() && !th.querySelector('.sr-only')) {
                    var span = document.createElement('span');
                    span.className = 'sr-only';
                    span.textContent = 'Akcje';
                    th.appendChild(span);
                }
            });
            var headers = headerCells.map(function (th) { return th.textContent.replace(/\s+/g, ' ').trim(); });
            table.querySelectorAll('tbody tr').forEach(function (row, rowIndex) {
                var cells = Array.from(row.children);
                var named = row.querySelector('th');
                var rowName = named ? named.textContent.replace(/\s+/g, ' ').trim() : 'wiersz ' + (rowIndex + 1);
                cells.forEach(function (cell, index) {
                    cell.querySelectorAll('input:not([type="hidden"]), select, textarea').forEach(function (field) {
                        if ((field.labels && field.labels.length) || field.getAttribute('aria-label')) return;
                        var column = headers[index] || 'Pole';
                        field.setAttribute('aria-label', field.classList.contains('action-select') ? 'Zaznacz: ' + rowName : column + ' — ' + rowName);
                    });
                });
            });
        });
        /* Widżety złożone Django (data + godzina, filtr „dostępne / wybrane”): jedna etykieta
           wskazuje pole bez sufiksu — każda kontrolka dostaje nazwę z etykiety i dopiskiem. */
        var suffixes = { '_0': ' — data', '_1': ' — godzina', '_from': ' — dostępne', '_to': ' — wybrane' };
        document.querySelectorAll('#content input:not([type="hidden"]), #content select, #content textarea').forEach(function (field) {
            if ((field.labels && field.labels.length) || field.getAttribute('aria-label') || !field.id) return;
            var match = Object.keys(suffixes).find(function (s) { return field.id.slice(-s.length) === s; });
            if (!match) return;
            var label = document.querySelector('label[for="' + field.id.slice(0, -match.length) + '"]');
            if (!label) {
                var box = field.closest('.form-row, .field-line, fieldset > div, div');
                label = box && box.parentElement ? box.parentElement.querySelector('label') : null;
            }
            if (label) field.setAttribute('aria-label', label.textContent.replace(/[*:]/g, '').replace(/\s+/g, ' ').trim() + suffixes[match]);
        });
    }
    document.addEventListener('DOMContentLoaded', function () {
        fixLandmarks();
        new MutationObserver(fixLandmarks).observe(document.body, { childList: true, subtree: true });
        setTimeout(fixLandmarks, 0);
    });
}());
