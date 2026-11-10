/**
 * E-Learning Platform – Main JavaScript
 *
 * Provides:
 *   - CSRF token helper for fetch()
 *   - Mobile navigation toggle
 *   - Notification dropdown toggle
 */

/* -----------------------------------------------------------------------
   CSRF helper
   ----------------------------------------------------------------------- */

/**
 * Read the Django CSRF token from the meta tag.
 * @returns {string|null}
 */
function getCsrfToken() {
    const meta = document.querySelector('meta[name="csrf-token"]');
    return meta ? meta.getAttribute('content') : null;
}

/**
 * Convenience wrapper around fetch() that automatically includes the
 * CSRF token and JSON content-type headers.
 *
 * @param {string} url
 * @param {object} options - standard fetch options
 * @returns {Promise<Response>}
 */
function apiFetch(url, options = {}) {
    const headers = {
        'Content-Type': 'application/json',
        'X-CSRFToken': getCsrfToken(),
        ...(options.headers || {}),
    };

    return fetch(url, {
        credentials: 'same-origin',
        ...options,
        headers,
    });
}

/* -----------------------------------------------------------------------
   Mobile nav toggle & focus trap
   ----------------------------------------------------------------------- */

document.addEventListener('DOMContentLoaded', () => {
    const toggle = document.querySelector('.navbar__toggle');
    const menu = document.getElementById('primary-nav');

    if (toggle && menu) {
        const getFocusableInMenu = () => {
            return Array.from(
                menu.querySelectorAll(
                    'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
                )
            ).filter((el) => el.offsetParent !== null || window.getComputedStyle(el).display !== 'none');
        };

        const isMenuOpen = () => menu.classList.contains('is-open');

        const setMenuOpen = (open) => {
            menu.classList.toggle('is-open', open);
            toggle.setAttribute('aria-expanded', String(open));
            toggle.setAttribute(
                'aria-label',
                open ? 'Zamknij menu nawigacji' : 'Otwórz menu nawigacji',
            );

            if (open) {
                // Focus first interactive link in drawer on mobile
                const focusables = getFocusableInMenu();
                if (focusables.length > 0) {
                    focusables[0].focus();
                }
            } else {
                // Return focus to hamburger button
                if (menu.contains(document.activeElement)) {
                    toggle.focus();
                }
            }
        };

        toggle.addEventListener('click', () => {
            setMenuOpen(!isMenuOpen());
        });

        // Close menu when clicking outside
        document.addEventListener('click', (e) => {
            if (
                isMenuOpen() &&
                !menu.contains(e.target) &&
                !toggle.contains(e.target)
            ) {
                setMenuOpen(false);
            }
        });

        // Keyboard handler: Tab loop (focus trap) and Escape to exit
        document.addEventListener('keydown', (e) => {
            if (!isMenuOpen()) return;

            if (e.key === 'Escape') {
                e.preventDefault();
                setMenuOpen(false);
                toggle.focus();
                return;
            }

            if (e.key === 'Tab') {
                const focusables = getFocusableInMenu();
                if (!focusables.length) return;

                const firstEl = focusables[0];
                const lastEl = focusables[focusables.length - 1];

                if (e.shiftKey) {
                    // Shift + Tab on first element loops to last
                    if (document.activeElement === firstEl || document.activeElement === toggle) {
                        e.preventDefault();
                        lastEl.focus();
                    }
                } else {
                    // Tab on last element loops to first
                    if (document.activeElement === lastEl) {
                        e.preventDefault();
                        firstEl.focus();
                    }
                }
            }
        });

        // The menu is only a dropdown below the desktop breakpoint.
        const desktopQuery = window.matchMedia('(min-width: 64rem)');
        desktopQuery.addEventListener('change', (event) => {
            if (event.matches && isMenuOpen()) setMenuOpen(false);
        });
    }

    /* -------------------------------------------------------------------
       Notification dropdown toggle
       ------------------------------------------------------------------- */
    const notifBtn = document.querySelector('[data-notification-toggle]');
    if (notifBtn) {
        const dropdown = notifBtn
            .closest('.navbar__item--notifications')
            ?.querySelector('.navbar__notification-dropdown');

        if (dropdown) {
            notifBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                const isExpanded = notifBtn.getAttribute('aria-expanded') === 'true';
                notifBtn.setAttribute('aria-expanded', String(!isExpanded));
                dropdown.hidden = isExpanded;
                if (!dropdown.hidden) {
                    // Focus first focusable inside dropdown if any
                    const first = dropdown.querySelector('a[href], button:not([disabled])');
                    if (first) first.focus();
                }
            });

            // Close on outside click
            document.addEventListener('click', (e) => {
                if (!dropdown.contains(e.target) && !notifBtn.contains(e.target)) {
                    dropdown.hidden = true;
                    notifBtn.setAttribute('aria-expanded', 'false');
                }
            });

            // Close on Escape
            document.addEventListener('keydown', (e) => {
                if (e.key === 'Escape' && !dropdown.hidden) {
                    dropdown.hidden = true;
                    notifBtn.setAttribute('aria-expanded', 'false');
                    notifBtn.focus();
                }
            });

            // Fokus wyszedł z panelu (Tab dalej) — zamknij, żeby panel nie zasłaniał strony
            dropdown.addEventListener('focusout', (e) => {
                if (!dropdown.contains(e.relatedTarget) && e.relatedTarget !== notifBtn && e.relatedTarget) {
                    dropdown.hidden = true;
                    notifBtn.setAttribute('aria-expanded', 'false');
                }
            });
        }
    }

    /* -------------------------------------------------------------------
       Rozwijane menu w navbarze (<details>): motyw i konto – strzałki, Home/End,
       Esc (fokus wraca na przycisk), klik poza; otwarcie jednego zamyka drugie.
       ------------------------------------------------------------------- */
    const themeSwitchers = document.querySelectorAll('[data-theme-switcher], [data-account-menu]');

    if (themeSwitchers.length) {
        themeSwitchers.forEach((switcher) => {
            const summary = switcher.querySelector('summary');

            const getThemeFocusables = () => {
                return Array.from(
                    switcher.querySelectorAll('.theme-switcher__panel a[href], .theme-switcher__panel button:not([disabled]), .account-menu__panel a[href], .account-menu__panel button:not([disabled])')
                );
            };

            switcher.addEventListener('toggle', () => {
                if (!switcher.open) return;
                themeSwitchers.forEach((other) => {
                    if (other !== switcher) other.open = false;
                });
            });

            switcher.addEventListener('keydown', (event) => {
                if (!switcher.open) return;

                const focusables = getThemeFocusables();
                const currentIndex = focusables.indexOf(document.activeElement);

                if (event.key === 'Escape') {
                    event.preventDefault();
                    event.stopPropagation();
                    switcher.open = false;
                    if (summary) summary.focus();
                    return;
                }

                if (event.key === 'ArrowDown') {
                    event.preventDefault();
                    if (!focusables.length) return;
                    const nextIndex = currentIndex >= 0 ? (currentIndex + 1) % focusables.length : 0;
                    focusables[nextIndex].focus();
                    return;
                }

                if (event.key === 'ArrowUp') {
                    event.preventDefault();
                    if (!focusables.length) return;
                    const prevIndex = currentIndex > 0 ? currentIndex - 1 : focusables.length - 1;
                    focusables[prevIndex].focus();
                    return;
                }

                if (event.key === 'Home') {
                    event.preventDefault();
                    if (focusables.length) focusables[0].focus();
                    return;
                }

                if (event.key === 'End') {
                    event.preventDefault();
                    if (focusables.length) focusables[focusables.length - 1].focus();
                    return;
                }
            });
        });

        // Close theme switchers on outside click
        document.addEventListener('click', (event) => {
            themeSwitchers.forEach((switcher) => {
                if (switcher.open && !switcher.contains(event.target)) {
                    switcher.open = false;
                }
            });
        });
    }

    /* -------------------------------------------------------------------
       Toast Notifications (auto-dismiss, pause on hover, close button, Esc)
       ------------------------------------------------------------------- */
    const existingToasts = document.querySelectorAll('[data-toast]');
    existingToasts.forEach((toast) => {
        const durationAttr = toast.getAttribute('data-auto-dismiss');
        const duration = durationAttr !== null ? parseInt(durationAttr, 10) : 4500;
        setupToast(toast, isNaN(duration) ? 4500 : duration);
    });

    // Global Esc listener: pressing Esc anywhere on the page closes active toast notifications
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            const activeToasts = document.querySelectorAll('.toast:not(.is-hiding)');
            if (activeToasts.length > 0) {
                activeToasts.forEach((toast) => dismissToast(toast));
            }
        }
    });
});

/* -----------------------------------------------------------------------
   Global Toast Notification API
   ----------------------------------------------------------------------- */

const TOAST_ICONS = {
    success: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 6L9 17l-5-5"/></svg>',
    error: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>',
    danger: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>',
    warning: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
    info: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>',
};

/**
 * Dismiss a toast element with animation.
 * @param {HTMLElement} toast
 */
function dismissToast(toast) {
    if (!toast || toast.classList.contains('is-hiding')) return;
    toast.classList.add('is-hiding');
    const onEnd = () => {
        toast.remove();
    };
    toast.addEventListener('transitionend', onEnd, { once: true });
    // Fallback if transitionend does not fire
    setTimeout(() => {
        if (toast.isConnected) toast.remove();
    }, 350);
}

/**
 * Setup timer, interactions and close button on a toast element.
 * @param {HTMLElement} toast
 * @param {number} duration - Auto-dismiss timeout in ms (0 to disable)
 */
function setupToast(toast, duration = 4500) {
    const closeBtn = toast.querySelector('[data-toast-close]');
    if (closeBtn) {
        closeBtn.addEventListener('click', () => dismissToast(toast));
    }

    // Close on Escape if focused inside the toast
    toast.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            dismissToast(toast);
        }
    });

    if (duration > 0) {
        let timer = null;
        let remaining = duration;
        let startTime = Date.now();

        const startTimer = () => {
            startTime = Date.now();
            timer = setTimeout(() => dismissToast(toast), remaining);
        };

        const pauseTimer = () => {
            if (timer) {
                clearTimeout(timer);
                timer = null;
                remaining = Math.max(1000, remaining - (Date.now() - startTime));
            }
        };

        toast.addEventListener('mouseenter', pauseTimer);
        toast.addEventListener('mouseleave', startTimer);
        toast.addEventListener('focusin', pauseTimer);
        toast.addEventListener('focusout', startTimer);

        startTimer();
    }
}

/**
 * Global helper to show toast notifications programmatically.
 * Example usage: window.showToast("Pomyślnie zaktualizowano dane!", "success")
 *
 * @param {string} message
 * @param {'success'|'error'|'danger'|'warning'|'info'} [type='info']
 * @param {number} [duration=4500]
 * @returns {HTMLElement} The created toast element
 */
function showToast(message, type = 'info', duration = 4500) {
    let container = document.getElementById('toast-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toast-container';
        container.className = 'toast-container';
        container.setAttribute('role', 'region');
        container.setAttribute('aria-label', 'Powiadomienia systemowe');
        container.setAttribute('aria-live', 'polite');
        document.body.appendChild(container);
    }

    const toast = document.createElement('div');
    const safeType = ['success', 'error', 'danger', 'warning', 'info'].includes(type) ? type : 'info';
    toast.className = `toast toast--${safeType}`;
    toast.setAttribute('role', ['error', 'danger'].includes(safeType) ? 'alert' : 'status');
    toast.setAttribute('data-toast', '');

    const iconSvg = TOAST_ICONS[safeType] || TOAST_ICONS.info;

    toast.innerHTML = `
        <span class="toast__icon" aria-hidden="true">${iconSvg}</span>
        <div class="toast__content">
            <p class="toast__message"></p>
        </div>
        <button type="button" class="toast__close" aria-label="Zamknij powiadomienie" data-toast-close>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
        </button>
    `;

    toast.querySelector('.toast__message').textContent = message;

    container.appendChild(toast);
    setupToast(toast, duration);
    return toast;
}

// Global window registration
window.showToast = showToast;

/* -----------------------------------------------------------------------
   Accessible Modal & Dialog System (WCAG 2.1 AA / Usability Standard)
   - Traps Tab / Shift+Tab within the dialog
   - Closes on Escape and outside backdrop click
   - Restores focus to the triggering element upon close
   - Prevents background page scrolling while modal is open
   ----------------------------------------------------------------------- */

const modalStack = [];

function getModalFocusables(modal) {
    return Array.from(
        modal.querySelectorAll(
            'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
        )
    ).filter((el) => el.offsetParent !== null || window.getComputedStyle(el).display !== 'none');
}

/**
 * Open an accessible modal dialog.
 * @param {HTMLElement} modal
 * @param {HTMLElement} [triggerEl]
 */
function openAccessibleModal(modal, triggerEl = document.activeElement) {
    if (!modal) return;

    modalStack.push({ modal, triggerEl });
    document.body.classList.add('modal-open');

    if (modal.tagName === 'DIALOG' && typeof modal.showModal === 'function') {
        modal.showModal();
    } else {
        modal.removeAttribute('hidden');
        modal.classList.add('is-open');
    }

    // Move focus inside
    const focusables = getModalFocusables(modal);
    if (focusables.length > 0) {
        focusables[0].focus();
    } else {
        modal.setAttribute('tabindex', '-1');
        modal.focus();
    }
}

/**
 * Close an accessible modal dialog and restore previous focus.
 * @param {HTMLElement} [modal]
 */
function closeAccessibleModal(modal) {
    const entry = modal
        ? modalStack.find((m) => m.modal === modal)
        : modalStack.pop();

    if (!entry) return;

    if (modal) {
        const index = modalStack.indexOf(entry);
        if (index > -1) modalStack.splice(index, 1);
    }

    const { modal: targetModal, triggerEl } = entry;

    if (targetModal.tagName === 'DIALOG' && typeof targetModal.close === 'function') {
        if (targetModal.open) {
            targetModal.close();
        }
    } else {
        targetModal.setAttribute('hidden', '');
        targetModal.classList.remove('is-open');
    }

    if (modalStack.length === 0) {
        document.body.classList.remove('modal-open');
    }

    // Return focus to trigger
    if (triggerEl && typeof triggerEl.focus === 'function') {
        triggerEl.focus();
    }
}

// Global keyboard trap & Escape dismissal for modals
document.addEventListener('keydown', (e) => {
    if (modalStack.length === 0) return;

    const current = modalStack[modalStack.length - 1];
    const { modal } = current;

    if (e.key === 'Escape') {
        e.preventDefault();
        e.stopPropagation();
        closeAccessibleModal(modal);
        return;
    }

    if (e.key === 'Tab') {
        const focusables = getModalFocusables(modal);
        if (!focusables.length) {
            e.preventDefault();
            return;
        }

        const first = focusables[0];
        const last = focusables[focusables.length - 1];

        if (e.shiftKey) {
            if (document.activeElement === first || document.activeElement === modal) {
                e.preventDefault();
                last.focus();
            }
        } else {
            if (document.activeElement === last) {
                e.preventDefault();
                first.focus();
            }
        }
    }
});

// Auto-bind close buttons with [data-modal-close]
document.addEventListener('click', (e) => {
    const closeBtn = e.target.closest('[data-modal-close]');
    if (closeBtn) {
        const modal = closeBtn.closest('dialog, [role="dialog"], [data-modal]');
        if (modal) {
            closeAccessibleModal(modal);
        }
    }
});

/**
 * Display a custom accessible confirmation dialog using the active theme palette.
 * Resolves to true if user confirmed, false if cancelled/closed.
 *
 * @param {object} options
 * @param {string} [options.title='Potwierdzenie operacji']
 * @param {string} options.message
 * @param {string} [options.confirmText='Potwierdź']
 * @param {string} [options.cancelText='Anuluj']
 * @param {boolean} [options.isDanger=false]
 * @param {HTMLElement} [options.triggerEl]
 * @returns {Promise<boolean>}
 */
function showConfirmDialog({
    title = 'Potwierdzenie operacji',
    message = 'Czy na pewno chcesz wykonać tę akcję?',
    confirmText = 'Potwierdź',
    cancelText = 'Anuluj',
    isDanger = false,
    triggerEl = document.activeElement,
} = {}) {
    return new Promise((resolve) => {
        const dialog = document.getElementById('app-confirm-dialog');
        if (!dialog) {
            resolve(false);
            return;
        }

        const titleEl = document.getElementById('confirm-dialog-title');
        const msgEl = document.getElementById('confirm-dialog-message');
        const iconEl = dialog.querySelector('[data-confirm-icon]');
        const cancelBtn = dialog.querySelector('[data-confirm-cancel]');
        const acceptBtn = dialog.querySelector('[data-confirm-accept]');

        if (titleEl) titleEl.textContent = title;
        if (msgEl) msgEl.textContent = message;
        if (cancelBtn) cancelBtn.textContent = cancelText;

        if (acceptBtn) {
            acceptBtn.textContent = confirmText;
            if (isDanger) {
                acceptBtn.classList.remove('btn--primary');
                acceptBtn.classList.add('btn--danger');
            } else {
                acceptBtn.classList.remove('btn--danger');
                acceptBtn.classList.add('btn--primary');
            }
        }

        if (iconEl) {
            if (isDanger) {
                iconEl.classList.add('confirm-dialog__icon--danger');
                iconEl.classList.remove('confirm-dialog__icon--warning');
            } else {
                iconEl.classList.remove('confirm-dialog__icon--danger');
                iconEl.classList.add('confirm-dialog__icon--warning');
            }
        }

        let isResolved = false;

        const cleanup = (result) => {
            if (isResolved) return;
            isResolved = true;

            cancelBtn?.removeEventListener('click', onCancel);
            acceptBtn?.removeEventListener('click', onAccept);
            dialog.removeEventListener('cancel', onNativeCancel);
            dialog.removeEventListener('click', onBackdropClick);

            closeAccessibleModal(dialog);

            if (triggerEl && typeof triggerEl.focus === 'function') {
                triggerEl.focus();
            }

            resolve(result);
        };

        const onCancel = () => cleanup(false);
        const onAccept = () => cleanup(true);
        const onNativeCancel = (e) => {
            e.preventDefault();
            cleanup(false);
        };
        const onBackdropClick = (e) => {
            if (e.target === dialog) {
                cleanup(false);
            }
        };

        cancelBtn?.addEventListener('click', onCancel);
        acceptBtn?.addEventListener('click', onAccept);
        dialog.addEventListener('cancel', onNativeCancel);
        dialog.addEventListener('click', onBackdropClick);

        openAccessibleModal(dialog, triggerEl);

        // Focus the cancel button by default for safety
        if (cancelBtn) {
            cancelBtn.focus();
        }
    });
}

// Global declarative interception for forms with data-confirm
document.addEventListener('submit', async (e) => {
    const form = e.target;
    if (!(form instanceof HTMLFormElement)) return;
    const confirmMessage = form.getAttribute('data-confirm');
    if (!confirmMessage) return;

    if (form.dataset.confirmed === 'true') {
        delete form.dataset.confirmed;
        return;
    }

    e.preventDefault();
    const title = form.getAttribute('data-confirm-title') || 'Potwierdzenie operacji';
    const confirmText = form.getAttribute('data-confirm-btn') || 'Potwierdź';
    const cancelText = form.getAttribute('data-confirm-cancel-btn') || 'Anuluj';
    const isDanger = form.getAttribute('data-confirm-danger') === 'true';

    const triggerEl = form.querySelector('button[type="submit"]:focus') || form;

    const confirmed = await showConfirmDialog({
        title,
        message: confirmMessage,
        confirmText,
        cancelText,
        isDanger,
        triggerEl,
    });

    if (confirmed) {
        form.dataset.confirmed = 'true';
        form.submit();
    }
});

// Global declarative interception for buttons/links with data-confirm
document.addEventListener('click', async (e) => {
    const trigger = e.target.closest('button[data-confirm], a[data-confirm]');
    if (!trigger) return;

    if (trigger.type === 'submit' && trigger.form?.hasAttribute('data-confirm')) {
        return;
    }

    if (trigger.dataset.confirmed === 'true') {
        delete trigger.dataset.confirmed;
        return;
    }

    e.preventDefault();
    const title = trigger.getAttribute('data-confirm-title') || 'Potwierdzenie operacji';
    const message = trigger.getAttribute('data-confirm');
    const confirmText = trigger.getAttribute('data-confirm-btn') || 'Potwierdź';
    const cancelText = trigger.getAttribute('data-confirm-cancel-btn') || 'Anuluj';
    const isDanger = trigger.getAttribute('data-confirm-danger') === 'true';

    const confirmed = await showConfirmDialog({
        title,
        message,
        confirmText,
        cancelText,
        isDanger,
        triggerEl: trigger,
    });

    if (confirmed) {
        trigger.dataset.confirmed = 'true';
        if (trigger.tagName === 'A' && trigger.href) {
            window.location.href = trigger.href;
        } else if (trigger.type === 'submit' && trigger.form) {
            trigger.form.submit();
        } else if (typeof trigger.click === 'function') {
            trigger.click();
        }
    }
});

window.openAccessibleModal = openAccessibleModal;
window.closeAccessibleModal = closeAccessibleModal;
window.showConfirmDialog = showConfirmDialog;

/* -----------------------------------------------------------------------
   Accessible select (WAI-ARIA „select-only combobox”)
   Ulepsza <select data-select-enhance>: Tab → pole, Enter/Spacja/strzałki
   otwierają listę, strzałki/Home/End/litery wybierają, Enter zatwierdza,
   Esc zamyka bez zmiany, Tab zatwierdza i przechodzi dalej. Natywny
   <select> zostaje ukryty i nadal wysyła wartość w formularzu.
   ----------------------------------------------------------------------- */

function enhanceSelect(select) {
    if (select.dataset.selectEnhanced === 'true' || !select.id) return;
    select.dataset.selectEnhanced = 'true';

    const label = document.querySelector(`label[for="${select.id}"]`);
    if (label && !label.id) label.id = `${select.id}-label`;

    const wrapper = document.createElement('div');
    wrapper.className = 'select-box';
    select.parentNode.insertBefore(wrapper, select);

    const combo = document.createElement('div');
    combo.className = 'select-box__button';
    combo.id = `${select.id}-combo`;
    combo.tabIndex = 0;
    combo.setAttribute('role', 'combobox');
    combo.setAttribute('aria-haspopup', 'listbox');
    combo.setAttribute('aria-expanded', 'false');
    combo.setAttribute('aria-controls', `${select.id}-listbox`);
    if (label) combo.setAttribute('aria-labelledby', label.id);

    const valueEl = document.createElement('span');
    valueEl.className = 'select-box__value';
    combo.appendChild(valueEl);
    combo.insertAdjacentHTML(
        'beforeend',
        '<svg class="select-box__chevron" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M6 9l6 6 6-6"/></svg>',
    );

    const listbox = document.createElement('ul');
    listbox.className = 'select-box__list';
    listbox.id = `${select.id}-listbox`;
    listbox.setAttribute('role', 'listbox');
    listbox.tabIndex = -1;
    listbox.hidden = true;
    if (label) listbox.setAttribute('aria-labelledby', label.id);

    const options = Array.from(select.options).map((opt, index) => {
        const li = document.createElement('li');
        li.className = 'select-box__option';
        li.id = `${select.id}-opt-${index}`;
        li.setAttribute('role', 'option');
        li.textContent = opt.textContent.trim();
        listbox.appendChild(li);
        return li;
    });

    wrapper.append(combo, listbox, select);
    select.hidden = true;
    select.tabIndex = -1;
    select.setAttribute('aria-hidden', 'true');

    let activeIndex = select.selectedIndex;
    let typeBuffer = '';
    let typeTimer = null;

    const isOpen = () => combo.getAttribute('aria-expanded') === 'true';

    const render = () => {
        valueEl.textContent = options[select.selectedIndex]?.textContent || '';
        options.forEach((li, i) => {
            li.setAttribute('aria-selected', String(i === select.selectedIndex));
            li.classList.toggle('is-active', isOpen() && i === activeIndex);
        });
        if (isOpen() && options[activeIndex]) {
            combo.setAttribute('aria-activedescendant', options[activeIndex].id);
            options[activeIndex].scrollIntoView({ block: 'nearest' });
        } else {
            combo.removeAttribute('aria-activedescendant');
        }
    };

    const open = () => {
        if (isOpen()) return;
        activeIndex = select.selectedIndex;
        listbox.hidden = false;
        combo.setAttribute('aria-expanded', 'true');
        // Lista przy prawej krawędzi ekranu rozwija się w lewo
        listbox.classList.remove('select-box__list--end');
        if (listbox.getBoundingClientRect().right > document.documentElement.clientWidth - 8) {
            listbox.classList.add('select-box__list--end');
        }
        render();
    };

    const close = () => {
        if (!isOpen()) return;
        listbox.hidden = true;
        combo.setAttribute('aria-expanded', 'false');
        render();
    };

    const commit = (index) => {
        if (index < 0 || index >= options.length) return;
        if (select.selectedIndex !== index) {
            select.selectedIndex = index;
            select.dispatchEvent(new Event('change', { bubbles: true }));
        }
        activeIndex = index;
    };

    const moveTo = (index) => {
        activeIndex = Math.max(0, Math.min(options.length - 1, index));
        render();
    };

    const typeAhead = (char) => {
        clearTimeout(typeTimer);
        typeBuffer += char.toLocaleLowerCase('pl');
        typeTimer = setTimeout(() => { typeBuffer = ''; }, 600);
        const start = typeBuffer.length === 1 ? activeIndex + 1 : activeIndex;
        const ordered = options.map((_, i) => (start + i) % options.length);
        const match = ordered.find((i) => options[i].textContent.toLocaleLowerCase('pl').startsWith(typeBuffer));
        if (match !== undefined) {
            if (!isOpen()) open();
            moveTo(match);
        }
    };

    combo.addEventListener('keydown', (e) => {
        const { key } = e;

        if (!isOpen()) {
            if (['Enter', ' ', 'ArrowDown', 'ArrowUp'].includes(key)) {
                e.preventDefault();
                open();
            } else if (key === 'Home' || key === 'End') {
                e.preventDefault();
                open();
                moveTo(key === 'Home' ? 0 : options.length - 1);
            } else if (key.length === 1 && !e.ctrlKey && !e.metaKey && !e.altKey) {
                typeAhead(key);
            }
            return;
        }

        switch (key) {
            case 'ArrowDown':
                e.preventDefault();
                if (e.altKey) { commit(activeIndex); close(); } else { moveTo(activeIndex + 1); }
                break;
            case 'ArrowUp':
                e.preventDefault();
                if (e.altKey) { commit(activeIndex); close(); } else { moveTo(activeIndex - 1); }
                break;
            case 'Home':
                e.preventDefault();
                moveTo(0);
                break;
            case 'End':
                e.preventDefault();
                moveTo(options.length - 1);
                break;
            case 'PageDown':
                e.preventDefault();
                moveTo(activeIndex + 10);
                break;
            case 'PageUp':
                e.preventDefault();
                moveTo(activeIndex - 10);
                break;
            case 'Enter':
            case ' ':
                e.preventDefault();
                commit(activeIndex);
                close();
                break;
            case 'Escape':
                // Zamyka tylko listę — nie inne panele ani powiadomienia
                e.preventDefault();
                e.stopPropagation();
                close();
                break;
            case 'Tab':
                commit(activeIndex);
                close();
                break;
            default:
                if (key.length === 1 && !e.ctrlKey && !e.metaKey && !e.altKey) {
                    e.preventDefault();
                    typeAhead(key);
                }
        }
    });

    combo.addEventListener('click', () => {
        if (isOpen()) close(); else open();
    });

    // mousedown nie zabiera fokusu z pola — wybór myszą działa jak Enter
    listbox.addEventListener('mousedown', (e) => e.preventDefault());
    listbox.addEventListener('click', (e) => {
        const li = e.target.closest('[role="option"]');
        if (!li) return;
        commit(options.indexOf(li));
        close();
        combo.focus();
    });
    listbox.addEventListener('mousemove', (e) => {
        const li = e.target.closest('[role="option"]');
        if (li && options.indexOf(li) !== activeIndex) moveTo(options.indexOf(li));
    });

    combo.addEventListener('blur', () => close());

    // Etykieta <label for> wskazuje ukryty select — przekieruj kliknięcie na pole
    if (label) label.addEventListener('click', (e) => { e.preventDefault(); combo.focus(); });

    // Przywrócenie stanu formularza (np. przycisk Wstecz) synchronizuje widok
    select.addEventListener('change', render);

    render();
}

document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('select[data-select-enhance]').forEach(enhanceSelect);
});

/* -----------------------------------------------------------------------
   Rozwijane panele w miejscu (<details data-esc-close>): Esc zamyka panel
   i oddaje fokus na jego przycisk; kliknięcie poza też zamyka.
   ----------------------------------------------------------------------- */
document.addEventListener('keydown', (event) => {
    if (event.key !== 'Escape') return;
    const panel = document.activeElement?.closest('details[data-esc-close][open]');
    if (!panel) return;
    event.preventDefault();
    event.stopPropagation();
    panel.open = false;
    panel.querySelector('summary')?.focus();
}, true);

document.addEventListener('click', (event) => {
    document.querySelectorAll('details[data-esc-close][open]').forEach((panel) => {
        if (!panel.contains(event.target)) panel.open = false;
    });
});
