"""Bezpieczne wyświetlanie odpowiedzi modelu: najpierw escape, potem kilka elementów Markdownu
(bloki kodu, `kod`, **pogrubienie**, listy, akapity). Żaden HTML z modelu nie trafia na stronę."""

from __future__ import annotations

import re

from django.utils.html import escape
from django.utils.safestring import SafeString, mark_safe

FENCE = re.compile(r"```[ \t]*([\w+#.-]*)[ \t]*\n?(.*?)(?:```|\Z)", re.S)
INLINE_CODE = re.compile(r"`([^`\n]+)`")
BOLD = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*")
LIST_ITEM = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(.*)$")
HEADING = re.compile(r"^\s*#{1,6}\s+(.*)$")


def _inline(text: str) -> str:
    text = INLINE_CODE.sub(r"<code>\1</code>", text)
    return BOLD.sub(r"<strong>\1</strong>", text)


def _blocks(text: str) -> list[str]:
    html: list[str] = []
    for chunk in re.split(r"\n\s*\n", text.strip()):
        lines = [line for line in chunk.split("\n") if line.strip()]
        if not lines:
            continue
        items = [LIST_ITEM.match(line) for line in lines]
        if all(items):
            ordered = lines[0].lstrip()[0].isdigit()
            tag = "ol" if ordered else "ul"
            body = "".join(f"<li>{_inline(m.group(1))}</li>" for m in items)
            html.append(f"<{tag}>{body}</{tag}>")
            continue
        heading = HEADING.match(lines[0])
        if heading and len(lines) == 1:
            html.append(f"<p><strong>{_inline(heading.group(1))}</strong></p>")
            continue
        html.append("<p>" + "<br>".join(_inline(HEADING.sub(r"\1", line)) for line in lines) + "</p>")
    return html


def render_reply(text: str) -> SafeString:
    """Odpowiedź asystenta jako bezpieczny HTML."""
    escaped = escape(text or "")
    parts: list[str] = []
    position = 0
    for match in FENCE.finditer(escaped):
        parts.extend(_blocks(escaped[position:match.start()]))
        language, code = match.group(1), match.group(2).rstrip("\n")
        label = f' data-language="{language}"' if language else ""
        # tabindex: przewijany w poziomie blok kodu musi być osiągalny klawiaturą
        parts.append(f'<pre{label} tabindex="0"><code>{code}</code></pre>')
        position = match.end()
    parts.extend(_blocks(escaped[position:]))
    return mark_safe("".join(parts))
