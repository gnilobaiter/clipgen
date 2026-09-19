"""Colours and structure for the console log.

The pipeline only sends plain strings to its logger. Every message starts with an emoji that says what it is
about, and this module turns that into a consistent look:

  ══ 1/4 · 🎙️ Transcript & audio ══      stage header (bold, coloured by the stage's emoji)
  [05:56:38] │ 🎯 Window 1/8: 5 ...       detail line: dim time, a "│" that ties it to its stage, coloured text

Colours mean the same everywhere: cyan = audio, violet = AI, teal = selection/statistics, orange = video cutting,
green = success, amber = warning/cancel, red = error, grey = background chatter (Whisper progress).
"""

import html
from typing import NamedTuple, Optional

STAGE_MARK = "══"

PALETTE = {
    "audio": "#22d3ee",
    "ai": "#a78bfa",
    "ai_dim": "#8b7fc7",
    "select": "#2dd4bf",
    "video": "#fb923c",
    "success": "#34d399",
    "warn": "#fbbf24",
    "error": "#f87171",
    "info": "#38bdf8",
    "dim": "#64748b",
    "text": "#e2e8f0",
    "file": "#f8fafc",
    "rule": "#334155",
}

# first emoji of a message -> category
EMOJI_CATEGORY = [
    ("❌", "error"),
    ("⚠️", "warn"), ("🤷", "warn"), ("🛑", "warn"),
    ("✅", "success"), ("✨", "success"), ("🏁", "success"), ("💾", "success"), ("⚡", "success"),
    ("🎙️", "audio"), ("🎧", "audio"), ("🔊", "audio"), ("⬇️", "audio"), ("♻️", "audio"),
    ("🤖", "ai"), ("🧠", "ai"), ("🎯", "ai"), ("🏷️", "ai"), ("⚖️", "ai"), ("🔍", "ai"), ("📨", "ai"),
    ("🎚️", "select"), ("📊", "select"), ("📏", "select"),
    ("🎬", "video"), ("🎞️", "video"), ("✂️", "video"), ("🚀", "video"), ("📸", "video"), ("📱", "video"),
    ("⏳", "dim"),
]

# lines that belong to the level of the file / whole run, not to a stage
TOP_LEVEL_PREFIXES = ("🎬 Starting batch", "▶️ [", "✨ All queued", "🛑 Batch processing", "🛑 Processing cancelled", "⚠️ Batch completed",
                      "✨ Successfully exported")
BOLD_PREFIXES = ("▶️ [", "🎬 Starting batch", "✨ All queued", "✨ Successfully exported")
HARDWARE_WORDS = ("Hardware Check", "Processing Mode", "Active Hardware Engine")


class Style(NamedTuple):
    color: str
    bold: bool
    top_level: bool
    kind: str


def stage_header(index: int, total: int, icon: str, title: str) -> str:
    """The text of a stage header line, e.g. '══ 2/4 · 🤖 AI finds the best moments ══'."""
    return f"{STAGE_MARK} {index}/{total} · {icon} {title} {STAGE_MARK}"


def _category(message: str) -> Optional[str]:
    for emoji, category in EMOJI_CATEGORY:
        if message.startswith(emoji):
            return category
    return None


def classify(message: str) -> Style:
    m = message.strip()
    if m.startswith(STAGE_MARK):
        category = next((cat for emoji, cat in EMOJI_CATEGORY if emoji in m), None)
        return Style(PALETTE.get(category or "text", PALETTE["text"]), True, True, "header")
    if m.startswith("━"):
        return Style(PALETTE["rule"], False, True, "rule")

    top = m.startswith(TOP_LEVEL_PREFIXES)
    bold = m.startswith(BOLD_PREFIXES)
    category = _category(m)

    if m.startswith(("▶️ [", "🎬 Starting batch")):
        category = "file"
    elif any(word in m for word in HARDWARE_WORDS):
        category = "info"
    elif "Processed up to" in m:
        category = "dim"
    elif "is still thinking" in m:
        category = "ai_dim"
    elif category is None and "error" in m.lower():
        category = "error"
    return Style(PALETTE.get(category or "text", PALETTE["text"]), bold, top, category or "text")


def to_html(message: str, clock: str) -> str:
    """One console line as HTML. The message is escaped (transcript text may contain < and &)."""
    m = message.strip()
    has_time = m.startswith("[") and "]" in m[:25]
    style = classify(m)
    text = html.escape(m, quote=False)
    weight = "font-weight:700;" if style.bold else ""
    body = f'<span style="color:{style.color};{weight}white-space:pre-wrap;">{text}</span>'
    stamp = f'<span style="color:{PALETTE["dim"]};">[{clock}]</span>'
    if style.kind == "rule":
        return body
    if style.kind == "header":  # an empty line above each stage so the stages read as separate blocks
        return f'<br>{body if has_time else f"{stamp} {body}"}'
    if has_time:
        return body
    if style.top_level:
        return f"{stamp} {body}"
    return f'{stamp} <span style="color:{PALETTE["rule"]};">│</span> {body}'
