import re

import pytest

from src.utils import log_style as ls


def color(message):
    return ls.classify(message).color


P = ls.PALETTE


@pytest.mark.parametrize("message,expected", [
    # real lines from the application log
    ("🎙️ Transcribing audio [Lang: ru, Model: large-v3-turbo] and analyzing sound patterns...", "audio"),
    ("🎧 YAMNet finished on GPU / CUDA (separate process) in 3.7s: 15 laughter and 2 scream event(s) found.", "audio"),
    ("🔊 Audio events stored in the cache: LAUGHTER: 15, LOUD: 22, SCREAM: 2 (sound classifier not re-run).", "audio"),
    ("⬇️ Downloading the YAMNet sound-event model (16 MB, one time)...", "audio"),
    ("♻️ Found an older cached transcript (649 segments) - reusing the Whisper text...", "audio"),
    ("🤖 Routing to DeepSeek Engine (deepseek-v4-flash) with 999 segments...", "ai"),
    ("🧠 DeepSeek reasoning mode: ON (slower, better decisions), 4 requests in parallel.", "ai"),
    ("🏷️ Audio event tags included in the AI prompts: LAUGHTER: 15.", "ai"),
    ("🎯 Window 1/8: 5 candidate(s) (34s)", "ai"),
    ("⚖️ Re-ranking 39 candidates on a common scale...", "ai"),
    ("🔍 Clip 3: 486-597s -> 487-594s (setup needed)", "ai"),
    ("🎚️ Selected 18 of 39 candidate(s) (up to 18 at 12/hour). Dropped: 21 below score 6.", "select"),
    ("📊 Extracted approx 9,458 tokens (7,276 words) in 8 window(s) for the AI model.", "select"),
    ("📏 Clip lengths: from the AI median 48s -> final median 53s (min 10s, max 88s).", "select"),
    ("✂️ Placed 36 of 36 clip boundaries inside speech pauses (no cut words).", "video"),
    ("🎬 Sending 18 total timestamp(s) to FFmpeg...", "video"),
    ("🚀 FFmpeg Video Processing: GPU Hardware Acceleration active (NVENC/CUDA)", "video"),
    ("🎞️ [1/18] Cutting Clip (244.3s to 302.1s, 57.8s) -> Score 6/10", "video"),
    ("✅ [1/18] Rendered: 0918_clip_1.mp4", "success"),
    ("⚡ Loaded 999 cached audio segments from disk! (Skipping Whisper transcription)", "success"),
    ("💾 Cached transcription to disk (999 segments).", "success"),
    ("✨ Successfully exported 18 file(s) to your folder! (total 5m 40s)", "success"),
    ("⚠️ DeepSeek API Error (window 1/8, Attempt 1/3): timeout. Retrying...", "warn"),
    ("🤷‍♂️ Window 4/8: no candidates (12s)", "warn"),
    ("🛑 Cancelled - not waiting for the remaining AI answers.", "warn"),
    ("❌ DeepSeek API Error (re-ranking): 401", "error"),
    ("🧠 Hardware Check: NVIDIA GPU Detected (NVIDIA GeForce RTX 4080 (16.0 GB VRAM))", "info"),
    ("🚀 Processing Mode: GPU (CUDA fp16 acceleration active - maximum speed)", "info"),
    ("⏳ Processed up to 03:15.460: ну да я не я же не мой бы слышал...", "dim"),
    ("⏳ DeepSeek is still thinking... 7/8 windows answered (79s)", "ai_dim"),
    ("Something unexpected happened", "text"),
    ("Unhandled error while doing a thing", "error"),
])
def test_every_kind_of_message_gets_its_own_colour(message, expected):
    assert ls.classify(message).kind == expected
    assert color(message) == P[expected]


def test_palette_is_valid_and_the_main_categories_are_visually_distinct():
    assert all(re.fullmatch(r"#[0-9a-f]{6}", value) for value in P.values())
    main = ["audio", "ai", "select", "video", "success", "warn", "error", "info"]
    assert len({P[k] for k in main}) == len(main)


def test_stage_header_text_and_colour_follow_the_stage_icon():
    header = ls.stage_header(2, 4, "🤖", "AI finds and ranks the best moments")
    assert header == "══ 2/4 · 🤖 AI finds and ranks the best moments ══"
    style = ls.classify(header)
    assert style.kind == "header" and style.bold and style.top_level and style.color == P["ai"]
    assert ls.classify(ls.stage_header(1, 4, "🎙️", "Transcript & audio")).color == P["audio"]
    assert ls.classify(ls.stage_header(4, 4, "✂️", "Cutting & exporting")).color == P["video"]
    assert ls.classify("══ 9/9 · plain ══").color == P["text"]


def test_run_level_lines_are_top_level_and_details_are_not():
    for message in ("🎬 Starting batch highlight extraction (1 file(s))...", "▶️ [1/1] Processing: 0918.mp4", "✨ All queued videos processed successfully!"):
        assert ls.classify(message).top_level
    assert ls.classify("▶️ [1/1] Processing: 0918.mp4").bold
    assert ls.classify("▶️ [1/1] Processing: 0918.mp4").color == P["file"]
    assert not ls.classify("🎯 Window 1/8: 5 candidate(s) (34s)").top_level
    assert ls.classify("━━━━━━━━━━━━━━━━").kind == "rule"


def test_html_structure_detail_lines_get_a_bar_headers_and_run_lines_do_not():
    detail = ls.to_html("🎯 Window 1/8: 5 candidate(s) (34s)", "05:56:38")
    assert "[05:56:38]" in detail and "│" in detail and P["ai"] in detail
    header = ls.to_html(ls.stage_header(1, 4, "🎙️", "Transcript & audio"), "05:56:38")
    assert "│" not in header and "font-weight:700" in header and "[05:56:38]" in header
    top = ls.to_html("▶️ [1/1] Processing: 0918.mp4", "05:56:38")
    assert "│" not in top and "font-weight:700" in top
    assert ls.to_html("━━━━", "05:56:38").count("[05:56:38]") == 0  # the rule is a clean line


def test_html_escapes_transcript_text_and_keeps_existing_timestamps():
    line = ls.to_html("⏳ Processed up to 01:02: <b>bold</b> & co > 3", "05:56:38")
    assert "&lt;b&gt;bold&lt;/b&gt; &amp; co &gt; 3" in line and "<b>" not in line
    stamped = ls.to_html("[2026-09-19 05:00:00] 🎯 already timestamped", "05:56:38")
    assert "05:56:38" not in stamped and "[2026-09-19 05:00:00]" in stamped
