"""Pure (no network / no I/O) logic around highlight candidates.

Pipeline position:  transcript + audio events -> windows -> [LLM candidates] -> dedupe
                    -> density-based selection -> boundary refinement -> FFmpeg
"""

import math
import re
from typing import Any, Dict, List, Optional, Tuple

WINDOW_SECONDS = 720.0  # 12 min of transcript per LLM request
WINDOW_OVERLAP = 60.0  # context overlap so moments on a boundary are seen whole
SINGLE_WINDOW_LIMIT = 1.3  # videos up to 1.3 windows long are sent in one request

DEFAULT_CLIPS_PER_HOUR = 12
DEFAULT_MIN_SCORE = 6
OVERSAMPLE = 2.0  # ask the LLM for ~2x the target, selection trims it down

MIN_CLIP_SECONDS = 5.0
MAX_CLIP_SECONDS = 240.0
MIN_CONTEXT_SECONDS = 20.0  # a punchline clip shorter than this almost never carries the setup that makes it understandable
MAX_CONTEXT_EXTENSION = 20.0
SAFE_MAX_CLIP_SECONDS = MAX_CLIP_SECONDS - 8.0  # leaves room for the small pause-snapping shifts that come later
MIN_CLIP_GAP = 8.0  # min unclipped seconds between two selected clips

EXPORT_PAD = 0.5  # minimum gap kept between neighbouring clips after refinement
PEAK_LEAD = 4.0  # peak_time must have this much setup before it...
PEAK_TAIL = 2.5  # ...and this much reaction after it
MAX_SNAP_BACK = 2.0  # phrase snapping is only a coarse first step, cuts.py places the real boundary in a pause
MAX_SNAP_FORWARD = 2.0
MAX_LAUGH_EXTEND = 8.0
LOUDNESS_TAG_MIN = 60  # only ~15% of lines rise 12+ dB above the local level; lower values tag half the transcript
LONG_SEGMENT = 12.0  # segments longer than this are snapped at word level instead


# --------------------------------------------------------------------------------------
# Transcript formatting & windowing
# --------------------------------------------------------------------------------------
def format_entry(entry: Dict[str, Any]) -> str:
    """One transcript line. Segments carry [LOUDNESS]/[ACTION] tags, events are their own lines."""
    start, end = entry["start"], entry["end"]
    if entry.get("event"):
        return f"[{start:.1f}s - {end:.1f}s] [{entry['event']} {entry.get('strength', 0)}%]\n"
    tags = []
    loudness = entry.get("loudness")
    if loudness is not None and loudness >= LOUDNESS_TAG_MIN:
        tags.append(f"[LOUDNESS: {int(loudness)}%]")
    if entry.get("is_combat"):
        tags.append("[ACTION: COMBAT]")
    text = str(entry.get("text", "")).strip()
    prefix = " ".join(tags)
    return f"[{start:.1f}s - {end:.1f}s] {prefix + ' ' if prefix else ''}{text}\n"


def build_windows(entries: List[Dict[str, Any]], window: float = WINDOW_SECONDS, overlap: float = WINDOW_OVERLAP) -> List[Dict[str, Any]]:
    """Splits a timeline into overlapping windows. Each window: {start, end, lines: [entry,...]}."""
    if not entries:
        return []
    total_start = min(e["start"] for e in entries)
    total_end = max(e["end"] for e in entries)
    if total_end - total_start <= window * SINGLE_WINDOW_LIMIT:
        return [{"start": total_start, "end": total_end, "lines": list(entries)}]

    windows = []
    step = window - overlap
    cursor = total_start
    while cursor < total_end:
        w_end = min(cursor + window, total_end)
        if total_end - w_end <= overlap:  # a tail that is mostly overlap is absorbed instead of costing a request
            w_end = total_end
        lines = [e for e in entries if e["end"] > cursor and e["start"] < w_end]
        if lines:
            windows.append({"start": cursor, "end": w_end, "lines": lines})
        if w_end >= total_end:
            break
        cursor += step
    return windows


def target_clip_count(duration: float, clips_per_hour: float) -> Optional[int]:
    """Number of clips to keep for a video; None = unlimited (setting <= 0)."""
    if clips_per_hour <= 0:
        return None
    return max(1, int(math.ceil(duration / 3600.0 * clips_per_hour)))


# --------------------------------------------------------------------------------------
# Candidate merging & selection
# --------------------------------------------------------------------------------------
def _overlap_ratio(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    inter = min(a["end_time"], b["end_time"]) - max(a["start_time"], b["start_time"])
    if inter <= 0:
        return 0.0
    shorter = min(a["end_time"] - a["start_time"], b["end_time"] - b["start_time"])
    return inter / shorter if shorter > 0 else 0.0


def dedupe_candidates(candidates: List[Dict[str, Any]], min_overlap: float = 0.4) -> List[Dict[str, Any]]:
    """Collapses the same moment found by two overlapping windows: keeps the higher-scored clip and
    widens it to the union so neither window's framing is lost."""
    ordered = sorted(candidates, key=lambda c: (-c.get("virality_score", 0), c["start_time"]))
    kept: List[Dict[str, Any]] = []
    for cand in ordered:
        twin = next((k for k in kept if _overlap_ratio(k, cand) >= min_overlap), None)
        if twin is None:
            kept.append(dict(cand))
            continue
        union_start = min(twin["start_time"], cand["start_time"])
        union_end = max(twin["end_time"], cand["end_time"])
        if union_end - union_start <= SAFE_MAX_CLIP_SECONDS:  # never widen a moment beyond the absolute maximum
            twin["start_time"], twin["end_time"] = union_start, union_end
    return sorted(kept, key=lambda c: c["start_time"])


def select_clips(candidates: List[Dict[str, Any]], duration: float, clips_per_hour: float = DEFAULT_CLIPS_PER_HOUR,
                 min_score: int = DEFAULT_MIN_SCORE, min_gap: float = MIN_CLIP_GAP,
                 stats: Optional[Dict[str, int]] = None) -> List[Dict[str, Any]]:
    """Keeps the best non-overlapping candidates until the density target is reached.
    `stats`, when given, is filled with why candidates were dropped (low_score, too_short, too_close, over_limit) and the limit."""
    limit = target_clip_count(duration, clips_per_hour)
    counts = {"low_score": 0, "too_short": 0, "too_close": 0, "over_limit": 0, "limit": limit if limit is not None else -1}
    pool = []
    for c in candidates:
        if c.get("virality_score", 0) < min_score:
            counts["low_score"] += 1
        elif c["end_time"] - c["start_time"] < MIN_CLIP_SECONDS:
            counts["too_short"] += 1
        else:
            pool.append(c)
    pool.sort(key=lambda c: (-c["virality_score"], c["end_time"] - c["start_time"], c["start_time"]))

    chosen: List[Dict[str, Any]] = []
    for cand in pool:
        if limit is not None and len(chosen) >= limit:
            counts["over_limit"] += 1
            continue
        clash = any(cand["start_time"] < c["end_time"] + min_gap and cand["end_time"] > c["start_time"] - min_gap
                    for c in chosen)
        if clash:
            counts["too_close"] += 1
        else:
            chosen.append(cand)
    if stats is not None:
        stats.update(counts)
    return sorted(chosen, key=lambda c: c["start_time"])


def extend_short_clip(clip: Dict[str, Any], entries: List[Dict[str, Any]], min_len: float = MIN_CONTEXT_SECONDS,
                      max_extend: float = MAX_CONTEXT_EXTENSION) -> Dict[str, Any]:
    """Adds lead-in context to a clip that is too short to be understood on its own: the start moves back to the
    nearest earlier utterance start that makes the clip at least `min_len` long (never more than `max_extend` back)."""
    start, end = float(clip["start_time"]), float(clip["end_time"])
    if end - start >= min_len:
        return clip
    wanted = end - min_len
    starts = [float(e["start"]) for e in entries if not e.get("event") and start - max_extend <= float(e["start"]) <= wanted]
    new_start = max(starts) if starts else max(wanted, start - max_extend)
    new_start = max(0.0, min(new_start, start))
    if new_start >= start:
        return clip
    return dict(clip, start_time=round(new_start, 2))


def limit_length(clip: Dict[str, Any], max_len: float = SAFE_MAX_CLIP_SECONDS) -> Dict[str, Any]:
    """Hard cap for clips that came out too long (a model ignoring the maximum, unions, extensions).
    Keeps the punchline (peak_time) inside, otherwise keeps the beginning."""
    start, end = float(clip["start_time"]), float(clip["end_time"])
    if end - start <= max_len:
        return clip
    peak = clip.get("peak_time")
    if peak is not None and start <= float(peak) <= end:
        new_start = max(start, min(float(peak) - 0.4 * max_len, end - max_len))
    else:
        new_start = start
    return dict(clip, start_time=round(new_start, 2), end_time=round(new_start + max_len, 2))


def requested_candidates(window_len: float, clips_per_hour: float) -> int:
    """How many candidates to ask the LLM for in a window (oversampled; unlimited setting -> 6/10min)."""
    per_hour = clips_per_hour if clips_per_hour > 0 else 36.0
    return max(2, int(math.ceil(window_len / 3600.0 * per_hour * OVERSAMPLE)))


# --------------------------------------------------------------------------------------
# Boundary refinement
# --------------------------------------------------------------------------------------
def _speech_spans(segments: List[Dict[str, Any]]) -> List[Tuple[float, float]]:
    """Phrase-level spans; long Whisper segments are broken into word-level runs (split at pauses > 0.35 s)."""
    spans: List[Tuple[float, float]] = []
    for seg in segments:
        if seg.get("event"):
            continue
        s, e = float(seg["start"]), float(seg["end"])
        words = [w for w in (seg.get("words") or []) if "start" in w and "end" in w]
        if e - s <= LONG_SEGMENT or not words:
            spans.append((s, e))
            continue
        run_start, prev_end = float(words[0]["start"]), float(words[0]["end"])
        for w in words[1:]:
            if float(w["start"]) - prev_end > 0.35:
                spans.append((run_start, prev_end))
                run_start = float(w["start"])
            prev_end = float(w["end"])
        spans.append((run_start, prev_end))
    return spans


def refine_clip(clip: Dict[str, Any], segments: List[Dict[str, Any]], duration: float) -> Dict[str, Any]:
    """Snaps a clip to speech boundaries, guarantees setup before / reaction after the peak, and
    lets laughter finish. Never shortens a clip below what the LLM asked for except by snapping
    a boundary that landed inside a phrase."""
    spans = _speech_spans(segments)
    events = [s for s in segments if s.get("event")]
    start, end = float(clip["start_time"]), float(clip["end_time"])

    peak = clip.get("peak_time")
    if peak is not None:
        peak = float(peak)
        if start <= peak <= end + PEAK_TAIL:
            start = min(start, peak - PEAK_LEAD)
            end = max(end, peak + PEAK_TAIL)

    for s, e in spans:
        if s < start < e:
            if start - s <= MAX_SNAP_BACK:
                start = s
            elif e - start <= MAX_SNAP_FORWARD:
                start = e
            break
    for s, e in spans:
        if s < end < e:
            if e - end <= MAX_SNAP_FORWARD:
                end = e
            elif end - s <= MAX_SNAP_BACK:
                end = s
            break

    # laughter still going at the end (or starting right after the last line) is the payoff: let it play out
    for ev in events:
        if ev["event"] != "LAUGHTER":
            continue
        if (ev["start"] <= end <= ev["end"] + 0.5 and ev["end"] > end) or end < ev["start"] <= end + 2.0:
            end = min(ev["end"], end + MAX_LAUGH_EXTEND)
            break

    start = max(0.0, start)
    end = min(duration, end)

    refined = dict(clip)
    refined["start_time"] = round(start, 2)
    refined["end_time"] = round(max(end, start + MIN_CLIP_SECONDS), 2)
    return refined


def resolve_overlaps(clips: List[Dict[str, Any]], pad: float = EXPORT_PAD) -> List[Dict[str, Any]]:
    """Refinement can push neighbouring clips into each other; keep `pad` seconds between them, trimming the earlier clip's end, or the later clip's start if that would
    leave the earlier one too short."""
    ordered = sorted((dict(c) for c in clips), key=lambda c: c["start_time"])
    for prev, nxt in zip(ordered, ordered[1:]):
        limit = nxt["start_time"] - pad
        if prev["end_time"] <= limit:
            continue
        if limit - prev["start_time"] >= MIN_CLIP_SECONDS:
            prev["end_time"] = round(limit, 2)
        else:
            nxt["start_time"] = round(prev["end_time"] + pad, 2)
    return [c for c in ordered if c["end_time"] - c["start_time"] >= MIN_CLIP_SECONDS]


# --------------------------------------------------------------------------------------
# Utterances: transcript lines that follow real speech pauses instead of Whisper's chunking
# --------------------------------------------------------------------------------------
UTTERANCE_GAP = 0.3  # a silence this long between two Whisper segments is a real pause
UTTERANCE_MAX = 14.0
_SENTENCE_END = re.compile(r"[.?!…]\s*$")


def build_utterances(segments: List[Dict[str, Any]], gap: float = UTTERANCE_GAP, max_len: float = UTTERANCE_MAX) -> List[Dict[str, Any]]:
    """Joins Whisper segments that are one continuous stretch of speech.

    Whisper splits arbitrarily: a phrase can be cut between two lines ("... от 1 от" / "0 до 9 ..."), which makes any
    clip boundary chosen from those lines logically ragged. Two neighbouring segments are merged when there is no real
    pause between them (< `gap` s of silence) AND the first one does not end a sentence with . ? ! ... Word
    timestamps are useless for this (93% of word gaps are exactly zero), segment gaps and punctuation are not.
    Audio events pass through unchanged and never merge anything."""
    out: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    for seg in sorted(segments, key=lambda e: (e["start"], 0 if e.get("event") else 1)):
        if seg.get("event"):
            out.append(dict(seg))
            continue
        if current is not None:
            silence = float(seg["start"]) - float(current["end"])
            joins = (silence < gap and not _SENTENCE_END.search(str(current["text"]))
                     and float(seg["end"]) - float(current["start"]) <= max_len)
            if joins:
                current["end"] = max(float(current["end"]), float(seg["end"]))
                current["text"] = f"{str(current['text']).strip()} {str(seg['text']).strip()}"
                if seg.get("loudness") is not None:
                    current["loudness"] = max(current.get("loudness", 0), seg["loudness"])
                current["is_combat"] = bool(current.get("is_combat") or seg.get("is_combat"))
                current["words"] = list(current.get("words") or []) + list(seg.get("words") or [])
                continue
            out.append(current)
        current = dict(seg)
        current["words"] = list(seg.get("words") or [])
        current["text"] = str(seg.get("text", "")).strip()
    if current is not None:
        out.append(current)
    for entry in out:
        if not entry.get("words"):
            entry.pop("words", None)
    return sorted(out, key=lambda e: (e["start"], 0 if e.get("event") else 1))
