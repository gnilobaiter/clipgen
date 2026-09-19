"""Cut-point placement on the waveform: clip starts/ends go *inside pauses*, never inside words.

Whisper timestamps (segment and even word level) drift by hundreds of milliseconds, and a fixed pre/post-roll
lands wherever it lands, often in the middle of a neighbouring word. Systems that clip long recordings
therefore treat the ASR/LLM boundary only as a hint and move it to the quietest point of a nearby silence gap
(VAD-style). This module does that with a plain energy envelope: no extra model, works on game audio too.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

SAMPLE_RATE = 16000
FRAME = SAMPLE_RATE // 100  # 10 ms
SMOOTH_FRAMES = 3
MIN_PAUSE = 0.10  # s; shorter dips are consonant closures inside words, not pauses
REAL_PAUSE = 0.30  # s; a pause that can end a phrase (shorter ones are usually breaths inside continuing speech)
QUIET_FRACTION = 0.30  # a frame is "quiet" below floor + 30% of the way up to the speech level
MIN_CONTRAST_DB = 8.0  # windows without this much speech/quiet contrast cannot be snapped
SEARCH_TIERS = (1.6, 3.5)  # s to look earlier than the target for a start cut / later for an end cut; wider on retry
SEARCH_SLACK = 0.3  # s to look on the "wrong" side of the target
START_LEAD = 0.12  # a start cut sits this long before the first word
END_TAIL = 0.18  # an end cut sits this long after the last word
WINDOW_MARGIN = 1.5  # audio read beyond the search range so the loudness statistics see real speech
UNSNAPPED_PAD = 0.2  # tiny safety margin on a side that could not be snapped


def frame_db(audio: np.ndarray) -> np.ndarray:
    """Smoothed level in dB per 10 ms frame."""
    n = len(audio) // FRAME
    if n == 0:
        return np.zeros(0)
    frames = np.asarray(audio[: n * FRAME], dtype=np.float64).reshape(n, FRAME)
    db = 20.0 * np.log10(np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-7)
    kernel = np.ones(SMOOTH_FRAMES) / SMOOTH_FRAMES
    return np.convolve(np.pad(db, SMOOTH_FRAMES // 2, mode="edge"), kernel, mode="valid")


def find_pauses(db: np.ndarray) -> List[Tuple[int, int]]:
    """(first_frame, last_frame) of every quiet run of at least MIN_PAUSE."""
    if len(db) == 0:
        return []
    speech = float(np.percentile(db, 85))
    floor = float(np.percentile(db, 10))
    if speech - floor < MIN_CONTRAST_DB:
        return []
    quiet = db <= floor + QUIET_FRACTION * (speech - floor)
    pauses, run_start = [], None
    for i, q in enumerate(quiet):
        if q and run_start is None:
            run_start = i
        elif not q and run_start is not None:
            if (i - run_start) * 0.01 >= MIN_PAUSE:
                pauses.append((run_start, i - 1))
            run_start = None
    if run_start is not None and (len(quiet) - run_start) * 0.01 >= MIN_PAUSE:
        pauses.append((run_start, len(quiet) - 1))
    return pauses


def find_cut(audio: np.ndarray, window_start: float, target: float, kind: str, reach: float = SEARCH_TIERS[0],
             min_pause: float = MIN_PAUSE) -> Optional[float]:
    """Absolute time of the best cut near `target` (kind = "start" or "end"), or None if no pause is nearby.

    `audio` holds the samples that begin at absolute time `window_start`."""
    pauses = find_pauses(frame_db(audio))
    if not pauses:
        return None
    rel = target - window_start
    lo, hi = (rel - reach, rel + SEARCH_SLACK) if kind == "start" else (rel - SEARCH_SLACK, rel + reach)

    best, best_score = None, None
    for first, last in pauses:
        p_start, p_end = first * 0.01, (last + 1) * 0.01
        if p_end - p_start < min_pause - 1e-9 or p_end < lo or p_start > hi:
            continue
        distance = 0.0 if p_start <= rel <= p_end else min(abs(rel - p_start), abs(rel - p_end))
        score = distance - 1.0 * min(p_end - p_start, 0.8)  # nearer wins, but a real pause beats a tiny gap between words
        if best_score is None or score < best_score:
            best, best_score = (p_start, p_end), score
    if best is None:
        return None
    p_start, p_end = best
    length = p_end - p_start
    cut = p_end - min(START_LEAD, length / 2) if kind == "start" else p_start + min(END_TAIL, length / 2)
    return window_start + min(max(cut, lo), hi)


def snap_clip(clip: Dict[str, Any], get_audio: Callable[[float, float], np.ndarray], duration: float) -> Dict[str, Any]:
    """Moves both ends of `clip` into nearby pauses. `get_audio(start, length)` returns 16 kHz mono samples.

    The result carries pre_roll / post_roll = 0 when a side was snapped (the exporter must not pad it again)
    and a small safety margin when it could not be."""
    snapped = dict(clip)
    for kind, key, pad_key in (("start", "start_time", "pre_roll"), ("end", "end_time", "post_roll")):
        target = float(clip[key])
        cut = None
        for reach in SEARCH_TIERS:  # nearest pause first, a wider search only if there is none
            win_start = max(0.0, target - reach - WINDOW_MARGIN)
            win_end = min(duration, target + reach + WINDOW_MARGIN)
            if win_end - win_start < 1.0:
                break
            try:
                audio = get_audio(win_start, win_end - win_start)
                for min_pause in (REAL_PAUSE, MIN_PAUSE):  # a real pause beats a breath inside continuing speech
                    cut = find_cut(audio, win_start, target, kind, reach, min_pause)
                    if cut is not None:
                        break
            except Exception:
                cut = None
            if cut is not None:
                break
        if cut is None:
            snapped[pad_key] = UNSNAPPED_PAD
        else:
            snapped[key] = round(max(0.0, min(cut, duration)), 3)
            snapped[pad_key] = 0.0
    if snapped["end_time"] - snapped["start_time"] < 1.0:  # snapping must never collapse a clip
        return dict(clip)
    return snapped
