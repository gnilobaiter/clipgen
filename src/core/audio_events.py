"""Audio events for the transcript timeline.

Whisper transcripts carry no information about laughter or screaming, so the raw waveform is analysed:

  * LOUD              - short burst far above the surrounding background level (numpy, this module)
  * LAUGHTER / SCREAM - pretrained sound-event classifier (see sound_classifier.py)

Levels are always measured *relative to the local background* (rolling median), never against the global
maximum: one huge scream must not turn the rest of the stream into "quiet".

(An earlier amplitude-modulation heuristic for laughter was removed: on real speech it fired ~4x per
minute on things that were not laughter.)
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

SAMPLE_RATE = 16000
FRAME = SAMPLE_RATE // 100  # 10 ms envelope frames
RMS_BLOCK_FRAMES = 6000  # 1 minute of frames per block
FRAMES_PER_STEP = 50  # analysis grid step = 0.5 s
STEP_S = FRAMES_PER_STEP / 100.0
BASELINE_RADIUS_STEPS = 60  # +-30 s rolling median
SILENCE_DB = -55.0

BURST_EXCESS_DB = 18.0  # calibrated on a real 90-min gameplay VOD: 12 dB fired ~1x/min, 18 dB ~1 per 4 min
MERGE_GAP_S = 1.0


def _frame_rms(audio: np.ndarray) -> np.ndarray:
    """RMS per 10 ms frame, computed in ~1-minute blocks so a multi-hour recording never needs
    float64 copies of the whole waveform (that was ~4x the audio size in extra RAM)."""
    n = len(audio) // FRAME
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    rms = np.empty(n, dtype=np.float64)
    for lo in range(0, n, RMS_BLOCK_FRAMES):
        hi = min(n, lo + RMS_BLOCK_FRAMES)
        block = audio[lo * FRAME:hi * FRAME].astype(np.float64).reshape(hi - lo, FRAME)
        rms[lo:hi] = np.sqrt(np.mean(block ** 2, axis=1))
    return rms


def _rolling_median(values: np.ndarray, radius: int) -> np.ndarray:
    """Centered rolling median; the window shrinks at the edges instead of repeating the edge value
    (O(n * radius), fine for hours of audio)."""
    if len(values) == 0:
        return values
    padded = np.pad(values.astype(np.float64), radius, mode="constant", constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * radius + 1)
    return np.nanmedian(windows, axis=1)


def compute_level_profile(audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (rms per 10 ms frame, level dB per 0.5 s step, local baseline dB per step)."""
    rms = _frame_rms(audio)
    steps = len(rms) // FRAMES_PER_STEP
    if steps == 0:
        return rms, np.zeros(0), np.zeros(0)
    power = (rms[: steps * FRAMES_PER_STEP] ** 2).reshape(steps, FRAMES_PER_STEP).mean(axis=1)
    level_db = 10.0 * np.log10(power + 1e-12)
    baseline_db = _rolling_median(level_db, BASELINE_RADIUS_STEPS)
    return rms, level_db, baseline_db


def group_runs(mask: np.ndarray, max_gap: int) -> List[Tuple[int, int]]:
    """Groups True indices into (first, last) runs, bridging gaps of up to `max_gap` False entries."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    runs = []
    first = prev = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i - prev > max_gap + 1:
            runs.append((first, prev))
            first = i
        prev = i
    runs.append((first, prev))
    return runs


def detect_loud_events(level_db: np.ndarray, baseline_db: np.ndarray) -> List[Dict[str, Any]]:
    """Short bursts far above the local level: jump scares, explosions, sudden shouts."""
    if len(level_db) == 0:
        return []
    excess = level_db - baseline_db
    mask = (excess >= BURST_EXCESS_DB) & (level_db > SILENCE_DB)
    events = []
    for first, last in group_runs(mask, max_gap=int(MERGE_GAP_S / STEP_S)):
        peak = float(excess[first:last + 1].max())
        events.append({
            "type": "LOUD",
            "start": round(first * STEP_S, 2),
            "end": round((last + 1) * STEP_S, 2),
            "strength": int(min(100, round(peak / 30.0 * 100))),
        })
    return events


def detect_audio_events(audio: np.ndarray, loud: bool = True, classify: bool = True,
                        logger: Optional[Callable[[str], None]] = None) -> List[Dict[str, Any]]:
    """Runs all detectors and returns events sorted by start time. The classifier degrades gracefully:
    if it is unavailable the reason is logged and the loudness events are still returned."""
    events: List[Dict[str, Any]] = []
    if loud:
        _rms, level_db, baseline_db = compute_level_profile(audio)
        events.extend(detect_loud_events(level_db, baseline_db))
    if classify:
        from src.core import sound_classifier  # lazy: importing onnxruntime is slow and optional
        events.extend(sound_classifier.detect_sound_events(audio, logger))
    events.sort(key=lambda e: (e["start"], e["type"]))
    return events


def segment_loudness(start: float, end: float, level_db: np.ndarray, baseline_db: np.ndarray) -> int:
    """0-100 loudness of [start, end] relative to the *local* baseline (0 = at baseline, 100 = +20 dB)."""
    if len(level_db) == 0:
        return 0
    lo = max(0, min(int(start / STEP_S), len(level_db) - 1))
    hi = max(lo + 1, min(int(np.ceil(end / STEP_S)), len(level_db)))
    excess = float((level_db[lo:hi] - baseline_db[lo:hi]).max())
    return int(max(0.0, min(100.0, excess / 20.0 * 100.0)))
