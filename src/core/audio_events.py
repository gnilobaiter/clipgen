"""Local, dependency-free (numpy only) audio event detection.

Whisper transcripts carry no information about laughter or screaming, so this module
looks at the raw waveform instead. Everything is measured *relative to the local
background level* (rolling median), never against the global maximum: one huge scream
must not turn the rest of the stream into "quiet".

Detected events are heuristics, not a trained classifier:
  * LOUD    - short burst well above the surrounding level (SCREAM when it is also high-pitched)
  * LAUGHTER - sustained, rhythmic (3-8 Hz) amplitude modulation that is louder than the baseline
"""

from typing import Any, Dict, List, Tuple

import numpy as np

SAMPLE_RATE = 16000
FRAME = SAMPLE_RATE // 100  # 10 ms envelope frames
RMS_BLOCK_FRAMES = 6000  # 1 minute of frames per block
FRAMES_PER_STEP = 50  # analysis grid step = 0.5 s
STEP_S = FRAMES_PER_STEP / 100.0
BASELINE_RADIUS_STEPS = 60  # +-30 s rolling median
LAUGH_WINDOW_FRAMES = 200  # 2 s modulation window
SILENCE_DB = -55.0

BURST_EXCESS_DB = 18.0  # calibrated on a real 90-min gameplay VOD: 12 dB fired ~1x/min, 18 dB ~1 per 4 min
SCREAM_CENTROID_HZ = 1500.0
LAUGH_EXCESS_DB = 6.0
LAUGH_BAND_HZ = (3.0, 8.0)
LAUGH_MIN_PEAKINESS = 9.0  # ordinary speech already sits at a median of ~3.5, top 10% at ~7
LAUGH_MIN_DEPTH = 0.6
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


def _spectral_centroid(audio: np.ndarray) -> float:
    if len(audio) < 256:
        return 0.0
    spectrum = np.abs(np.fft.rfft(audio.astype(np.float64) * np.hanning(len(audio)))) ** 2  # power-weighted
    freqs = np.fft.rfftfreq(len(audio), 1.0 / SAMPLE_RATE)
    total = spectrum.sum()
    return float((freqs * spectrum).sum() / total) if total > 0 else 0.0


def _runs(mask: np.ndarray, max_gap_steps: int) -> List[Tuple[int, int]]:
    """Groups True indices into (first, last) runs, bridging gaps up to max_gap_steps."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    runs = []
    first = prev = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i - prev > max_gap_steps + 1:
            runs.append((first, prev))
            first = i
        prev = i
    runs.append((first, prev))
    return runs


def detect_loud_events(audio: np.ndarray, level_db: np.ndarray, baseline_db: np.ndarray) -> List[Dict[str, Any]]:
    """Short bursts far above the local level: screams, jump scares, explosions."""
    if len(level_db) == 0:
        return []
    excess = level_db - baseline_db
    mask = (excess >= BURST_EXCESS_DB) & (level_db > SILENCE_DB)
    events = []
    for first, last in _runs(mask, max_gap_steps=int(MERGE_GAP_S / STEP_S)):
        start_s = first * STEP_S
        end_s = (last + 1) * STEP_S
        peak = float(excess[first:last + 1].max())
        clip = audio[int(start_s * SAMPLE_RATE):int(min(end_s, start_s + 3.0) * SAMPLE_RATE)]
        kind = "SCREAM" if _spectral_centroid(clip) >= SCREAM_CENTROID_HZ else "LOUD"
        events.append({
            "type": kind,
            "start": round(start_s, 2),
            "end": round(end_s, 2),
            "strength": int(min(100, round(peak / 30.0 * 100))),
        })
    return events


def detect_laughter_events(rms: np.ndarray, level_db: np.ndarray, baseline_db: np.ndarray) -> List[Dict[str, Any]]:
    """Laughter = sustained, regular 3-8 Hz amplitude pulsing ("ha-ha-ha") that is louder than the baseline.

    Plain speech also pulses at 4-6 Hz, but it is irregular (broad modulation spectrum) and does not sit
    above the rolling median; requiring a *peaky* modulation spectrum plus a level excess filters most of it.
    """
    steps = len(level_db)
    if steps == 0 or len(rms) < LAUGH_WINDOW_FRAMES:
        return []

    freqs = np.fft.rfftfreq(LAUGH_WINDOW_FRAMES, d=0.01)
    band = (freqs >= LAUGH_BAND_HZ[0]) & (freqs <= LAUGH_BAND_HZ[1])
    broad = (freqs >= 0.5) & (freqs <= 20.0)
    window = np.hanning(LAUGH_WINDOW_FRAMES)

    hits = np.zeros(steps, dtype=bool)
    scores = np.zeros(steps)
    max_start = (len(rms) - LAUGH_WINDOW_FRAMES) // FRAMES_PER_STEP
    for i in range(min(max_start + 1, steps)):
        seg = rms[i * FRAMES_PER_STEP:i * FRAMES_PER_STEP + LAUGH_WINDOW_FRAMES]
        mean = seg.mean()
        if mean <= 0:
            continue
        span = slice(i, min(steps, i + LAUGH_WINDOW_FRAMES // FRAMES_PER_STEP))
        if level_db[span].mean() <= SILENCE_DB or (level_db[span] - baseline_db[span]).mean() < LAUGH_EXCESS_DB:
            continue
        depth = float(seg.std() / mean)
        if depth < LAUGH_MIN_DEPTH:
            continue
        spec = np.abs(np.fft.rfft((seg - mean) * window)) ** 2
        broad_mean = spec[broad].mean()
        if broad_mean <= 0:
            continue
        peakiness = float(spec[band].max() / broad_mean)
        if peakiness >= LAUGH_MIN_PEAKINESS:
            hits[i] = True
            scores[i] = peakiness

    events = []
    window_steps = LAUGH_WINDOW_FRAMES // FRAMES_PER_STEP
    for first, last in _runs(hits, max_gap_steps=int(MERGE_GAP_S / STEP_S)):
        start_s = first * STEP_S
        end_s = (last + window_steps) * STEP_S
        mean_score = float(scores[first:last + 1][hits[first:last + 1]].mean())
        events.append({
            "type": "LAUGHTER",
            "start": round(start_s, 2),
            "end": round(end_s, 2),
            "strength": int(min(100, round(mean_score / 15.0 * 100))),
        })
    return events


def detect_audio_events(audio: np.ndarray, loud: bool = True, laughter: bool = True) -> List[Dict[str, Any]]:
    """Runs all detectors and returns events sorted by start time."""
    rms, level_db, baseline_db = compute_level_profile(audio)
    events: List[Dict[str, Any]] = []
    if loud:
        events.extend(detect_loud_events(audio, level_db, baseline_db))
    if laughter:
        events.extend(detect_laughter_events(rms, level_db, baseline_db))
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
