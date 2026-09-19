import numpy as np

from src.core import audio_events as ae

SR = 16000


def _speech_like_background(seconds: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = SR * seconds
    envelope = np.abs(np.convolve(rng.standard_normal(n), np.ones(1600) / 1600, "same"))
    return (rng.standard_normal(n) * 0.03 * (0.3 + envelope / envelope.max() * 3)).astype(np.float32)


def test_detects_laughter_and_scream_without_false_positives():
    audio = _speech_like_background(180)
    t = np.arange(len(audio)) / SR

    laugh = (t >= 60) & (t < 66)
    pulses = (0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 5 * t))) * np.exp(-((t * 5) % 1) * 3)
    rng = np.random.default_rng(1)
    audio[laugh] += (rng.standard_normal(laugh.sum()) * 0.15 * pulses[laugh]).astype(np.float32)

    scream = (t >= 120) & (t < 122.5)
    audio[scream] += (0.6 * np.sin(2 * np.pi * 2200 * t[scream])).astype(np.float32)

    events = ae.detect_audio_events(audio)
    kinds = [e["type"] for e in events]
    assert kinds == ["LAUGHTER", "SCREAM"]
    laughter, scream_ev = events
    assert 58 <= laughter["start"] <= 62 and 64 <= laughter["end"] <= 68
    assert 119 <= scream_ev["start"] <= 121 and 122 <= scream_ev["end"] <= 124
    assert 0 < laughter["strength"] <= 100


def test_low_pitched_burst_is_loud_not_scream():
    audio = _speech_like_background(60)
    t = np.arange(len(audio)) / SR
    burst = (t >= 30) & (t < 31.5)
    audio[burst] += (0.6 * np.sin(2 * np.pi * 150 * t[burst])).astype(np.float32)
    events = ae.detect_audio_events(audio, laughter=False)
    assert [e["type"] for e in events] == ["LOUD"]


def test_constant_noise_produces_no_events():
    audio = (np.random.default_rng(3).standard_normal(SR * 120) * 0.05).astype(np.float32)
    assert ae.detect_audio_events(audio) == []


def test_silence_and_short_audio_are_safe():
    assert ae.detect_audio_events(np.zeros(SR * 30, dtype=np.float32)) == []
    assert ae.detect_audio_events(np.zeros(10, dtype=np.float32)) == []
    assert ae.detect_audio_events(np.zeros(0, dtype=np.float32)) == []


def test_detector_flags_can_disable_each_detector():
    audio = _speech_like_background(60)
    t = np.arange(len(audio)) / SR
    burst = (t >= 30) & (t < 32)
    audio[burst] += (0.6 * np.sin(2 * np.pi * 2200 * t[burst])).astype(np.float32)
    assert ae.detect_audio_events(audio, loud=False, laughter=False) == []
    assert ae.detect_audio_events(audio, laughter=False)[0]["type"] == "SCREAM"


def test_loudness_is_relative_to_local_baseline_not_global_max():
    # a huge scream at the end must not flatten the loudness of an earlier moderately loud stretch
    audio = np.full(SR * 120, 0.02, dtype=np.float32)
    audio[SR * 20:SR * 24] = 0.1  # +14 dB over background
    audio[SR * 110:SR * 112] = 1.0  # +34 dB
    _rms, level_db, baseline_db = ae.compute_level_profile(audio)
    moderate = ae.segment_loudness(20, 24, level_db, baseline_db)
    quiet = ae.segment_loudness(40, 44, level_db, baseline_db)
    assert moderate >= 60
    assert quiet == 0
    assert ae.segment_loudness(0, 1, np.zeros(0), np.zeros(0)) == 0


def test_chunked_rms_matches_direct_computation_across_block_boundaries():
    rng = np.random.default_rng(5)
    audio = (rng.standard_normal(ae.FRAME * (ae.RMS_BLOCK_FRAMES * 2 + 123) + 17) * 0.1).astype(np.float32)
    n = len(audio) // ae.FRAME
    expected = np.sqrt(np.mean(audio[: n * ae.FRAME].astype(np.float64).reshape(n, ae.FRAME) ** 2, axis=1))
    np.testing.assert_allclose(ae._frame_rms(audio), expected)
