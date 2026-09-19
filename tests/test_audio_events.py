import numpy as np

from src.core import audio_events as ae

SR = 16000


def _background(seconds: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = SR * seconds
    envelope = np.abs(np.convolve(rng.standard_normal(n), np.ones(1600) / 1600, "same"))
    return (rng.standard_normal(n) * 0.03 * (0.3 + envelope / envelope.max() * 3)).astype(np.float32)


def test_loud_burst_is_detected_relative_to_local_background():
    audio = _background(120)
    t = np.arange(len(audio)) / SR
    burst = (t >= 60) & (t < 62.5)
    audio[burst] += (0.8 * np.sin(2 * np.pi * 300 * t[burst])).astype(np.float32)
    events = ae.detect_audio_events(audio, classify=False)
    assert [e["type"] for e in events] == ["LOUD"]
    assert 59 <= events[0]["start"] <= 61 and 62 <= events[0]["end"] <= 64
    assert 0 < events[0]["strength"] <= 100


def test_constant_noise_and_speech_like_background_produce_no_events():
    noise = (np.random.default_rng(3).standard_normal(SR * 120) * 0.05).astype(np.float32)
    assert ae.detect_audio_events(noise, classify=False) == []
    assert ae.detect_audio_events(_background(180), classify=False) == []


def test_silence_and_short_audio_are_safe():
    assert ae.detect_audio_events(np.zeros(SR * 30, dtype=np.float32), classify=False) == []
    assert ae.detect_audio_events(np.zeros(10, dtype=np.float32), classify=False) == []
    assert ae.detect_audio_events(np.zeros(0, dtype=np.float32), classify=False) == []


def test_flags_and_classifier_events_are_merged_and_sorted(monkeypatch):
    audio = _background(120)
    t = np.arange(len(audio)) / SR
    burst = (t >= 60) & (t < 62)
    audio[burst] += (0.8 * np.sin(2 * np.pi * 300 * t[burst])).astype(np.float32)

    calls = []

    def fake_detect(a, logger=None):
        calls.append(logger)
        return [{"type": "LAUGHTER", "start": 10.0, "end": 12.0, "strength": 55}]

    monkeypatch.setattr("src.core.sound_classifier.detect_sound_events", fake_detect)
    logger = lambda msg: None  # noqa: E731
    events = ae.detect_audio_events(audio, logger=logger)
    assert [e["type"] for e in events] == ["LAUGHTER", "LOUD"]
    assert calls == [logger]

    assert [e["type"] for e in ae.detect_audio_events(audio, loud=False)] == ["LAUGHTER"]
    assert [e["type"] for e in ae.detect_audio_events(audio, classify=False)] == ["LOUD"]
    assert ae.detect_audio_events(audio, loud=False, classify=False) == []


def test_group_runs_bridges_small_gaps():
    mask = np.array([1, 1, 0, 1, 0, 0, 0, 1], dtype=bool)
    assert ae.group_runs(mask, 1) == [(0, 3), (7, 7)]
    assert ae.group_runs(mask, 0) == [(0, 1), (3, 3), (7, 7)]
    assert ae.group_runs(np.zeros(4, dtype=bool), 2) == []


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
