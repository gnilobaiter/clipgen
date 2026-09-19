import numpy as np
import pytest

from src.core import cuts

SR = 16000


def _speech(total_s, words, seed=0):
    """Low background noise plus loud 'words' (start, end) in seconds."""
    rng = np.random.default_rng(seed)
    audio = (rng.standard_normal(int(total_s * SR)) * 0.001).astype(np.float32)
    t = np.arange(len(audio)) / SR
    for start, end in words:
        m = (t >= start) & (t < end)
        audio[m] += (0.3 * np.sin(2 * np.pi * 220 * t[m]) + rng.standard_normal(m.sum()) * 0.05).astype(np.float32)
    return audio


# phrase A: three words with 0.15 s gaps; 0.6 s pause; phrase B: two words
WORDS = [(0.0, 0.3), (0.45, 0.75), (0.9, 1.2), (1.8, 2.1), (2.25, 2.55)]


def test_frame_db_and_pauses_find_the_real_gaps():
    db = cuts.frame_db(_speech(3.0, WORDS))
    pauses = cuts.find_pauses(db)
    spans = [(round(a * 0.01, 2), round((b + 1) * 0.01, 2)) for a, b in pauses]
    assert any(1.2 <= a <= 1.3 and 1.7 <= b <= 1.85 for a, b in spans)  # the 0.6 s pause is found
    assert cuts.frame_db(np.zeros(5)).size == 0 and cuts.find_pauses(np.zeros(0)) == []


def test_start_cut_lands_in_the_pause_before_the_word_not_inside_it():
    audio = _speech(3.0, WORDS)
    cut = cuts.find_cut(audio, 0.0, target=1.95, kind="start")  # target is in the middle of the first word of phrase B
    assert cut is not None and 1.2 < cut < 1.8
    assert cut == pytest.approx(1.8 - cuts.START_LEAD, abs=0.05)


def test_end_cut_lands_in_the_pause_after_the_last_word():
    audio = _speech(3.0, WORDS)
    cut = cuts.find_cut(audio, 0.0, target=1.1, kind="end")  # target is inside the last word of phrase A
    assert cut is not None and 1.2 < cut < 1.8
    assert cut == pytest.approx(1.2 + cuts.END_TAIL, abs=0.05)


def test_cut_moves_at_most_the_search_slack_on_the_wrong_side():
    audio = _speech(3.0, WORDS)
    cut = cuts.find_cut(audio, 0.0, target=1.5, kind="start")  # target already inside the pause
    assert cut is not None and 1.2 <= cut <= 1.5 + cuts.SEARCH_SLACK + 1e-6
    assert cuts.find_cut(audio, 0.0, target=1.5, kind="end") >= 1.5 - cuts.SEARCH_SLACK - 1e-6


def test_window_offset_is_respected():
    audio = _speech(3.0, WORDS)
    cut = cuts.find_cut(audio, 100.0, target=101.95, kind="start")
    assert 101.2 < cut < 101.8


def test_no_contrast_or_no_pause_means_no_cut():
    noise = (np.random.default_rng(1).standard_normal(SR * 3) * 0.05).astype(np.float32)
    assert cuts.find_cut(noise, 0.0, 1.5, "start") is None
    solid = _speech(3.0, [(0.0, 3.0)])
    assert cuts.find_cut(solid, 0.0, 1.5, "end") is None


def test_snap_clip_moves_both_ends_into_pauses_and_zeroes_the_exporter_padding():
    audio = _speech(6.0, WORDS + [(3.5, 3.8), (3.95, 4.3)])  # phrase C after another pause

    def get_audio(start, length):
        return audio[int(start * SR):int((start + length) * SR)]

    clip = {"start_time": 1.95, "end_time": 4.1, "virality_score": 8}
    snapped = cuts.snap_clip(clip, get_audio, duration=6.0)
    assert 1.2 < snapped["start_time"] < 1.8 and snapped["pre_roll"] == 0.0
    assert 4.3 <= snapped["end_time"] <= 5.5 and snapped["post_roll"] == 0.0
    assert clip["start_time"] == 1.95  # input untouched
    assert snapped["virality_score"] == 8


def test_snap_clip_keeps_time_and_adds_small_margin_when_a_side_cannot_be_snapped():
    noise = (np.random.default_rng(2).standard_normal(SR * 10) * 0.05).astype(np.float32)
    snapped = cuts.snap_clip({"start_time": 4.0, "end_time": 8.0}, lambda s, n: noise[int(s * SR):int((s + n) * SR)], duration=10.0)
    assert (snapped["start_time"], snapped["end_time"]) == (4.0, 8.0)
    assert snapped["pre_roll"] == snapped["post_roll"] == cuts.UNSNAPPED_PAD


def test_snap_clip_survives_audio_errors_and_video_edges():
    def broken(start, length):
        raise RuntimeError("ffmpeg died")

    snapped = cuts.snap_clip({"start_time": 0.0, "end_time": 5.0}, broken, duration=5.0)
    assert (snapped["start_time"], snapped["end_time"]) == (0.0, 5.0) and snapped["pre_roll"] == cuts.UNSNAPPED_PAD

    audio = _speech(6.0, WORDS)
    clamped = cuts.snap_clip({"start_time": 0.1, "end_time": 5.95}, lambda s, n: audio[int(s * SR):int((s + n) * SR)], duration=6.0)
    assert 0.0 <= clamped["start_time"] < clamped["end_time"] <= 6.0


def test_snapping_never_collapses_a_clip():
    audio = _speech(6.0, [(0.0, 0.3), (0.6, 0.9), (1.2, 1.5)] + [(2.0 + i * 0.5, 2.3 + i * 0.5) for i in range(6)])
    clip = {"start_time": 2.2, "end_time": 2.9}

    def get_audio(start, length):
        return audio[int(start * SR):int((start + length) * SR)]

    snapped = cuts.snap_clip(clip, get_audio, duration=6.0)
    assert snapped["end_time"] - snapped["start_time"] >= 0.5


def test_short_dips_between_words_are_not_pauses_only_real_gaps_are():
    """On a real VOD 28% of cuts sat in 0.1-0.2 s dips (soft syllables, gaps between words) and clipped a word."""
    db = cuts.frame_db(_speech(3.0, WORDS))  # WORDS has 0.15 s gaps inside the phrases and one 0.6 s pause
    lengths = [(last - first + 1) * 0.01 for first, last in cuts.find_pauses(db)]
    assert lengths and min(lengths) >= cuts.MIN_PAUSE - 1e-9 and max(lengths) >= 0.5
    assert cuts.MIN_PAUSE >= 0.2


def test_a_cut_prefers_the_real_pause_over_a_nearer_word_gap():
    audio = _speech(3.0, WORDS)
    cut = cuts.find_cut(audio, 0.0, target=0.82, kind="end")  # right at the 0.15 s gap after the second word
    assert cut is not None and 1.2 < cut < 1.8  # it goes to the 0.6 s pause instead of the gap
