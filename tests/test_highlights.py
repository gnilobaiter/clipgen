import pytest

from src.core import highlights as hl


def _seg(start, end, text="speech", **extra):
    return {"start": start, "end": end, "text": text, **extra}


def _clip(start, end, score=8, **extra):
    return {"start_time": start, "end_time": end, "virality_score": score, "reasoning": "r", **extra}


# --------------------------- formatting & windows ---------------------------
def test_format_entry_tags_and_events():
    line = hl.format_entry(_seg(1.0, 2.5, " hi ", loudness=80, is_combat=True))
    assert line == "[1.0s - 2.5s] [LOUDNESS: 80%] [ACTION: COMBAT] hi\n"
    assert hl.format_entry(_seg(1.0, 2.0, "plain", loudness=10)) == "[1.0s - 2.0s] plain\n"
    event = {"start": 5.0, "end": 8.0, "text": "", "event": "LAUGHTER", "strength": 70}
    assert hl.format_entry(event) == "[5.0s - 8.0s] [LAUGHTER 70%]\n"


def test_build_windows_single_for_short_video():
    entries = [_seg(0, 10), _seg(10, 600)]
    windows = hl.build_windows(entries)
    assert len(windows) == 1 and windows[0]["end"] == 600 and len(windows[0]["lines"]) == 2
    assert hl.build_windows([]) == []


def test_build_windows_overlap_and_coverage():
    entries = [_seg(i * 10, i * 10 + 10) for i in range(540)]  # 90 minutes
    windows = hl.build_windows(entries)
    assert len(windows) > 5
    assert windows[0]["start"] == 0 and windows[-1]["end"] == 5400
    for prev, nxt in zip(windows, windows[1:]):
        assert nxt["start"] < prev["end"]  # overlap
        assert nxt["start"] - prev["start"] == pytest.approx(hl.WINDOW_SECONDS - hl.WINDOW_OVERLAP)
    covered = {id(e) for w in windows for e in w["lines"]}
    assert len(covered) == len(entries)


def test_target_clip_count():
    assert hl.target_clip_count(5400, 12) == 18
    assert hl.target_clip_count(10, 12) == 1
    assert hl.target_clip_count(5400, 0) is None
    assert hl.requested_candidates(720, 12) == 5
    assert hl.requested_candidates(720, 0) >= 2


# --------------------------- dedupe & selection ---------------------------
def test_dedupe_merges_same_moment_from_two_windows():
    a = _clip(100, 130, 7)
    b = _clip(105, 140, 9)
    c = _clip(400, 430, 6)
    result = hl.dedupe_candidates([a, b, c])
    assert len(result) == 2
    merged = result[0]
    assert (merged["start_time"], merged["end_time"], merged["virality_score"]) == (100, 140, 9)


def test_select_clips_honours_density_score_and_gaps():
    cands = [_clip(i * 100, i * 100 + 30, 6 + i % 5) for i in range(40)]
    cands.append(_clip(5, 8, 10))  # shorter than the minimum length
    cands.append(_clip(1000, 1030, 3))  # below min score
    chosen = hl.select_clips(cands, duration=3600, clips_per_hour=10)
    assert len(chosen) == 10
    assert all(c["virality_score"] >= 6 for c in chosen)
    assert [c["start_time"] for c in chosen] == sorted(c["start_time"] for c in chosen)
    assert min(c["virality_score"] for c in chosen) >= 9  # best ones win


def test_select_clips_skips_overlapping_and_adjacent():
    best = _clip(100, 130, 10)
    overlapping = _clip(120, 150, 9)
    adjacent = _clip(133, 160, 9)  # gap 3s < MIN_CLIP_GAP
    far = _clip(300, 330, 8)
    chosen = hl.select_clips([best, overlapping, adjacent, far], duration=3600, clips_per_hour=0)
    assert [c["start_time"] for c in chosen] == [100, 300]


def test_select_clips_unlimited_keeps_all_valid():
    cands = [_clip(i * 100, i * 100 + 20, 7) for i in range(30)]
    assert len(hl.select_clips(cands, duration=3600, clips_per_hour=0)) == 30


# --------------------------- boundary refinement ---------------------------
def test_refine_snaps_start_and_end_to_phrase_boundaries():
    segments = [_seg(90.0, 100.0), _seg(100.0, 110.0), _seg(110.0, 118.0), _seg(130.0, 140.0)]
    refined = hl.refine_clip(_clip(101.5, 116.5), segments, duration=1000)
    # start inside phrase 100-110 -> back to its start; end inside 110-118 -> forward to its end
    assert refined["start_time"] == pytest.approx(100.0)
    assert refined["end_time"] == pytest.approx(118.0)


def test_refine_does_not_stretch_a_clip_by_more_than_the_small_snap_radius():
    long_segment = _seg(100.0, 140.0, "one long segment without words")
    refined = hl.refine_clip(_clip(110.0, 130.0), [long_segment], duration=1000)
    assert (refined["start_time"], refined["end_time"]) == (110.0, 130.0)  # nothing within 2 s to snap to
    refined = hl.refine_clip(_clip(101.0, 139.0), [long_segment], duration=1000)
    assert (refined["start_time"], refined["end_time"]) == (100.0, 140.0)


def test_refine_extends_over_laughter_and_adds_tail():
    segments = [
        _seg(10.0, 18.0, "setup"),
        _seg(18.0, 22.0, "punchline"),
        {"start": 21.0, "end": 27.0, "text": "", "event": "LAUGHTER", "strength": 90},
    ]
    refined = hl.refine_clip(_clip(10.0, 22.0), segments, duration=1000)
    assert refined["end_time"] == pytest.approx(27.0)


def test_refine_laughter_starting_just_after_clip_is_included():
    segments = [_seg(0.0, 12.0, "x"), {"start": 13.0, "end": 17.0, "text": "", "event": "LAUGHTER", "strength": 80}]
    refined = hl.refine_clip(_clip(2.0, 12.0), segments, duration=1000)
    assert refined["end_time"] >= 17.0


def test_refine_uses_peak_time_for_setup_and_reaction():
    segments = [_seg(0.0, 3.0)]
    refined = hl.refine_clip(_clip(50.0, 52.0, peak_time=51.0), segments, duration=1000)
    assert refined["start_time"] <= 51.0 - hl.PEAK_LEAD
    assert refined["end_time"] >= 51.0 + hl.PEAK_TAIL


def test_refine_word_level_snapping_inside_long_segment():
    words = [{"word": "a", "start": 40.0, "end": 40.5}, {"word": "b", "start": 41.0, "end": 41.5},
             {"word": "c", "start": 50.0, "end": 50.6}, {"word": "d", "start": 50.7, "end": 51.4}]
    seg = _seg(40.0, 60.0, "long", words=words)
    refined = hl.refine_clip(_clip(50.3, 70.0), [seg], duration=1000)
    assert refined["start_time"] == pytest.approx(50.0)  # snapped to word-run start, not 40.0


def test_refine_clamps_to_video_bounds_and_min_length():
    refined = hl.refine_clip(_clip(0.2, 1.0), [], duration=3.0)
    assert refined["start_time"] == 0.2
    assert refined["end_time"] - refined["start_time"] >= hl.MIN_CLIP_SECONDS
    refined = hl.refine_clip(_clip(50.0, 70.0), [_seg(60.0, 90.0)], duration=65.0)
    assert refined["end_time"] == 65.0  # never past the end of the video


def test_refine_adds_no_padding_of_its_own():
    refined = hl.refine_clip(_clip(100.0, 120.0), [], duration=1000)
    assert (refined["start_time"], refined["end_time"]) == (100.0, 120.0)


def test_resolve_overlaps_trims_earlier_clip():
    clips = [_clip(100, 140), _clip(139, 170)]
    a, b = hl.resolve_overlaps(clips)
    assert a["end_time"] == pytest.approx(139 - hl.EXPORT_PAD)
    assert b["start_time"] == 139


def test_resolve_overlaps_shifts_later_clip_when_earlier_would_be_too_short():
    clips = [_clip(100, 108), _clip(101, 150)]
    a, b = hl.resolve_overlaps(clips)
    assert a["end_time"] == 108
    assert b["start_time"] == pytest.approx(108 + hl.EXPORT_PAD)


def test_resolve_overlaps_leaves_distant_clips_and_drops_slivers():
    clips = [_clip(100, 130), _clip(200, 230)]
    assert [(c["start_time"], c["end_time"]) for c in hl.resolve_overlaps(clips)] == [(100, 130), (200, 230)]
    assert hl.resolve_overlaps([_clip(100, 130), _clip(100, 131)])[0]["start_time"] == 100
    assert hl.resolve_overlaps([]) == []


def test_build_windows_absorbs_tiny_tail():
    entries = [_seg(i * 10, i * 10 + 10) for i in range(139)]  # 1390 s: last window would be 70 s of mostly overlap
    windows = hl.build_windows(entries)
    assert windows[-1]["end"] == 1390
    assert all(w["end"] - w["start"] > hl.WINDOW_OVERLAP * 2 for w in windows)


# --------------------------- utterances ---------------------------
def _w(word, start, end):
    return {"word": word, "start": start, "end": end}


def test_utterances_join_continuous_speech_that_whisper_split_mid_phrase():
    segs = [
        _seg(10.0, 14.0, " там от 1 от", words=[_w(" там", 10.0, 10.5)], loudness=20),
        _seg(14.1, 17.0, " 0 до 9 цифры", words=[_w(" 0", 14.1, 14.4)], loudness=70, is_combat=True),
        _seg(20.0, 22.0, " Новая тема."),
    ]
    out = hl.build_utterances(segs)
    assert [(u["start"], u["end"]) for u in out] == [(10.0, 17.0), (20.0, 22.0)]
    assert out[0]["text"] == "там от 1 от 0 до 9 цифры"
    assert out[0]["loudness"] == 70 and out[0]["is_combat"] is True
    assert [w["word"] for w in out[0]["words"]] == [" там", " 0"]
    assert segs[0]["end"] == 14.0  # input untouched


def test_utterances_do_not_join_across_a_real_pause_or_a_finished_sentence():
    real_pause = [_seg(0.0, 3.0, "первая фраза"), _seg(3.6, 6.0, "вторая фраза")]  # 0.6 s of silence
    assert len(hl.build_utterances(real_pause)) == 2
    sentence_end = [_seg(0.0, 3.0, "Первая фраза."), _seg(3.05, 6.0, "Вторая фраза")]  # tiny gap but the sentence ended
    assert len(hl.build_utterances(sentence_end)) == 2
    question = [_seg(0.0, 3.0, "Что?"), _seg(3.0, 4.0, "Да")]
    assert len(hl.build_utterances(question)) == 2


def test_utterances_respect_the_length_cap_and_pass_events_through():
    run_on = [_seg(i * 4.0, i * 4.0 + 4.0, f"часть {i}") for i in range(6)]  # 24 s of gapless speech
    out = hl.build_utterances(run_on)
    assert len(out) >= 2 and all(u["end"] - u["start"] <= hl.UTTERANCE_MAX + 1e-6 for u in out)
    event = {"start": 5.0, "end": 7.0, "text": "", "event": "LAUGHTER", "strength": 60}
    mixed = hl.build_utterances([_seg(0.0, 4.0, "а"), event, _seg(4.1, 8.0, "б")])
    assert sum(1 for u in mixed if u.get("event")) == 1  # the event is kept and never merged into speech
    speech = [u for u in mixed if not u.get("event")]
    assert len(speech) == 1 and (speech[0]["start"], speech[0]["end"]) == (0.0, 8.0)  # the two speech lines are one stretch
    assert hl.build_utterances([]) == []
