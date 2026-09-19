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


# --------------------------- hard length cap & selection statistics ---------------------------
def test_limit_length_keeps_the_punchline_or_the_beginning():
    ok = _clip(0, 100)
    assert hl.limit_length(ok) is ok  # untouched when within the cap
    cap = hl.SAFE_MAX_CLIP_SECONDS
    long_no_peak = hl.limit_length(_clip(1000, 1254))
    assert (long_no_peak["start_time"], long_no_peak["end_time"]) == (1000, 1000 + cap)
    with_peak = hl.limit_length(_clip(1000, 1254, peak_time=1100.0))
    assert with_peak["end_time"] - with_peak["start_time"] == pytest.approx(cap)
    assert with_peak["start_time"] <= 1100.0 <= with_peak["end_time"]
    assert 1000 <= with_peak["start_time"] and with_peak["end_time"] <= 1254
    late_peak = hl.limit_length(_clip(1000, 1254, peak_time=1250.0))
    assert late_peak["start_time"] <= 1250.0 <= late_peak["end_time"] and late_peak["end_time"] <= 1254
    assert hl.limit_length(_clip(0, 500), max_len=100)["end_time"] - 0 == 100


def test_dedupe_never_widens_a_moment_beyond_the_absolute_maximum():
    a = _clip(100, 300, 9)  # 200 s
    b = _clip(180, 380, 7)  # overlaps a lot, the union would be 280 s
    kept = hl.dedupe_candidates([a, b])
    assert len(kept) == 1 and (kept[0]["start_time"], kept[0]["end_time"]) == (100, 300)
    small = hl.dedupe_candidates([_clip(100, 130, 9), _clip(110, 150, 7)])
    assert (small[0]["start_time"], small[0]["end_time"]) == (100, 150)


def test_select_clips_reports_why_candidates_were_dropped():
    cands = [_clip(100, 130, 9), _clip(120, 150, 8), _clip(300, 330, 5), _clip(500, 503, 9),
             _clip(600, 630, 7), _clip(800, 830, 7), _clip(1000, 1030, 7)]
    stats = {}
    chosen = hl.select_clips(cands, duration=3600, clips_per_hour=3, min_score=6, stats=stats)
    assert [c["start_time"] for c in chosen] == [100, 600, 800]
    assert stats == {"low_score": 1, "too_short": 1, "too_close": 1, "over_limit": 1, "limit": 3}
    unlimited = {}
    hl.select_clips(cands, duration=3600, clips_per_hour=0, min_score=6, stats=unlimited)
    assert unlimited["limit"] == -1 and unlimited["over_limit"] == 0


# --------------------------- finishing the thought (real cases from the test video) ---------------------------
def _u(start, end, text, **extra):
    return {"start": start, "end": end, "text": text, **extra}


def _cut(start, end, **extra):
    return {"start_time": start, "end_time": end, "virality_score": 8, "reasoning": "r", **extra}


def test_hesitation_pause_in_the_middle_of_a_sentence_does_not_end_the_clip():
    """Clip 15: the clip stopped after "..., это" and the sentence went on 0.9 s later."""
    entries = [_u(3240.0, 3246.9, "Нет, смотри, короче, первый, это"), _u(3247.8, 3252.0, "первая лампочка, то есть имеется в виду не загоревшаяся."),
               _u(3260.0, 3262.0, "Совсем другая тема.")]
    out = hl.complete_thoughts(_cut(3226.0, 3246.9), entries)
    assert out["end_time"] == pytest.approx(3252.0)  # the sentence is finished, the next topic is not touched


def test_closing_remark_after_the_punchline_is_included_as_an_opening_of_the_next_line():
    """Clip 10: "...А пенис выше, типа наверх." is followed after 0.5 s by "все логично же по сути" + a new topic in ONE long line."""
    entries = [_u(2322.3, 2325.1, "А пенис выше, типа наверх."),
               _u(2325.6, 2336.4, "все логично же по сути открываем смотри 1 у нас тут пола этот не ползунки у нас")]
    out = hl.complete_thoughts(_cut(2282.0, 2325.1), entries)
    assert out["end_time"] == pytest.approx(2325.6 + hl.PARTIAL_TAIL_SECONDS)  # only the opening of the long line
    assert out["end_time"] < 2336.4


def test_short_reaction_after_a_finished_sentence_is_included_but_a_new_topic_after_a_pause_is_not():
    """Clip 16 vs clip 7."""
    reaction = [_u(3826.4, 3830.8, "Их две, их два песочных часа, одни закруглые, одни треугольные."), _u(3831.8, 3832.2, "Ааа."),
                _u(3832.7, 3834.7, "Я же говорю, треугольники с вершинами."), _u(3834.9, 3840.8, "Я их не понял. Бля, это пиздец, сука. Это еще")]
    out = hl.complete_thoughts(_cut(3775.0, 3830.8), reaction)
    assert 3834.7 <= out["end_time"] <= 3834.9 + hl.PARTIAL_TAIL_SECONDS + 1e-6

    topic_change = [_u(1607.8, 1609.0, "Так я и говорю, что 4."), _u(1609.1, 1609.3, "Да, да."), _u(1610.4, 1612.5, "А кто сказал 57-34?")]
    assert hl.complete_thoughts(_cut(1554.0, 1609.3), topic_change)["end_time"] == 1609.3  # a 1.1 s pause and a new subject: leave it


def test_clip_that_stops_inside_a_line_finishes_that_line_within_the_limit():
    entries = [_u(100.0, 108.0, "Я только по шрифту этого вижу кнопки числа."), _u(120.0, 125.0, "А потом ещё")]
    assert hl.complete_thoughts(_cut(50.0, 104.0), entries)["end_time"] == 108.0
    assert hl.complete_thoughts(_cut(50.0, 101.0), [_u(100.0, 160.0, "очень длинная реплика")])["end_time"] == 101.0  # too far to finish


def test_tail_extension_is_bounded_even_in_endless_chatter():
    entries = [_u(100.0 + i * 3.0, 102.9 + i * 3.0, f"часть {i}") for i in range(12)]  # gapless, no sentence ends
    out = hl.complete_thoughts(_cut(50.0, 102.9), entries)
    assert out["end_time"] - 102.9 <= hl.MAX_TAIL_EXTENSION + 1e-6


def test_start_in_the_middle_of_a_sentence_moves_back_only_when_the_previous_line_is_unfinished_and_adjacent():
    open_prev = [_u(92.0, 95.0, "один только загорелся дальше цифра какая"), _u(95.5, 99.0, "5 цифра 5 и зеленый")]
    assert hl.complete_thoughts(_cut(95.5, 130.0), open_prev)["start_time"] == 92.0
    finished_prev = [_u(90.0, 95.0, "А как?"), _u(95.5, 99.0, "я смотрю но мне показывают")]
    assert hl.complete_thoughts(_cut(95.5, 130.0), finished_prev)["start_time"] == 95.5  # a new sentence starts here
    far_prev = [_u(80.0, 90.0, "что-то без точки"), _u(95.5, 99.0, "новая мысль")]
    assert hl.complete_thoughts(_cut(95.5, 130.0), far_prev)["start_time"] == 95.5  # a real pause between them
    too_far = [_u(60.0, 95.3, "очень длинная реплика без точки"), _u(95.5, 99.0, "продолжение")]
    assert hl.complete_thoughts(_cut(95.5, 130.0), too_far)["start_time"] == 95.5  # would add more than MAX_HEAD_EXTENSION


def test_complete_thoughts_leaves_clips_alone_without_speech_and_never_mutates_input():
    clip = _cut(10.0, 20.0)
    assert hl.complete_thoughts(clip, []) is clip
    assert hl.complete_thoughts(clip, [{"start": 12.0, "end": 14.0, "text": "", "event": "LAUGHTER"}]) is clip
    entries = [_u(19.0, 20.0, "и вот")]
    hl.complete_thoughts(clip, entries + [_u(20.5, 21.0, "да")])
    assert clip["end_time"] == 20.0


# --------------------------- pause markers in the transcript shown to the AI ---------------------------
def test_transcript_shows_real_pauses_and_none_inside_continuous_speech():
    lines = [_u(0.0, 3.0, "первая"), _u(3.3, 5.0, "вторая без паузы"), _u(6.5, 8.0, "третья после паузы"),
             {"start": 8.2, "end": 9.0, "text": "", "event": "LAUGHTER", "strength": 60}, _u(12.0, 13.0, "четвёртая")]
    text = hl.format_transcript(lines)
    assert "⏸ 1.5s" in text and "⏸ 4.0s" in text  # the gap to the line after the laugh is measured from the last speech end
    assert text.count("⏸") == 2  # the 0.3 s gap between the first two lines is not a pause
    assert text.index("вторая без паузы") < text.index("⏸ 1.5s") < text.index("третья после паузы")
    assert "[LAUGHTER 60%]" in text
    marked = hl.format_transcript(lines, lambda e: "> " if e["start"] < 6 else "  ")
    assert marked.startswith("> [0.0s - 3.0s]") and "  ⏸ 1.5s" in marked
    assert hl.format_transcript([]) == ""
