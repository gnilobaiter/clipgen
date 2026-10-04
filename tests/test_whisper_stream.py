import pytest

from src.core import transcriber
from src.core.transcriber import TranscriptionCancelled, _Progress, split_by_speaker, speakers_of, tighten_word_timings


def test_progress_logs_every_quarter():
    logs = []
    progress = _Progress(logs.append, None, "Transcribing")
    for pct in (0, 5, 10, 24.9):
        progress(pct)
    assert logs == ["⏳ Transcribing: 0%"]
    progress(25)
    progress(40)
    progress(50)
    progress(100)
    assert [line.split(": ")[1] for line in logs] == ["0%", "25%", "50%", "100%"]


def test_progress_aborts_when_cancelled():
    state = {"cancel": False}
    progress = _Progress(None, lambda: state["cancel"], "Transcribing")
    progress(10)
    state["cancel"] = True
    with pytest.raises(TranscriptionCancelled):
        progress(20)


def test_progress_without_callbacks_never_raises():
    _Progress(None, None, "x")(50)


def _word(text, start, end, speaker=None):
    w = {"word": text, "start": start, "end": end}
    if speaker:
        w["speaker"] = speaker
    return w


def test_split_by_speaker_cuts_at_speaker_change():
    seg = {"start": 0.0, "end": 6.0, "text": "how are you fine thanks and you", "speaker": "SPEAKER_00", "words": [
        _word("how", 0.0, 0.5, "SPEAKER_00"), _word("are", 0.5, 1.0, "SPEAKER_00"), _word("you", 1.0, 1.5, "SPEAKER_00"),
        _word("fine", 2.0, 2.6, "SPEAKER_01"), _word("thanks", 2.6, 3.2, "SPEAKER_01"), _word("and", 3.2, 3.6, "SPEAKER_01"), _word("you", 3.6, 4.2, "SPEAKER_01"),
    ]}
    out = split_by_speaker([seg])
    assert [(o["speaker"], o["start"], o["end"], o["text"]) for o in out] == [
        ("SPEAKER_00", 0.0, 1.5, "how are you"), ("SPEAKER_01", 2.0, 4.2, "fine thanks and you")]


def test_split_by_speaker_absorbs_one_word_flicker():
    words = [_word("a", 0.0, 0.4, "SPEAKER_00"), _word("b", 0.4, 0.8, "SPEAKER_00"), _word("c", 0.8, 1.2, "SPEAKER_00"),
             _word("x", 1.2, 1.4, "SPEAKER_01"),
             _word("d", 1.4, 1.8, "SPEAKER_00"), _word("e", 1.8, 2.2, "SPEAKER_00")]
    seg = {"start": 0.0, "end": 2.2, "text": "a b c x d e", "speaker": "SPEAKER_00", "words": words}
    out = split_by_speaker([seg])
    assert len(out) == 1 and out[0]["speaker"] == "SPEAKER_00" and out[0]["text"] == "a b c x d e"


def test_split_by_speaker_keeps_single_speaker_and_unlabelled_segments():
    plain = {"start": 0.0, "end": 1.0, "text": "hi", "words": [_word("hi", 0.0, 1.0)]}
    one = {"start": 1.0, "end": 2.0, "text": "yo", "speaker": "SPEAKER_01", "words": [_word("yo", 1.0, 2.0, "SPEAKER_01")]}
    assert split_by_speaker([plain, one]) == [plain, one]


def test_speakers_of_lists_unique_sorted():
    assert speakers_of([{"speaker": "SPEAKER_01"}, {"speaker": "SPEAKER_00"}, {"speaker": "SPEAKER_01"}, {}]) == ["SPEAKER_00", "SPEAKER_01"]


def test_available_models_has_all_whisper_families():
    models = transcriber.available_models()
    for name in ("tiny", "large-v3", "large-v3-turbo", "distil-large-v3"):
        assert name in models


def test_resolve_hf_token_priority(tmp_path, monkeypatch):
    monkeypatch.setattr(transcriber, "get_data_dir", lambda: str(tmp_path))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert transcriber.resolve_hf_token({}) is None
    (tmp_path / ".hf_token").write_text("file-token\n", encoding="utf-8")
    assert transcriber.resolve_hf_token({}) == "file-token"
    monkeypatch.setenv("HF_TOKEN", "env-token")
    assert transcriber.resolve_hf_token({}) == "env-token"
    assert transcriber.resolve_hf_token({"huggingface": {"token": " cfg-token "}}) == "cfg-token"


def test_is_model_downloaded(tmp_path, monkeypatch):
    monkeypatch.setattr(transcriber, "get_models_dir", lambda: str(tmp_path))
    monkeypatch.setattr(transcriber, "_model_repo", lambda name: "Systran/faster-whisper-tiny" if name == "tiny" else "")
    assert not transcriber.is_model_downloaded("tiny")
    (tmp_path / "whisper" / "models--Systran--faster-whisper-tiny").mkdir(parents=True)
    assert transcriber.is_model_downloaded("tiny")
    assert not transcriber.is_model_downloaded("unknown")


# --------------------------- WhisperX stretches the last word of a line over the following silence ---------------------------
def test_tighten_cuts_the_stretched_last_word_and_ends_the_line_where_speech_ends():
    seg = {"start": 1.0, "end": 8.0, "text": "блядь, шапкой поцарапал, нахуй.", "words": [
        _word("блядь,", 1.0, 1.2), _word("шапкой", 1.2, 1.6), _word("поцарапал,", 1.6, 2.9), _word("нахуй.", 2.9, 8.0)]}
    out = tighten_word_timings([seg])[0]
    last = out["words"][-1]
    assert last["start"] == 2.9 and 0.5 < last["end"] - last["start"] <= 0.9  # a 6-letter word: well under a second, not 5 s
    assert out["end"] == last["end"] and out["end"] < 4.0  # the line ends with its last word, not 5 s of silence later
    assert out["words"][0] == seg["words"][0] and out["start"] == 1.0 and out["text"] == seg["text"]


def test_tighten_leaves_normal_words_and_long_but_possible_ones_alone():
    seg = {"start": 0.0, "end": 2.0, "text": "промахнулся!", "words": [_word("промахнулся!", 0.0, 1.35)]}
    assert tighten_word_timings([seg])[0]["end"] == 1.35  # 12 letters ~ 1.45 s allowed
    laugh = {"start": 0.0, "end": 30.0, "text": "Ха-ха-ха-ха-ха-ха-ха-ха-ха-ха", "words": [_word("Ха-ха-ха-ха-ха-ха-ха-ха-ха-ха", 0.0, 30.0)]}
    assert tighten_word_timings([laugh])[0]["end"] == 2.0  # never longer than MAX_WORD_SECONDS


def test_tighten_keeps_lines_without_word_timings_and_untimed_words():
    plain = {"start": 0.0, "end": 9.0, "text": "no alignment available"}
    untimed = {"start": 0.0, "end": 5.0, "text": "в 2026 году", "words": [_word("в", 0.0, 0.2), {"word": "2026"}, _word("году", 0.5, 4.5)]}
    out = tighten_word_timings([plain, untimed])
    assert out[0] == plain
    assert out[1]["words"][1] == {"word": "2026"} and out[1]["end"] < 1.5
    assert untimed["words"][2]["end"] == 4.5  # the input is not mutated


def test_tighten_uses_the_vad_speech_regions_as_the_ceiling_for_word_ends():
    # real case: speech 1591.2-1600.1, then 6 s of silence, then more speech at 1606.2
    seg = {"start": 1596.9, "end": 1606.1, "text": "шапкой поцарапал, нахуй.", "words": [
        _word("шапкой", 1596.9, 1597.3), _word("поцарапал,", 1597.3, 1598.7), _word("нахуй.", 1598.7, 1606.1)]}
    regions = [(1585.5, 1587.3), (1591.2, 1600.1), (1606.2, 1608.2)]
    out = tighten_word_timings([seg], regions)[0]
    assert out["words"][-1]["end"] <= 1600.1 + 0.25 + 1e-6 and out["end"] == out["words"][-1]["end"]  # not 1606.1
    assert out["words"][0] == seg["words"][0]


def test_tighten_regions_do_not_shorten_words_inside_speech_or_touch_words_in_silence():
    inside = {"start": 10.0, "end": 12.0, "text": "промахнулся!", "words": [_word("промахнулся!", 10.0, 11.3)]}
    assert tighten_word_timings([inside], [(9.5, 12.5)])[0]["end"] == 11.3
    in_silence = {"start": 20.0, "end": 20.4, "text": "эм", "words": [_word("эм", 20.0, 20.4)]}  # no region at all around it
    assert tighten_word_timings([in_silence], [(1.0, 5.0)])[0]["end"] == 20.4
    assert tighten_word_timings([in_silence], [])[0]["end"] == 20.4 and tighten_word_timings([in_silence], None)[0]["end"] == 20.4
