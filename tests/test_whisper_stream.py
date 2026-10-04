import pytest

from src.core import transcriber
from src.core.transcriber import TranscriptionCancelled, _Progress, split_by_speaker, speakers_of


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
