import sys
import types

import numpy as np
import pytest

from src.core import transcriber
from src.core.transcriber import Transcriber, TranscriptionCancelled

AUDIO = np.zeros(16000 * 60, dtype=np.float32)


class _Env:
    """A fake `whisperx` + `whisperx.diarize` pair that records what the Transcriber does with them."""

    def __init__(self):
        self.diarizer_created = 0
        self.align_calls = 0
        self.model_calls = []
        self.load_model_kwargs = None
        self.align_error = None
        self.align_model_error = None
        self.diarize_error = None
        self.model_cancel_after = None  # callback that raises inside the model's progress callback
        self.segments = [{"start": 0.0, "end": 6.0, "text": "how are you fine thanks"}]
        self.words = [
            {"word": "how", "start": 0.0, "end": 0.5}, {"word": "are", "start": 0.5, "end": 1.0}, {"word": "you", "start": 1.0, "end": 1.5},
            {"word": "fine", "start": 2.0, "end": 2.6}, {"word": "thanks", "start": 2.6, "end": 3.4},
        ]
        self.word_speakers = ["SPEAKER_00", "SPEAKER_00", "SPEAKER_00", "SPEAKER_01", "SPEAKER_01"]


@pytest.fixture
def env(monkeypatch, tmp_path):
    e = _Env()

    class FakeModel:
        def transcribe(self, audio, batch_size=None, language=None, progress_callback=None):
            e.model_calls.append({"batch_size": batch_size, "language": language})
            if progress_callback:
                progress_callback(100.0)
            return {"language": language or "en", "segments": [dict(s) for s in e.segments]}

    class FakeDiarizationPipeline:
        def __init__(self, token=None, device=None):
            if e.diarize_error:
                raise e.diarize_error
            e.diarizer_created += 1

        def __call__(self, audio, progress_callback=None, **kwargs):
            if progress_callback:
                progress_callback(100.0)
            return "turns"

    def load_model(name, device, compute_type=None, download_root=None, asr_options=None):
        e.load_model_kwargs = {"name": name, "device": device, "compute_type": compute_type, "download_root": download_root, "asr_options": asr_options}
        return FakeModel()

    def load_align_model(language, device):
        if e.align_model_error:
            raise e.align_model_error
        return ("align-model", {"language": language})

    def align(segments, model, meta, audio, device, progress_callback=None, **_kwargs):
        e.align_calls += 1
        if e.align_error:
            raise e.align_error
        if progress_callback:
            progress_callback(100.0)
        return {"segments": [dict(segments[0], words=[dict(w) for w in e.words])]}

    def assign_word_speakers(turns, result):
        seg = dict(result["segments"][0])
        seg["words"] = [dict(w, speaker=s) for w, s in zip(seg["words"], e.word_speakers)]
        seg["speaker"] = "SPEAKER_00"
        return {"segments": [seg]}

    whisperx = types.ModuleType("whisperx")
    whisperx.setup_logging = lambda **_kwargs: None
    whisperx.load_model = load_model
    whisperx.load_align_model = load_align_model
    whisperx.align = align
    whisperx.assign_word_speakers = assign_word_speakers
    diarize = types.ModuleType("whisperx.diarize")
    diarize.DiarizationPipeline = FakeDiarizationPipeline
    monkeypatch.setitem(sys.modules, "whisperx", whisperx)
    monkeypatch.setitem(sys.modules, "whisperx.diarize", diarize)
    monkeypatch.setattr(transcriber, "setup_model_env", lambda: None)
    monkeypatch.setattr(transcriber, "get_models_dir", lambda: str(tmp_path))
    monkeypatch.setattr(transcriber, "get_hardware_status", lambda: {"vram_gb": 24.0})
    monkeypatch.setattr(Transcriber, "_ensure_nltk_data", staticmethod(lambda logger=None: True))
    return e


def _engine(logs=None, device="cuda"):
    return Transcriber("tiny", device, logs.append if logs is not None else None)


def test_model_is_loaded_with_the_expected_options(env, tmp_path, monkeypatch):
    monkeypatch.setattr(transcriber, "get_models_dir", lambda: str(tmp_path))
    _engine()
    assert env.load_model_kwargs["compute_type"] == "float16" and env.load_model_kwargs["download_root"] == str(tmp_path / "whisper")
    assert env.load_model_kwargs["asr_options"]["condition_on_previous_text"] is False and env.load_model_kwargs["asr_options"]["initial_prompt"]
    _engine(device="cpu")
    assert env.load_model_kwargs["compute_type"] == "int8"


def test_full_pipeline_aligns_diarizes_and_splits_by_speaker(env):
    logs = []
    out = _engine(logs).transcribe(AUDIO, None, "hf_x")
    assert out.language == "en" and out.diarized and out.speakers == ["SPEAKER_00", "SPEAKER_01"] and not out.align_failed
    assert [(s["speaker"], s["text"]) for s in out.segments] == [("SPEAKER_00", "how are you"), ("SPEAKER_01", "fine thanks")]
    assert any("Detected language: en" in line for line in logs)  # Auto-Detect is reported
    assert any("Speaker diarization of 1 min" in line for line in logs)
    assert env.model_calls == [{"batch_size": 16, "language": None}]


def test_alignment_result_is_tightened_so_lines_do_not_run_on_over_silence(env):
    env.words = [{"word": "how", "start": 0.0, "end": 0.4}, {"word": "are", "start": 0.4, "end": 0.7}, {"word": "you", "start": 0.7, "end": 9.0}]
    env.word_speakers = ["SPEAKER_00"] * 3
    out = _engine().transcribe(AUDIO, "en", None)
    assert out.segments[0]["end"] < 2.0  # "you" is not allowed to last 8 s, so the line ends where the speech does
    assert out.segments[0]["words"][-1]["end"] < 1.7


def test_vad_speech_regions_cap_word_ends(env, monkeypatch):
    env.words = [{"word": "how", "start": 0.0, "end": 0.4}, {"word": "are", "start": 0.4, "end": 0.9}, {"word": "you", "start": 0.9, "end": 1.9}]
    env.word_speakers = ["SPEAKER_00"] * 3
    monkeypatch.setattr(Transcriber, "_speech_regions", lambda self, audio: [(0.0, 1.2)])  # the VAD heard speech only until 1.2 s
    out = _engine().transcribe(AUDIO, "en", None)
    assert out.segments[0]["words"][-1]["end"] <= 1.2 + 0.25 + 1e-6


def test_a_vad_that_cannot_be_asked_is_not_fatal(env):
    logs = []
    engine = _engine(logs)
    assert engine._speech_regions(AUDIO) is None  # the fake whisperx has no VAD: reported, not raised
    assert any("Speech regions unavailable" in line for line in logs)
    out = engine.transcribe(AUDIO, "en", None)
    assert out.segments and not out.align_failed  # word-length limits still apply


def test_small_gpus_use_a_smaller_batch(env, monkeypatch):
    monkeypatch.setattr(transcriber, "get_hardware_status", lambda: {"vram_gb": 4.0})
    _engine().transcribe(AUDIO, "en")
    assert env.model_calls[0]["batch_size"] == transcriber.LOW_VRAM_BATCH_SIZE


def test_without_a_token_speakers_are_skipped_silently(env):
    logs = []
    out = _engine(logs).transcribe(AUDIO, "en", None)
    assert not out.diarized and out.speakers == [] and env.diarizer_created == 0
    assert not any("token" in line.lower() for line in logs)  # speaker detection is opt-in: nothing to warn about


def test_diarization_failure_degrades_instead_of_failing(env):
    env.diarize_error = RuntimeError("401 gated repo")
    logs = []
    out = _engine(logs).transcribe(AUDIO, "en", "hf_bad")
    assert not out.diarized and out.speakers == [] and out.segments and out.segments[0]["words"]  # alignment result is kept
    assert any("Speaker diarization unavailable (401 gated repo)" in line for line in logs)


def test_alignment_failure_keeps_segments_and_is_reported(env):
    env.align_error = RuntimeError("cuda oom")
    logs = []
    out = _engine(logs).transcribe(AUDIO, "en", None)
    assert out.align_failed and out.segments == [{"start": 0.0, "end": 6.0, "text": "how are you fine thanks"}]
    assert any("Word alignment failed (cuda oom)" in line for line in logs)


def test_language_without_an_alignment_model_is_not_a_transient_failure(env):
    env.align_model_error = ValueError("no align model for 'xx'")
    out = _engine().transcribe(AUDIO, "xx", None)
    assert not out.align_failed and env.align_calls == 0  # permanent: the result may be cached


def test_missing_nltk_data_skips_alignment_and_is_reported(env, monkeypatch):
    monkeypatch.setattr(Transcriber, "_ensure_nltk_data", staticmethod(lambda logger=None: False))
    out = _engine().transcribe(AUDIO, "en", None)
    assert out.align_failed and env.align_calls == 0


def test_cancellation_always_propagates(env):
    state = {"cancel": False}
    engine = _engine()
    state["cancel"] = True
    with pytest.raises(TranscriptionCancelled):
        engine.transcribe(AUDIO, "en", "hf_x", lambda: state["cancel"])  # the very first progress callback aborts

    env.align_error = TranscriptionCancelled()  # cancelled while aligning: not swallowed as an "alignment failure"
    with pytest.raises(TranscriptionCancelled):
        _engine().transcribe(AUDIO, "en", None)
    env.align_error = None
    env.diarize_error = TranscriptionCancelled()  # ... nor as a diarization failure
    with pytest.raises(TranscriptionCancelled):
        _engine().transcribe(AUDIO, "en", "hf_x")


def test_close_releases_models_and_is_idempotent(env):
    engine = _engine()
    engine.transcribe(AUDIO, "en", "hf_x")
    engine.close()
    engine.close()
    assert engine._model is None and engine._diarizer is None and engine._align_models == {}


def test_ensure_nltk_data_failure_is_not_fatal(monkeypatch):
    nltk = types.ModuleType("nltk")
    nltk.data = types.SimpleNamespace(find=lambda name: (_ for _ in ()).throw(LookupError(name)))
    nltk.download = lambda *a, **k: False
    monkeypatch.setitem(sys.modules, "nltk", nltk)
    monkeypatch.setenv("NLTK_DATA", "x")
    logs = []
    assert Transcriber._ensure_nltk_data(logs.append) is False
    assert any("punkt_tab" in line for line in logs)
    nltk.download = lambda *a, **k: True
    assert Transcriber._ensure_nltk_data() is True
