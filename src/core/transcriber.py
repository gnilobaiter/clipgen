"""WhisperX transcription: batched faster-whisper, wav2vec2 word alignment and pyannote speaker diarization.

Every model (Whisper, alignment, diarization) is stored under <project root>/models. The heavy imports are done
lazily so that importing this module (and the rest of the app) stays cheap and testable without the GPU stack.
"""

import bisect
import gc
import logging
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from src.utils.hardware import get_hardware_status
from src.utils.paths import get_data_dir, get_models_dir, setup_model_env

Logger = Optional[Callable[[str], None]]

# Same set as faster_whisper.utils._MODELS; only used when faster-whisper itself cannot be imported.
FALLBACK_MODELS: Tuple[str, ...] = (
    "tiny.en", "tiny", "base.en", "base", "small.en", "small", "medium.en", "medium",
    "large-v1", "large-v2", "large-v3", "large", "distil-large-v2", "distil-medium.en", "distil-small.en",
    "distil-large-v3", "distil-large-v3.5", "large-v3-turbo", "turbo",
)

GPU_COMPUTE_TYPE = "float16"
CPU_COMPUTE_TYPE = "int8"
BATCH_SIZE = 16
LOW_VRAM_BATCH_SIZE = 8
LOW_VRAM_GB = 6.0

ASR_OPTIONS: Dict[str, Any] = {
    "condition_on_previous_text": False,
    "initial_prompt": "Gameplay livestream recording with conversational banter, laughter, discord chat, and gaming terms.",
}

MIN_RUN_WORDS = 2  # a speaker run shorter than this (and MIN_RUN_SECONDS) is diarization flicker, not a turn
MIN_RUN_SECONDS = 0.6


class TranscriptionCancelled(Exception):
    """Raised from inside a WhisperX progress callback to abort a running transcription."""


@dataclass
class TranscriptionOutput:
    language: Optional[str]
    segments: List[Dict[str, Any]]
    diarized: bool = False
    speakers: List[str] = field(default_factory=list)
    align_failed: bool = False  # word alignment was possible but failed (transient): the transcript is coarser than it should be


def available_models() -> List[str]:
    """Every model faster-whisper (and therefore WhisperX) can load, so the UI never drifts from the library."""
    try:
        from faster_whisper.utils import available_models as fw_available
        models = list(fw_available())
        if models:
            return models
    except Exception:
        pass
    return list(FALLBACK_MODELS)


def _model_repo(name: str) -> str:
    try:
        from faster_whisper.utils import _MODELS
        return _MODELS.get(name, "")
    except Exception:
        return ""


def whisper_download_dir() -> str:
    return os.path.join(get_models_dir(), "whisper")


def is_model_downloaded(name: str) -> bool:
    """faster-whisper keeps models as <download_root>/models--<org>--<repo>."""
    repo = _model_repo(name)
    return bool(repo) and os.path.isdir(os.path.join(whisper_download_dir(), f"models--{repo.replace('/', '--')}"))


def resolve_hf_token(config: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """HuggingFace token for the diarization model: Settings field -> HF_TOKEN env -> data/.hf_token."""
    token = str(((config or {}).get("huggingface") or {}).get("token", "")).strip()
    if token:
        return token
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    token_file = os.path.join(get_data_dir(), ".hf_token")
    try:
        with open(token_file, "r", encoding="utf-8") as f:
            return f.read().strip() or None
    except OSError:
        return None


def pick_batch_size() -> int:
    vram = get_hardware_status().get("vram_gb")
    return LOW_VRAM_BATCH_SIZE if vram is not None and vram < LOW_VRAM_GB else BATCH_SIZE


class _Progress:
    """A WhisperX progress callback (0-100): aborts on cancel, logs every 25%."""

    def __init__(self, logger: Logger, is_cancelled: Optional[Callable[[], bool]], label: str):
        self.logger, self.is_cancelled, self.label = logger, is_cancelled, label
        self._next_log = 0

    def __call__(self, percent: float) -> None:
        if self.is_cancelled and self.is_cancelled():
            raise TranscriptionCancelled()
        if self.logger and percent >= self._next_log:
            self.logger(f"⏳ {self.label}: {int(percent)}%")
            self._next_log = (int(percent) // 25 + 1) * 25


MAX_WORD_SECONDS = 2.0  # even a drawn-out word is not longer than this
BASE_WORD_SECONDS = 0.25
SECONDS_PER_LETTER = 0.1


def _word_cap(word: str) -> float:
    """The longest a spoken word of this length plausibly lasts (a 6-letter word ~ 0.85 s, never more than MAX_WORD_SECONDS)."""
    letters = sum(ch.isalnum() for ch in word)
    return min(MAX_WORD_SECONDS, BASE_WORD_SECONDS + SECONDS_PER_LETTER * letters)


SPEECH_END_PAD = 0.25  # a word may end this long after the VAD's speech region ends (the VAD cuts at a probability threshold)
SPEECH_START_SLACK = 0.3  # a word that starts this close before a speech region still belongs to it


def _region_end_for(start: float, regions: List[Tuple[float, float]], starts: List[float]) -> Optional[float]:
    """End of the VAD speech region the word (starting at `start`) belongs to, None when it starts in silence."""
    i = bisect.bisect_right(starts, start + SPEECH_START_SLACK) - 1
    if i < 0:
        return None
    region_start, region_end = regions[i]
    return region_end if start < region_end + SPEECH_END_PAD else None


def tighten_word_timings(segments: List[Dict[str, Any]], speech: Optional[List[Tuple[float, float]]] = None) -> List[Dict[str, Any]]:
    """WhisperX's forced alignment stretches the LAST word of a line over the silence (or non-speech) that follows it:
    in a real stream 1 line in 3 ended with a "word" lasting 1.5 s - 25 s (a normal word lasts 0.2 s), so every line looked
    like it ran on for seconds, pauses vanished and clips were cut long after the joke was over. Each word is limited to
    what it can plausibly last and, when the VAD's speech regions are known (`speech`: sorted (start, end) seconds), to the
    end of the region it was spoken in. A line ends where its last word really ends. Lines without word timings are kept."""
    regions = sorted(speech) if speech else []
    starts = [r[0] for r in regions]
    out: List[Dict[str, Any]] = []
    for seg in segments:
        words = seg.get("words")
        if not words:
            out.append(seg)
            continue
        tight = []
        for w in words:
            if "start" in w and "end" in w:
                end = min(float(w["end"]), float(w["start"]) + _word_cap(str(w.get("word", ""))))
                region_end = _region_end_for(float(w["start"]), regions, starts) if regions else None
                if region_end is not None:
                    end = min(end, max(float(w["start"]), region_end + SPEECH_END_PAD))
                w = dict(w, end=round(end, 3))
            tight.append(w)
        timed = [w for w in tight if "start" in w and "end" in w]
        entry = dict(seg, words=tight)
        if timed:
            entry["end"] = round(max(float(seg["start"]), min(float(seg["end"]), max(float(w["end"]) for w in timed))), 3)
        out.append(entry)
    return out


def split_by_speaker(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """A WhisperX segment can contain several speakers. Splits such a segment at every speaker change (judged by its
    words), so each transcript line has one speaker. Tiny runs (diarization flicker) are absorbed by their neighbour."""
    out: List[Dict[str, Any]] = []
    for seg in segments:
        words = [w for w in (seg.get("words") or []) if isinstance(w, dict)]
        if len({w.get("speaker") for w in words if w.get("speaker")}) <= 1:
            out.append(seg)
            continue

        runs: List[Dict[str, Any]] = []
        current = None
        for w in words:
            speaker = w.get("speaker") or (runs[-1]["speaker"] if runs else None)
            if not runs or speaker != runs[-1]["speaker"]:
                runs.append({"speaker": speaker, "words": []})
            runs[-1]["words"].append(w)
            current = runs[-1]
        if current is None:
            out.append(seg)
            continue

        def span(run: Dict[str, Any]) -> float:
            timed = [w for w in run["words"] if "start" in w and "end" in w]
            return float(timed[-1]["end"]) - float(timed[0]["start"]) if timed else 0.0

        merged: List[Dict[str, Any]] = []
        for i, run in enumerate(runs):
            tiny = len(run["words"]) < MIN_RUN_WORDS and span(run) < MIN_RUN_SECONDS
            if tiny and (merged or i + 1 < len(runs)):
                if merged:
                    merged[-1]["words"].extend(run["words"])
                else:
                    runs[i + 1]["words"][0:0] = run["words"]
                continue
            if merged and merged[-1]["speaker"] == run["speaker"]:
                merged[-1]["words"].extend(run["words"])
            else:
                merged.append({"speaker": run["speaker"], "words": list(run["words"])})
        if len(merged) <= 1:
            out.append(seg)
            continue

        spaced = len(str(seg.get("text", "")).split()) > 1
        for run in merged:
            timed = [w for w in run["words"] if "start" in w and "end" in w]
            if not timed:
                continue
            piece = dict(seg)
            piece["words"] = run["words"]
            piece["start"], piece["end"] = float(timed[0]["start"]), float(timed[-1]["end"])
            piece["text"] = (" " if spaced else "").join(str(w.get("word", "")) for w in run["words"]).strip()
            if run["speaker"]:
                piece["speaker"] = run["speaker"]
            else:
                piece.pop("speaker", None)
            out.append(piece)
    return out


def speakers_of(segments: List[Dict[str, Any]]) -> List[str]:
    return sorted({str(s["speaker"]) for s in segments if s.get("speaker")})


class Transcriber:
    """One loaded Whisper model with lazily created alignment / diarization models. Call close() to free the VRAM."""

    def __init__(self, model_name: str, device: str, logger: Logger = None):
        setup_model_env()
        warnings.filterwarnings("ignore", message=r"(?s).*torchcodec is not installed correctly.*")
        warnings.filterwarnings("ignore", message=r".*TensorFloat-32.*")
        warnings.filterwarnings("ignore", message=r".*degrees of freedom is <= 0.*")
        import whisperx  # noqa: PLC0415 (heavy; needs the env set above)
        whisperx.setup_logging(level="warning")
        for name in ("lightning.pytorch", "lightning.fabric", "pytorch_lightning", "huggingface_hub.file_download"):
            logging.getLogger(name).setLevel(logging.ERROR)

        self._whisperx = whisperx
        self.model_name, self.device, self.logger = model_name, device, logger
        self._align_ready = self._ensure_nltk_data(logger)
        compute_type = GPU_COMPUTE_TYPE if device == "cuda" else CPU_COMPUTE_TYPE
        os.makedirs(whisper_download_dir(), exist_ok=True)
        self._model = whisperx.load_model(model_name, device, compute_type=compute_type,
                                          download_root=whisper_download_dir(), asr_options=dict(ASR_OPTIONS))
        self._align_models: Dict[str, Optional[tuple]] = {}
        self._diarizer: Any = None

    @staticmethod
    def _ensure_nltk_data(logger: Logger = None) -> bool:
        """Word alignment tokenizes the text with nltk punkt_tab; downloaded once into data/models/nltk.
        False (and a warning) when it is unavailable: transcription then works without word alignment."""
        try:
            import nltk
            try:
                nltk.data.find("tokenizers/punkt_tab")
                return True
            except LookupError:
                if nltk.download("punkt_tab", download_dir=os.environ["NLTK_DATA"], quiet=True):
                    return True
        except Exception as e:
            if logger: logger(f"⚠️ nltk error: {e}")
        if logger: logger("⚠️ Could not get the nltk 'punkt_tab' data: word alignment is skipped (segment timestamps only).")
        return False

    def _align_model(self, language: str) -> Optional[tuple]:
        if language not in self._align_models:
            try:
                self._align_models[language] = self._whisperx.load_align_model(language, self.device)
            except Exception as e:
                if self.logger: self.logger(f"⚠️ No word-alignment model for '{language}' ({e}); using segment timestamps only.")
                self._align_models[language] = None
        return self._align_models[language]

    def _diarize(self, audio: np.ndarray, segments: List[Dict[str, Any]], token: str, is_cancelled: Optional[Callable[[], bool]]) -> List[Dict[str, Any]]:
        from whisperx.diarize import DiarizationPipeline
        if self.logger:
            self.logger(f"🗣️ Speaker diarization of {len(audio) / 16000 / 60:.0f} min of audio (the whole recording at once: long streams need a lot of memory)...")
        if self._diarizer is None:
            self._diarizer = DiarizationPipeline(token=token, device=self.device)
        turns = self._diarizer(audio, progress_callback=_Progress(self.logger, is_cancelled, "Speaker diarization"))
        return self._whisperx.assign_word_speakers(turns, {"segments": segments})["segments"]

    def _speech_regions(self, audio: np.ndarray) -> Optional[List[Tuple[float, float]]]:
        """Where the WhisperX VAD (pyannote) hears speech, as (start, end) seconds; None when that cannot be asked."""
        try:
            from whisperx.vads.pyannote import Binarize, Pyannote
            params = getattr(self._model, "_vad_params", None) or {}
            scores = self._model.vad_model({"waveform": Pyannote.preprocess_audio(audio), "sample_rate": 16000})
            active = Binarize(onset=params.get("vad_onset", 0.5), offset=params.get("vad_offset", 0.363))(scores)
            return [(float(s.start), float(s.end)) for s in active.get_timeline().support()]
        except Exception as e:
            if self.logger: self.logger(f"ℹ️ Speech regions unavailable ({e}); word ends are limited by word length only.")
            return None

    def transcribe(self, audio: np.ndarray, language: Optional[str], hf_token: Optional[str] = None,
                   is_cancelled: Optional[Callable[[], bool]] = None) -> TranscriptionOutput:
        """Speech recognition -> word alignment -> speaker diarization (only when a `hf_token` is given: speaker detection
        is opt-in in Settings). Alignment and diarization failures degrade the result (no words / no speakers) instead of
        failing the whole transcription; cancellation always propagates."""
        result = self._model.transcribe(audio, batch_size=pick_batch_size(), language=language,
                                        progress_callback=_Progress(self.logger, is_cancelled, "Transcribing"))
        detected = result.get("language") or language
        segments = result["segments"]
        if self.logger and not language and detected:
            self.logger(f"🌐 Detected language: {detected}")

        align_failed = bool(detected) and not self._align_ready  # no nltk data: alignment could not even be tried
        align = self._align_model(detected) if detected and self._align_ready else None
        if align is not None:
            try:
                aligned = self._whisperx.align(
                    segments, *align, audio, self.device, return_char_alignments=False,
                    progress_callback=_Progress(self.logger, is_cancelled, "Aligning words"))["segments"]
                segments = tighten_word_timings(aligned, self._speech_regions(audio))
            except TranscriptionCancelled:
                raise
            except Exception as e:
                align_failed = True
                if self.logger: self.logger(f"⚠️ Word alignment failed ({e}); using segment timestamps only.")

        diarized = False
        if hf_token:
            try:
                segments = split_by_speaker(self._diarize(audio, segments, hf_token, is_cancelled))
                diarized = True
            except TranscriptionCancelled:
                raise
            except Exception as e:  # includes out-of-memory on very long recordings
                self._free_gpu_cache()
                if self.logger: self.logger(f"⚠️ Speaker diarization unavailable ({e}); continuing without speakers. "
                                            "Check the HuggingFace token and accept the terms of pyannote/speaker-diarization-community-1.")
        return TranscriptionOutput(detected, segments, diarized, speakers_of(segments), align_failed)

    def close(self) -> None:
        """Drops the models and frees GPU memory so YAMNet / NVENC are not starved."""
        self._model = None
        self._align_models.clear()
        self._diarizer = None
        self._free_gpu_cache()

    @staticmethod
    def _free_gpu_cache() -> None:
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
