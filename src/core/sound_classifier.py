"""Pretrained AudioSet sound-event classification (YAMNet via ONNX Runtime, CUDA when available).

Whisper transcripts carry no information about laughter or screaming. YAMNet is a small (16 MB) MobileNet
trained on AudioSet that scores 521 sound classes per 0.96 s window (0.48 s hop). Only the laughter and
screaming/shouting classes are used here. On an RTX 4080 a 90-minute recording takes ~1.5 s.

The model file is downloaded once into the app-data folder from a pinned Hugging Face commit and verified
against a SHA256 before it is ever loaded.

GPU inference runs in a separate Python process (sound_worker.py). onnxruntime-gpu is built for CUDA 13 while
PyTorch (Whisper) ships CUDA 12, and both use identical DLL names (cudnn64_9.dll, ...). Loading both into one
Windows process makes the loader pick whichever came first and can kill the interpreter with 0xc0000139.
A child process has its own DLL namespace, so a crash there can never take the application down.
"""

import hashlib
import os
import subprocess
import sys
import tempfile
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from src.core.audio_events import group_runs
from src.utils.paths import get_app_data_path, get_project_root

MODEL_URL = "https://huggingface.co/andrelgomes/yamnet-onnx/resolve/8a03a1572569685c42fdbef54ff36435dbaaf689/yamnet.onnx"
MODEL_SHA256 = "1510041dce24a2e9e84ec546807ac408ae496da6d1ed41bc3ccba649623f8e19"
MODEL_FILENAME = "yamnet.onnx"
DOWNLOAD_TIMEOUT = 120

SAMPLE_RATE = 16000
FRAME_HOP = 0.48
FRAME_LENGTH = 0.96
MIN_SAMPLES = int(FRAME_LENGTH * SAMPLE_RATE)
CHUNK_FRAMES = 1250  # 600 s of hops per request keeps GPU/CPU memory small
CHUNK_SAMPLES = CHUNK_FRAMES * int(FRAME_HOP * SAMPLE_RATE)

# AudioSet class indices in yamnet_class_map.csv
LAUGHTER_CLASSES = (13, 14, 15, 16, 17, 18)  # Laughter, Baby laughter, Giggle, Snicker, Belly laugh, Chuckle
SCREAM_CLASSES = (6, 9, 11)  # Shout, Yell, Screaming
LAUGHTER_THRESHOLD = 0.30
SCREAM_THRESHOLD = 0.30
MERGE_GAP_FRAMES = 2
WORKER_TIMEOUT = 900

_cpu_session: Optional[Any] = None


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_model(logger: Optional[Callable[[str], None]] = None) -> str:
    """Returns the path of a verified YAMNet model, downloading it on first use. Raises on failure."""
    model_dir = os.path.join(get_app_data_path(), "models")
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, MODEL_FILENAME)
    if os.path.exists(path):
        if _sha256(path) == MODEL_SHA256:
            return path
        if logger: logger("⚠️ Cached sound model failed its integrity check - downloading it again.")

    if logger: logger("⬇️ Downloading the YAMNet sound-event model (16 MB, one time)...")
    fd, tmp_path = tempfile.mkstemp(dir=model_dir, suffix=".part")
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(MODEL_URL, timeout=DOWNLOAD_TIMEOUT) as response:
            for block in iter(lambda: response.read(1 << 20), b""):
                out.write(block)
        if _sha256(tmp_path) != MODEL_SHA256:
            raise RuntimeError("downloaded model does not match the expected SHA256")
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return path


def create_session(model_path: str, use_gpu: bool) -> Any:
    """ONNX Runtime session. With `use_gpu` the CUDA provider is preferred (falls back to CPU); without it no
    CUDA library is ever loaded, which is the only safe mode inside the main application process."""
    import onnxruntime as ort

    providers = ["CPUExecutionProvider"]
    if use_gpu:
        try:  # lets onnxruntime-gpu find the CUDA/cuDNN DLLs installed next to it
            ort.preload_dlls(directory="")
        except Exception:
            pass
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ort.InferenceSession(model_path, providers=providers)


def _get_cpu_session(model_path: str) -> Any:
    global _cpu_session
    if _cpu_session is None:
        _cpu_session = create_session(model_path, use_gpu=False)
    return _cpu_session


def _classify_isolated(audio: np.ndarray, model_path: str) -> Tuple[np.ndarray, str]:
    """Runs the classifier (GPU when possible) in a child process. Returns (scores, provider name)."""
    if getattr(sys, "frozen", False):
        raise RuntimeError("no separate Python interpreter in a frozen build")
    with tempfile.TemporaryDirectory(prefix="clipgen_sound_") as tmp:
        audio_path = os.path.join(tmp, "audio.npy")
        scores_path = os.path.join(tmp, "scores.npy")
        np.save(audio_path, np.asarray(audio, dtype=np.float32))
        proc = subprocess.run(
            [sys.executable, "-m", "src.core.sound_worker", model_path, audio_path, scores_path],
            cwd=get_project_root(), capture_output=True, text=True, timeout=WORKER_TIMEOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        if proc.returncode != 0 or not os.path.exists(scores_path):
            last_error = (proc.stderr.strip().splitlines() or ["no error output"])[-1][:200]
            raise RuntimeError(f"worker exited with code {proc.returncode & 0xffffffff:#x}: {last_error}")
        provider = next((line.split("=", 1)[1] for line in proc.stdout.splitlines() if line.startswith("provider=")), "unknown")
        return np.load(scores_path), provider


def classify_frames(audio: np.ndarray, session: Any) -> np.ndarray:
    """Scores per 0.48 s hop, shape (frames, 521); frame i starts at i * 0.48 s.

    Requests are 600 s + one hop long so that every request yields exactly CHUNK_FRAMES frames and the
    concatenated result stays aligned with the timeline (plain back-to-back chunks lose a frame each)."""
    if len(audio) == 0:
        return np.zeros((0, 521), dtype=np.float32)
    input_name = session.get_inputs()[0].name
    hop_samples = int(FRAME_HOP * SAMPLE_RATE)
    parts = []
    for start in range(0, len(audio), CHUNK_SAMPLES):
        is_last = start + CHUNK_SAMPLES >= len(audio)
        chunk = np.asarray(audio[start:start + CHUNK_SAMPLES + hop_samples], dtype=np.float32)
        target = max(len(chunk), MIN_SAMPLES) if is_last else CHUNK_SAMPLES + hop_samples
        if len(chunk) < target:  # keeps every non-final request at exactly CHUNK_FRAMES frames
            chunk = np.pad(chunk, (0, target - len(chunk)))
        scores = session.run(None, {input_name: chunk})[0]
        parts.append(scores if is_last else scores[:CHUNK_FRAMES])
    return np.concatenate(parts)


def scores_to_events(scores: np.ndarray) -> List[Dict[str, Any]]:
    """Merges consecutive confident frames into LAUGHTER / SCREAM events (strength = peak confidence in %)."""
    events: List[Dict[str, Any]] = []
    if scores.size == 0:
        return events
    for kind, classes, threshold in (("LAUGHTER", LAUGHTER_CLASSES, LAUGHTER_THRESHOLD), ("SCREAM", SCREAM_CLASSES, SCREAM_THRESHOLD)):
        peak = scores[:, list(classes)].max(axis=1)
        for first, last in group_runs(peak >= threshold, MERGE_GAP_FRAMES):
            events.append({
                "type": kind,
                "start": round(first * FRAME_HOP, 2),
                "end": round(last * FRAME_HOP + FRAME_LENGTH, 2),
                "strength": int(min(100, round(float(peak[first:last + 1].max()) * 100))),
            })
    return sorted(events, key=lambda e: (e["start"], e["type"]))


def detect_sound_events(audio: np.ndarray, logger: Optional[Callable[[str], None]] = None) -> List[Dict[str, Any]]:
    """Laughter / screaming events from the classifier. Any failure (no network for the first model download,
    missing onnxruntime, driver problems) is reported through `logger` and yields no events."""
    try:
        model_path = ensure_model(logger)
    except Exception as e:
        if logger: logger(f"⚠️ Sound-event classifier unavailable ({e}); continuing without laughter/scream tags.")
        return []
    try:
        scores, provider = _classify_isolated(audio, model_path)
        if logger: logger(f"🎧 Sound-event classifier finished on {'GPU / CUDA' if provider == 'CUDAExecutionProvider' else 'CPU'} (separate process).")
    except Exception as e:
        if logger: logger(f"⚠️ GPU classifier process failed ({e}); retrying on CPU inside the app...")
        try:
            scores = classify_frames(audio, _get_cpu_session(model_path))
        except Exception as e2:
            if logger: logger(f"⚠️ Sound-event classifier unavailable ({e2}); continuing without laughter/scream tags.")
            return []
    return scores_to_events(scores)
