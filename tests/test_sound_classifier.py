import hashlib
import io
import os
import subprocess
import sys
import types

import numpy as np
import pytest

from src.core import sound_classifier as sc
from src.core import sound_worker


class FakeSession:
    """Behaves like the YAMNet ONNX session: 0.96 s windows, 0.48 s hop. Column 0 carries the first sample of
    every window so tests can check that concatenated chunks stay aligned with the timeline."""

    class _Input:
        name = "waveform"

    def __init__(self, providers=("CPUExecutionProvider",)):
        self.requests = []
        self._providers = list(providers)

    def get_inputs(self):
        return [self._Input()]

    def get_providers(self):
        return self._providers

    def run(self, _outputs, feed):
        wave = feed["waveform"]
        self.requests.append(len(wave))
        hop = int(sc.FRAME_HOP * sc.SAMPLE_RATE)
        n = (len(wave) - sc.MIN_SAMPLES) // hop + 1
        scores = np.zeros((n, 521), dtype=np.float32)
        scores[:, 0] = wave[np.arange(n) * hop]
        return [scores]


# ------------------------------------------------------------------ events & chunking
def test_scores_to_events_merges_frames_and_maps_times():
    scores = np.zeros((40, 521), dtype=np.float32)
    scores[10:13, 15] = [0.4, 0.7, 0.35]  # Giggle
    scores[14, 13] = 0.5  # Laughter one frame later (gap of 1) -> same event
    scores[30, 11] = 0.6  # Screaming
    scores[20, 13] = 0.29  # below threshold
    events = sc.scores_to_events(scores)
    assert [e["type"] for e in events] == ["LAUGHTER", "SCREAM"]
    laugh, scream = events
    assert laugh["start"] == pytest.approx(10 * 0.48) and laugh["end"] == pytest.approx(14 * 0.48 + 0.96)
    assert laugh["strength"] == 70
    assert scream["start"] == pytest.approx(30 * 0.48) and scream["strength"] == 60
    assert sc.scores_to_events(np.zeros((0, 521))) == []
    assert sc.scores_to_events(np.zeros((10, 521))) == []


def test_classify_frames_stays_aligned_across_chunks():
    hop = int(sc.FRAME_HOP * sc.SAMPLE_RATE)
    total_frames = sc.CHUNK_FRAMES * 2 + 700
    audio = np.arange(total_frames * hop + sc.MIN_SAMPLES, dtype=np.float32)
    session = FakeSession()
    scores = sc.classify_frames(audio, session)
    assert len(session.requests) == 3
    np.testing.assert_array_equal(scores[:, 0], audio[np.arange(len(scores)) * hop])
    assert len(scores) >= total_frames  # no frame lost at the chunk seams


def test_classify_frames_pads_tiny_tail_and_handles_empty():
    session = FakeSession()
    assert sc.classify_frames(np.ones(100, dtype=np.float32), session).shape == (1, 521)
    assert sc.classify_frames(np.zeros(0, dtype=np.float32), session).shape == (0, 521)
    # a tail shorter than one hop after a full chunk must not shift the timeline
    scores = sc.classify_frames(np.ones(sc.CHUNK_SAMPLES + 100, dtype=np.float32), FakeSession())
    assert len(scores) == sc.CHUNK_FRAMES + 1


# ------------------------------------------------------------------ model download
class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def _use_payload(monkeypatch, tmp_path, payload: bytes, served=None):
    monkeypatch.setattr(sc, "get_app_data_path", lambda: str(tmp_path))
    monkeypatch.setattr(sc, "MODEL_SHA256", hashlib.sha256(payload).hexdigest())
    downloads = []

    def fake_urlopen(url, **_kwargs):
        downloads.append(url)
        return _FakeResponse(served if served is not None else payload)

    monkeypatch.setattr(sc.urllib.request, "urlopen", fake_urlopen)
    return downloads


def test_ensure_model_downloads_verifies_and_reuses(monkeypatch, tmp_path):
    downloads = _use_payload(monkeypatch, tmp_path, b"model-bytes")
    logs = []
    path = sc.ensure_model(logs.append)
    assert path == str(tmp_path / "models" / sc.MODEL_FILENAME) and open(path, "rb").read() == b"model-bytes"
    assert downloads == [sc.MODEL_URL] and any("Downloading" in line for line in logs)
    assert sc.MODEL_URL.startswith("https://") and "8a03a1572569685c42fdbef54ff36435dbaaf689" in sc.MODEL_URL  # pinned commit

    assert sc.ensure_model() == path
    assert len(downloads) == 1  # verified file is reused without touching the network


def test_ensure_model_redownloads_corrupted_file(monkeypatch, tmp_path):
    downloads = _use_payload(monkeypatch, tmp_path, b"good")
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / sc.MODEL_FILENAME).write_bytes(b"corrupted")
    logs = []
    path = sc.ensure_model(logs.append)
    assert open(path, "rb").read() == b"good" and len(downloads) == 1
    assert any("integrity" in line for line in logs)


def test_ensure_model_rejects_tampered_download_and_leaves_no_files(monkeypatch, tmp_path):
    _use_payload(monkeypatch, tmp_path, b"expected", served=b"evil")
    with pytest.raises(RuntimeError, match="SHA256"):
        sc.ensure_model()
    assert os.listdir(tmp_path / "models") == []


# ------------------------------------------------------------------ sessions
def _fake_ort(monkeypatch, providers, preload_raises=False):
    created = {"preloaded": False}

    def preload_dlls(directory=None):
        created["preloaded"] = True
        if preload_raises:
            raise OSError("no dlls")

    def make_session(path, providers):
        created["path"], created["providers"] = path, providers
        return FakeSession(providers)

    fake = types.SimpleNamespace(get_available_providers=lambda: providers, preload_dlls=preload_dlls, InferenceSession=make_session)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)
    return created


def test_create_session_gpu_prefers_cuda_and_tolerates_preload_failure(monkeypatch):
    created = _fake_ort(monkeypatch, ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"])
    sc.create_session("m.onnx", use_gpu=True)
    assert created["providers"] == ["CUDAExecutionProvider", "CPUExecutionProvider"] and created["preloaded"]

    created = _fake_ort(monkeypatch, ["CUDAExecutionProvider", "CPUExecutionProvider"], preload_raises=True)
    sc.create_session("m.onnx", use_gpu=True)
    assert created["providers"][0] == "CUDAExecutionProvider"

    created = _fake_ort(monkeypatch, ["CPUExecutionProvider"])
    sc.create_session("m.onnx", use_gpu=True)
    assert created["providers"] == ["CPUExecutionProvider"]


def test_create_session_cpu_mode_never_loads_cuda_libraries(monkeypatch):
    created = _fake_ort(monkeypatch, ["CUDAExecutionProvider", "CPUExecutionProvider"])
    sc.create_session("m.onnx", use_gpu=False)
    assert created["providers"] == ["CPUExecutionProvider"] and created["preloaded"] is False


def test_cpu_session_is_cached(monkeypatch):
    monkeypatch.setattr(sc, "_cpu_session", None)
    _fake_ort(monkeypatch, ["CPUExecutionProvider"])
    first = sc._get_cpu_session("m.onnx")
    assert sc._get_cpu_session("m.onnx") is first


# ------------------------------------------------------------------ isolated worker
def _fake_run(monkeypatch, returncode=0, stdout="provider=CUDAExecutionProvider\n", stderr="", write_scores=True):
    calls = {}

    def run(cmd, **kwargs):
        calls["cmd"], calls["kwargs"] = cmd, kwargs
        if write_scores:
            np.save(cmd[-1], np.full((3, 521), 0.5, dtype=np.float32))
        return subprocess.CompletedProcess(cmd, returncode, stdout, stderr)

    monkeypatch.setattr(sc.subprocess, "run", run)
    return calls


def test_classify_isolated_runs_worker_module_in_project_root(monkeypatch):
    calls = _fake_run(monkeypatch)
    scores, provider = sc._classify_isolated(np.zeros(1000, dtype=np.float64), "model.onnx")
    assert provider == "CUDAExecutionProvider" and scores.shape == (3, 521)
    assert calls["cmd"][:3] == [sys.executable, "-m", "src.core.sound_worker"] and calls["cmd"][3] == "model.onnx"
    assert os.path.isdir(calls["kwargs"]["cwd"]) and os.path.exists(os.path.join(calls["kwargs"]["cwd"], "src", "core", "sound_worker.py"))
    assert calls["kwargs"]["timeout"] == sc.WORKER_TIMEOUT


def test_classify_isolated_reports_crash_with_exit_code(monkeypatch):
    _fake_run(monkeypatch, returncode=0xC0000139 - (1 << 32), stderr="boom\nEntry point not found\n", write_scores=False)
    with pytest.raises(RuntimeError, match="0xc0000139.*Entry point not found"):
        sc._classify_isolated(np.zeros(1000, dtype=np.float32), "model.onnx")


def test_classify_isolated_refuses_frozen_builds(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    with pytest.raises(RuntimeError, match="frozen"):
        sc._classify_isolated(np.zeros(10, dtype=np.float32), "m")


def test_worker_main_writes_scores_and_reports_provider(monkeypatch, tmp_path, capsys):
    audio_path, out_path = tmp_path / "a.npy", tmp_path / "s.npy"
    np.save(audio_path, np.zeros(sc.SAMPLE_RATE * 3, dtype=np.float32))
    monkeypatch.setattr(sound_worker.sound_classifier, "create_session", lambda path, use_gpu: FakeSession(("CUDAExecutionProvider",)) if use_gpu else None)
    sound_worker.main(["model.onnx", str(audio_path), str(out_path)])
    assert np.load(out_path).shape[1] == 521
    assert "provider=CUDAExecutionProvider" in capsys.readouterr().out


# ------------------------------------------------------------------ end to end (all mocked)
def test_detect_sound_events_uses_isolated_process(monkeypatch):
    scores = np.zeros((50, 521), dtype=np.float32)
    scores[20, 13] = 0.8
    monkeypatch.setattr(sc, "ensure_model", lambda logger=None: "model.onnx")
    monkeypatch.setattr(sc, "_classify_isolated", lambda audio, path: (scores, "CUDAExecutionProvider"))
    logs = []
    events = sc.detect_sound_events(np.zeros(sc.SAMPLE_RATE * 30, dtype=np.float32), logs.append)
    assert [e["type"] for e in events] == ["LAUGHTER"] and events[0]["start"] == pytest.approx(20 * 0.48)
    assert any("Running the YAMNet" in line for line in logs)  # visible start message
    finished = [line for line in logs if "YAMNet finished" in line]
    assert finished and "GPU / CUDA" in finished[0] and "separate process" in finished[0] and "1 laughter and 0 scream" in finished[0]


def test_detect_sound_events_falls_back_to_cpu_when_worker_crashes(monkeypatch):
    def crash(audio, path):
        raise RuntimeError("worker exited with code 0xc0000139")

    class Hot(FakeSession):
        def run(self, outputs, feed):
            scores = super().run(outputs, feed)[0]
            scores[3, 11] = 0.9
            return [scores]

    monkeypatch.setattr(sc, "ensure_model", lambda logger=None: "model.onnx")
    monkeypatch.setattr(sc, "_classify_isolated", crash)
    monkeypatch.setattr(sc, "_get_cpu_session", lambda path: Hot())
    logs = []
    events = sc.detect_sound_events(np.zeros(sc.SAMPLE_RATE * 10, dtype=np.float32), logs.append)
    assert [e["type"] for e in events] == ["SCREAM"]
    assert any("retrying on CPU" in line for line in logs)
    assert any("YAMNet finished on CPU (inside the app)" in line and "0 laughter and 1 scream" in line for line in logs)


def test_detect_sound_events_degrades_gracefully(monkeypatch):
    audio = np.zeros(sc.SAMPLE_RATE * 5, dtype=np.float32)

    def no_network(logger=None):
        raise OSError("network is down")

    monkeypatch.setattr(sc, "ensure_model", no_network)
    logs = []
    assert sc.detect_sound_events(audio, logs.append) == []
    assert any("unavailable" in line and "network is down" in line for line in logs)

    def broken(*args, **kwargs):
        raise RuntimeError("cpu also broken")

    monkeypatch.setattr(sc, "ensure_model", lambda logger=None: "model.onnx")
    monkeypatch.setattr(sc, "_classify_isolated", broken)
    monkeypatch.setattr(sc, "_get_cpu_session", broken)
    logs = []
    assert sc.detect_sound_events(audio, logs.append) == []
    assert any("cpu also broken" in line for line in logs)
