import importlib.util
import os
import sys
import types

import pytest

SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "ensure_onnxruntime_gpu.py")


@pytest.fixture
def script():
    spec = importlib.util.spec_from_file_location("ensure_onnxruntime_gpu", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_ort(monkeypatch, providers):
    fake = types.ModuleType("onnxruntime")
    fake.get_available_providers = lambda: providers
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)


def test_nothing_to_do_when_the_cuda_provider_is_there(script, monkeypatch):
    _fake_ort(monkeypatch, ["CUDAExecutionProvider", "CPUExecutionProvider"])
    calls = []
    monkeypatch.setattr(script.subprocess, "call", lambda cmd: calls.append(cmd) or 0)
    assert script.main() == 0 and calls == []


def test_reinstalls_the_gpu_build_when_the_cpu_package_won(script, monkeypatch):
    _fake_ort(monkeypatch, ["CPUExecutionProvider"])
    calls = []
    monkeypatch.setattr(script.subprocess, "call", lambda cmd: calls.append(cmd) or 0)
    assert script.main() == 0
    assert calls == [["uv", "pip", "install", "--python", sys.executable, "--reinstall", "--no-deps", script.PIN]]


def test_a_broken_onnxruntime_counts_as_lost_and_a_locked_file_is_retried_once(script, monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime", None)  # `import onnxruntime` raises
    monkeypatch.setattr(script.time, "sleep", lambda s: None)
    results = iter([2, 0])
    calls = []
    monkeypatch.setattr(script.subprocess, "call", lambda cmd: calls.append(cmd) or next(results))
    assert script.main() == 0 and len(calls) == 2


def test_gives_up_after_one_retry(script, monkeypatch, capsys):
    _fake_ort(monkeypatch, [])
    monkeypatch.setattr(script.time, "sleep", lambda s: None)
    monkeypatch.setattr(script.subprocess, "call", lambda cmd: 2)
    assert script.main() == 2
    assert "close programs using .venv" in capsys.readouterr().out
