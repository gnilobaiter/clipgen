from unittest.mock import MagicMock

from src.utils.hardware import get_hardware_status, log_hardware_info


def test_cuda_detected_gpu_mode(monkeypatch):
    mock_props = MagicMock()
    mock_props.total_memory = 16 * (1024 ** 3)  # 16 GB

    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    monkeypatch.setattr("torch.cuda.get_device_name", lambda idx: "NVIDIA GeForce RTX 4080")
    monkeypatch.setattr("torch.cuda.get_device_properties", lambda idx: mock_props)

    status = get_hardware_status()
    assert status["cuda_available"] is True
    assert status["device"] == "cuda"
    assert status["gpu_name"] == "NVIDIA GeForce RTX 4080"
    assert status["vram_gb"] == 16.0
    assert status["fp16_supported"] is True

    logs = []
    summary = log_hardware_info(logs.append)
    assert "GPU: NVIDIA GeForce RTX 4080" in summary
    assert len(logs) == 2
    assert "NVIDIA GeForce RTX 4080" in logs[0]
    assert "CUDA fp16" in logs[1]


def test_cuda_unavailable_cpu_mode(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    status = get_hardware_status()
    assert status["cuda_available"] is False
    assert status["device"] == "cpu"
    assert status["gpu_name"] is None
    assert status["fp16_supported"] is False

    logs = []
    summary = log_hardware_info(logs.append)
    assert "CPU Mode" in summary
    assert len(logs) == 2
    assert "No NVIDIA CUDA GPU detected" in logs[0]
    assert "Processing Mode: CPU" in logs[1]


def test_cuda_exception_handling(monkeypatch):
    def raise_err():
        raise RuntimeError("CUDA Driver Mismatch")

    monkeypatch.setattr("torch.cuda.is_available", raise_err)

    status = get_hardware_status()
    assert status["cuda_available"] is False
    assert status["device"] == "cpu"
    assert status["gpu_name"] is None


def test_log_hardware_info_default_logger(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    summary = log_hardware_info(logger=None)
    assert "CPU Mode" in summary
