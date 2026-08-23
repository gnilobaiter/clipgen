import logging
import sys
from unittest.mock import MagicMock

import main


def test_setup_logging_standard():
    logger = main.setup_logging()
    assert isinstance(logger, logging.Logger)
    assert logger.name == "main"


def test_setup_logging_reconfigure_exceptions(monkeypatch):
    class BadStream:
        def reconfigure(self, **kwargs):
            raise RuntimeError("Stream error")

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "stdout", BadStream())
    monkeypatch.setattr(sys, "stderr", BadStream())

    logger = main.setup_logging()
    assert isinstance(logger, logging.Logger)


def test_main_execution(monkeypatch):
    monkeypatch.setattr("src.utils.paths.inject_bin_to_path", lambda: None)
    monkeypatch.setattr("src.utils.hardware.log_hardware_info", lambda *args, **kwargs: "GPU: RTX 4080")

    mock_qapp = MagicMock()
    mock_qapp.exec.return_value = 0
    monkeypatch.setattr(main.QtWidgets, "QApplication", MagicMock(return_value=mock_qapp))

    mock_window = MagicMock()
    monkeypatch.setattr("main.ClipGenPySideApp", lambda: mock_window)
    monkeypatch.setattr(sys, "exit", lambda *_code: None)

    main.main()
    mock_window.show.assert_called_once()


def test_main_ctypes_exception(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")

    class FakeWindll:
        @property
        def shell32(self):
            raise RuntimeError("Shell32 unavailable")

    class FakeCtypes:
        windll = FakeWindll()

    monkeypatch.setitem(sys.modules, "ctypes", FakeCtypes())
    monkeypatch.setattr("src.utils.paths.inject_bin_to_path", lambda: None)
    monkeypatch.setattr("src.utils.hardware.log_hardware_info", lambda *args, **kwargs: "CPU")

    mock_qapp = MagicMock()
    mock_qapp.exec.return_value = 0
    monkeypatch.setattr(main.QtWidgets, "QApplication", MagicMock(return_value=mock_qapp))

    mock_window = MagicMock()
    monkeypatch.setattr("main.ClipGenPySideApp", lambda: mock_window)
    monkeypatch.setattr(sys, "exit", lambda *_code: None)

    main.main()
    mock_window.show.assert_called_once()

