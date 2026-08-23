"""Entry Point.

Orchestrates hardware capability detection (GPU/CUDA vs CPU),
runtime path configuration, and launches the PySide6 GUI workstation.
"""

from __future__ import annotations

import logging
import os
import sys

from PySide6 import QtWidgets

from src.ui.pyside_app import ClipGenPySideApp
from src.utils.hardware import log_hardware_info
from src.utils.paths import inject_bin_to_path


def setup_logging() -> logging.Logger:
    """Configures root logging with clean timestamped formatting."""
    if sys.platform == "win32":
        if hasattr(sys.stdout, "reconfigure"):
            try:
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
        if hasattr(sys.stderr, "reconfigure"):
            try:
                sys.stderr.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("main")


def main() -> None:
    """Main application entry point."""
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("jbahrvr.clipgenerator.app.1")
        except Exception:
            pass

    logger = setup_logging()
    logger.info("Starting jBahr's Clip Generator...")

    # Ensure project local bin/ tools (ffmpeg, ffprobe) are on PATH
    inject_bin_to_path()

    # Detect and log hardware execution mode (GPU CUDA acceleration vs CPU)
    hw_summary = log_hardware_info(logger.info)
    logger.info("Active Hardware Engine: %s", hw_summary)

    # Launch PySide6 GUI
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")

    window = ClipGenPySideApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
