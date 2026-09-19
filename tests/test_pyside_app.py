import os
import sys

from PySide6 import QtWidgets
import pytest

from src.core import config as config_manager
from src.ui.pyside_app import ClipExtractionWorker, ClipGenPySideApp, WorkerSignals


@pytest.fixture(scope="session")
def qapp():
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    return app


@pytest.fixture
def temp_config_env(tmp_path, monkeypatch):
    test_config_path = tmp_path / "config.json"
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    appdata_dir = tmp_path / "appdata"
    appdata_dir.mkdir(parents=True, exist_ok=True)

    default_cfg = config_manager.get_default_config()
    default_cfg["settings"]["clips_dir"] = str(clips_dir)
    config_manager.save_config(default_cfg, filepath=str(test_config_path))

    monkeypatch.setattr("src.core.config.get_config_path", lambda: str(test_config_path))
    monkeypatch.setattr("src.ui.pyside_app.config_manager.get_config_path", lambda: str(test_config_path))
    monkeypatch.setattr("src.ui.pyside_app.get_app_data_path", lambda: str(appdata_dir))
    monkeypatch.setattr("src.utils.paths.get_app_data_path", lambda: str(appdata_dir))
    monkeypatch.setattr("src.core.editor.get_app_data_path", lambda: str(appdata_dir))
    return str(test_config_path), str(clips_dir)


@pytest.mark.usefixtures("qapp")
def test_pyside_app_initialization(temp_config_env):
    window = ClipGenPySideApp()
    assert window.windowTitle() == "Clip Generator — AI Highlight Workstation"
    assert window.stacked_widget.count() == 4
    assert window.extractor_profile_combo.count() >= 1
    assert window.prompt_profile_combo.count() >= 1
    assert window.prompt_profile_combo.currentText() == "Default"
    assert window.prompt_edit_box.isReadOnly() is True
    assert window.save_prompt_btn.isEnabled() is False
    assert window.delete_profile_btn.isEnabled() is False
    assert window.whisper_lang_combo.currentText() == "Auto-Detect"
    assert window.whisper_model_combo.currentText() == "medium"
    window.close()


@pytest.mark.usefixtures("qapp")
def test_pyside_app_navigation(temp_config_env):
    window = ClipGenPySideApp()
    assert window.stacked_widget.currentIndex() == 0

    # Switch to Prompt tab
    window.nav_prompt_btn.click()
    assert window.stacked_widget.currentIndex() == 1

    # Switch to Settings tab
    window.nav_settings_btn.click()
    assert window.stacked_widget.currentIndex() == 2

    # Switch to Gallery tab
    window.nav_gallery_btn.click()
    assert window.stacked_widget.currentIndex() == 3

    window.close()


@pytest.mark.usefixtures("qapp")
def test_pyside_app_save_settings(temp_config_env, monkeypatch):
    cfg_file, _ = temp_config_env
    window = ClipGenPySideApp()

    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *args, **kwargs: None)

    window.openai_key_edit.setText("sk-test-12345")
    window.whisper_lang_combo.setCurrentText("Russian")
    window.whisper_model_combo.setCurrentText("turbo")
    window.hw_encode_checkbox.setChecked(True)
    assert window.deepseek_thinking_checkbox.isChecked()  # on by default
    window.deepseek_thinking_checkbox.setChecked(False)

    window._save_settings()

    saved = config_manager.load_config(cfg_file)
    assert saved["settings"]["deepseek_thinking"] is False
    assert saved["openai"]["api_key"] == "sk-test-12345"
    assert saved["openai"]["whisper_language"] == "Russian"
    assert saved["openai"]["whisper_model"] == "turbo"
    assert saved["settings"]["hardware_encoding"] is True

    window.close()


@pytest.mark.usefixtures("qapp")
def test_pyside_app_prompt_profile_management(temp_config_env, monkeypatch):
    cfg_file, _ = temp_config_env
    window = ClipGenPySideApp()

    # Verify Default cannot be deleted or saved
    warnings = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda _w, title, msg: warnings.append(title))
    window._delete_prompt_profile()
    assert "Protected Profile" in warnings

    window._save_prompt_profile()
    assert "Protected Profile" in warnings

    # Try creating a profile with reserved name 'Default'
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *args, **kwargs: ("default", True))
    window._create_prompt_profile()
    assert "Reserved Name" in warnings

    # Create valid new profile
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *args, **kwargs: ("CustomProfile", True))
    window._create_prompt_profile()

    assert "CustomProfile" in window.config["prompts"]["profiles"]
    assert window.prompt_profile_combo.currentText() == "CustomProfile"
    assert window.prompt_edit_box.isReadOnly() is False
    assert window.save_prompt_btn.isEnabled() is True
    assert window.delete_profile_btn.isEnabled() is True

    # Sync with extractor combo
    assert window.extractor_profile_combo.currentText() == "CustomProfile"

    # Edit & Save
    window.prompt_edit_box.setPlainText("Custom test instructions for highlight extraction.")
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *args, **kwargs: None)
    window._save_prompt_profile()

    saved = config_manager.load_config(cfg_file)
    assert saved["prompts"]["profiles"]["CustomProfile"] == "Custom test instructions for highlight extraction."

    # Delete custom profile -> switches back to Default
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *args, **kwargs: QtWidgets.QMessageBox.Yes)
    window._delete_prompt_profile()

    saved_after_del = config_manager.load_config(cfg_file)
    assert "CustomProfile" not in saved_after_del["prompts"]["profiles"]
    assert window.prompt_profile_combo.currentText() == "Default"
    assert window.prompt_edit_box.isReadOnly() is True

    # Test jump button from Extractor tab to Prompt Editor tab
    window.stacked_widget.setCurrentIndex(0)
    window.edit_preset_btn.click()
    assert window.stacked_widget.currentIndex() == 1

    window.close()


@pytest.mark.usefixtures("qapp")
def test_pyside_app_profile_persistence_on_change(temp_config_env, monkeypatch):
    cfg_file, _ = temp_config_env
    window1 = ClipGenPySideApp()

    # Create profile B
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *args, **kwargs: ("ProfileB", True))
    window1._create_prompt_profile()
    assert window1.config["prompts"]["active_profile"] == "ProfileB"

    # Select Default on extractor combo
    window1.extractor_profile_combo.setCurrentText("Default")
    assert window1.config["prompts"]["active_profile"] == "Default"

    # Select ProfileB again
    window1.extractor_profile_combo.setCurrentText("ProfileB")
    assert window1.config["prompts"]["active_profile"] == "ProfileB"
    window1.close()

    # Verify new instance loads the persisted last selected profile
    window2 = ClipGenPySideApp()
    assert window2.extractor_profile_combo.currentText() == "ProfileB"
    assert window2.prompt_profile_combo.currentText() == "ProfileB"
    window2.close()


@pytest.mark.usefixtures("qapp")
def test_pyside_app_default_profile_auto_recovery(temp_config_env):
    cfg_file, _ = temp_config_env
    # Clear prompts profiles in config
    saved = config_manager.load_config(cfg_file)
    saved["prompts"]["profiles"] = {}
    saved["prompts"]["active_profile"] = ""
    config_manager.save_config(saved, filepath=cfg_file)

    # Launch app
    window = ClipGenPySideApp()
    assert "Default" in window.config["prompts"]["profiles"]
    assert window.prompt_profile_combo.currentText() == "Default"
    assert len(window.prompt_edit_box.toPlainText()) > 50
    window.close()


@pytest.mark.usefixtures("qapp")
def test_pyside_app_gallery_population(temp_config_env):
    _, clips_dir = temp_config_env
    window = ClipGenPySideApp()

    # Create dummy clip files
    dummy_mp4 = os.path.join(clips_dir, "test_vod_clip_1_score9_FunnyMoment.mp4")
    dummy_json = os.path.join(clips_dir, "test_vod_clip_1_score9_FunnyMoment.json")
    with open(dummy_mp4, "w") as f:
        f.write("dummy video")
    with open(dummy_json, "w", encoding="utf-8") as f:
        f.write('{"virality_score": 9, "reasoning": "Hilarious test highlight"}')

    window.populate_gallery()
    assert window.clip_list_widget.count() == 1

    item = window.clip_list_widget.item(0)
    assert "test_vod_clip_1_score9_FunnyMoment.mp4" in item.text()

    # Select item
    window.clip_list_widget.setCurrentItem(item)
    assert window.detail_score.text() == "Virality Score: 9/10"
    assert "Hilarious test highlight" in window.detail_reasoning_box.toPlainText()
    assert window.play_clip_btn.isEnabled()

    window.close()


@pytest.mark.usefixtures("qapp")
def test_pyside_extraction_worker():
    signals = WorkerSignals()
    logs = []
    signals.log_message.connect(lambda msg, _src: logs.append(msg))

    worker = ClipExtractionWorker("nonexistent_video.mp4", "Default", signals)
    worker.run()

    assert any("File not found" in log for log in logs)


def test_pyside_worker_passes_the_cancel_flag_into_process_video(tmp_path, monkeypatch):
    """Regression: Cancel did nothing because process_video never received is_cancelled."""
    video = tmp_path / "clip.mp4"
    video.write_text("x")
    signals = WorkerSignals()
    logs = []
    signals.log_message.connect(lambda msg, _src: logs.append(msg))
    worker = ClipExtractionWorker(str(video), "Default", signals)
    seen = {}

    def fake_process_video(path, prompt_profile="Default", logger=None, is_cancelled=None):
        seen["callable"] = callable(is_cancelled)
        seen["before"] = is_cancelled()
        worker.cancel()  # the user presses Cancel while the pipeline is running
        seen["after"] = is_cancelled()
        return False

    monkeypatch.setattr("src.ui.pyside_app.editor.process_video", fake_process_video)
    worker.run()
    assert seen == {"callable": True, "before": False, "after": True}
    assert any("cancelled by user" in line for line in logs)
    assert not any("processed successfully" in line for line in logs)


def test_pyside_worker_stops_the_queue_after_cancel(tmp_path, monkeypatch):
    first, second = tmp_path / "a.mp4", tmp_path / "b.mp4"
    first.write_text("x")
    second.write_text("x")
    signals = WorkerSignals()
    worker = ClipExtractionWorker(f"{first};{second}", "Default", signals)
    processed = []

    def fake_process_video(path, prompt_profile="Default", logger=None, is_cancelled=None):
        processed.append(os.path.basename(path))
        worker.cancel()
        return True

    monkeypatch.setattr("src.ui.pyside_app.editor.process_video", fake_process_video)
    worker.run()
    assert processed == ["a.mp4"]
