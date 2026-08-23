import json
import os
from unittest.mock import MagicMock


def test_app_initialization(headless_app):
    assert headless_app.title() == "Clip Generator — AI Highlight Workstation"
    assert headless_app.config is not None
    assert hasattr(headless_app, "manual_frame")
    assert hasattr(headless_app, "prompt_frame")
    assert hasattr(headless_app, "settings_frame")
    assert hasattr(headless_app, "gallery_frame")


def test_navigation_frames(headless_app):
    headless_app.show_manual_frame()
    headless_app.show_prompt_frame()
    headless_app.show_settings_frame()
    headless_app.show_gallery_frame()


def test_console_logging_and_tags(headless_app):
    headless_app.clear_console()
    headless_app.log_to_console("❌ Error encountered", source="manual")
    headless_app.log_to_console("✅ Success completed", source="manual")
    headless_app.log_to_console("🧠 AI thinking", source="manual")
    headless_app.log_to_console("✂️ FFmpeg slicing", source="manual")
    headless_app.log_to_console("Regular info", source="manual")
    headless_app.update()

    text_content = headless_app.console_box.get("1.0", "end")
    assert "Error encountered" in text_content
    assert "Success completed" in text_content


def test_prompt_management(headless_app, monkeypatch):
    # Test on_profile_change with the default profile
    headless_app.on_profile_change("Default")
    assert headless_app.config["prompts"]["active_profile"] == "Default"

    # Test save_current_prompt
    headless_app.prompt_textbox.delete("1.0", "end")
    headless_app.prompt_textbox.insert("1.0", "New Custom Prompt Instructions")
    headless_app.save_current_prompt()
    assert headless_app.config["prompts"]["profiles"]["Default"] == "New Custom Prompt Instructions"

    # Test create_new_profile - success
    monkeypatch.setattr("customtkinter.CTkInputDialog.get_input", lambda self: "Speedrun Clips")
    headless_app.create_new_profile()
    assert "Speedrun Clips" in headless_app.config["prompts"]["profiles"]

    # Test create_new_profile - duplicate
    headless_app.create_new_profile()

    # Test create_new_profile - empty
    monkeypatch.setattr("customtkinter.CTkInputDialog.get_input", lambda self: "")
    headless_app.create_new_profile()

    # Test delete_profile - confirmed
    monkeypatch.setattr("tkinter.messagebox.askyesno", lambda *args, **kwargs: True)
    headless_app.profile_dropdown.set("Speedrun Clips")
    headless_app.delete_profile()
    assert "Speedrun Clips" not in headless_app.config["prompts"]["profiles"]

    # Test delete_profile - cannot delete last profile
    headless_app.config["prompts"]["profiles"] = {"Only One": "prompt"}
    headless_app.profile_dropdown.set("Only One")
    monkeypatch.setattr("tkinter.messagebox.showwarning", lambda *args, **kwargs: None)
    headless_app.delete_profile()
    assert "Only One" in headless_app.config["prompts"]["profiles"]


def test_save_settings(headless_app, tmp_path):
    headless_app.openai_entry.delete(0, "end")
    headless_app.openai_entry.insert(0, "sk-test-openai")
    headless_app.deepseek_entry.delete(0, "end")
    headless_app.deepseek_entry.insert(0, "sk-test-ds")
    headless_app.clip_dir_entry.delete(0, "end")
    headless_app.clip_dir_entry.insert(0, str(tmp_path / "clips"))

    headless_app.save_settings()
    assert headless_app.config["openai"]["api_key"] == "sk-test-openai"
    assert headless_app.config["deepseek"]["api_key"] == "sk-test-ds"
    assert headless_app.config["settings"]["clips_dir"] == str(tmp_path / "clips")


def test_gallery_population_and_filtering(headless_app, tmp_path):
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    headless_app.config["settings"]["clips_dir"] = str(clips_dir)

    # Create dummy clips, metadata and thumbnails
    (clips_dir / "clip1_score9.0.mp4").write_text("dummy")
    (clips_dir / "clip1_score9.0.json").write_text(json.dumps({"virality_score": 9.0, "reasoning": "Insane shot"}))
    (clips_dir / "clip2_score6.0_vertical.mp4").write_text("dummy")
    (clips_dir / "clip2_score6.0.json").write_text(json.dumps({"virality_score": 6.0, "reasoning": "Funny moment"}))

    headless_app.sort_menu.set("Date (Newest)")
    headless_app.populate_gallery()
    headless_app.update()

    assert len(headless_app.marked_for_deletion) == 2

    # Toggle select all
    headless_app.select_all_var.set(True)
    headless_app.toggle_select_all()
    for var in headless_app.marked_for_deletion.values():
        assert var.get() is True

    # Test load_clip_details
    headless_app.load_clip_details("clip1_score9.0.mp4", str(clips_dir))
    assert "clip1_score9.0.mp4" in headless_app.detail_title.cget("text")
    assert "9.0/10" in headless_app.detail_score.cget("text")


def test_confirm_delete_marked(headless_app, tmp_path, monkeypatch):
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    headless_app.config["settings"]["clips_dir"] = str(clips_dir)

    test_clip = clips_dir / "clip_del.mp4"
    test_clip.write_text("dummy")
    headless_app.populate_gallery()

    # None selected test
    headless_app.confirm_delete_marked()

    # Select and confirm deletion
    headless_app.marked_for_deletion["clip_del.mp4"].set(True)
    monkeypatch.setattr("tkinter.messagebox.askyesno", lambda *args, **kwargs: True)
    headless_app.confirm_delete_marked()
    assert not test_clip.exists()


def test_api_key_testing_methods(headless_app, monkeypatch):
    class SyncThread:
        def __init__(self, target=None, args=(), kwargs=None, daemon=None):
            self.target = target
            self.args = args or ()
            self.kwargs = kwargs or {}
        def start(self):
            if self.target: self.target(*self.args, **self.kwargs)

    monkeypatch.setattr("threading.Thread", SyncThread)

    # Test OpenAI
    mock_client = MagicMock()
    mock_client.models.list.return_value = []
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: mock_client)
    headless_app.openai_entry.insert(0, "sk-test")
    headless_app.test_openai_key()

    # Test DeepSeek
    headless_app.deepseek_entry.insert(0, "sk-ds")
    headless_app.test_deepseek_key()

    # Test Anthropic
    mock_ant = MagicMock()
    monkeypatch.setattr("anthropic.Anthropic", lambda **kwargs: mock_ant)
    headless_app.anthropic_entry.insert(0, "sk-ant")
    headless_app.test_anthropic_key()

    # Test Grok
    headless_app.grok_entry.insert(0, "xai-test")
    headless_app.test_grok_key()

    # Test Google
    mock_genai_client = MagicMock()
    mock_genai_client.models.list.return_value = []
    monkeypatch.setattr("google.genai.Client", lambda **kwargs: mock_genai_client)
    headless_app.google_entry.insert(0, "AIzaSy")
    headless_app.test_google_key()


def test_discord_webhook_test_and_alert(headless_app, monkeypatch):
    class SyncThread:
        def __init__(self, target=None, args=(), kwargs=None, daemon=None):
            self.target = target
            self.args = args or ()
            self.kwargs = kwargs or {}
        def start(self):
            if self.target: self.target(*self.args, **self.kwargs)

    monkeypatch.setattr("threading.Thread", SyncThread)

    # Test empty URL
    headless_app.discord_entry.delete(0, "end")
    headless_app.test_discord_webhook()

    # Test valid URL mock
    mock_urlopen = MagicMock()
    mock_resp = MagicMock(status=204)
    mock_urlopen.return_value.__enter__.return_value = mock_resp
    monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)

    headless_app.discord_entry.insert(0, "https://discord.com/api/webhooks/123/abc")
    headless_app.test_discord_webhook()

    headless_app.config["integrations"]["discord_webhook"] = "https://discord.com/api/webhooks/123/abc"
    headless_app.send_discord_alert("Finished Video")


def test_batch_process_lifecycle(headless_app, monkeypatch, tmp_path):
    class SyncThread:
        def __init__(self, target=None, args=(), kwargs=None, daemon=None):
            self.target = target
            self.args = args or ()
            self.kwargs = kwargs or {}
        def start(self):
            if self.target: self.target(*self.args, **self.kwargs)

    monkeypatch.setattr("threading.Thread", SyncThread)

    video = tmp_path / "test.mp4"
    video.write_text("dummy")

    headless_app.file_input.delete(0, "end")
    headless_app.file_input.insert(0, str(video))

    monkeypatch.setattr("src.core.editor.process_video", lambda *args, **kwargs: None)
    headless_app.start_manual_process()

    headless_app.cancel_manual_process()
    assert headless_app.cancel_requested is True


def test_tray_and_window_lifecycle(headless_app, monkeypatch):
    mock_icon = MagicMock()
    monkeypatch.setattr("pystray.Icon", lambda *args, **kwargs: mock_icon)

    headless_app.minimize_to_tray()
    headless_app.show_window(mock_icon, None)


def test_open_local_folder_and_readme(headless_app, monkeypatch):
    mock_startfile = MagicMock()
    monkeypatch.setattr("os.startfile", mock_startfile, raising=False)

    headless_app.config["settings"]["clips_dir"] = os.getcwd()
    headless_app.open_local_folder("clips_dir")
    headless_app.open_readme()
    headless_app.open_logs()
