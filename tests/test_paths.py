import os
import shutil
import sys

from src.utils.paths import (
    get_project_root,
    get_bin_dir,
    get_app_data_path,
    get_config_path,
    inject_bin_to_path,
    get_ffmpeg_path,
    get_ffprobe_path,
)

def test_get_project_root():
    root = get_project_root()
    assert os.path.isabs(root)
    assert os.path.exists(os.path.join(root, "src"))

def test_get_bin_dir():
    bin_dir = get_bin_dir()
    assert os.path.isabs(bin_dir)
    assert bin_dir.endswith("bin")

def test_get_app_data_path_with_env(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "getenv", lambda key, default=None: str(tmp_path) if key == "APPDATA" else default)
    app_data = get_app_data_path()
    assert os.path.exists(app_data)
    assert app_data == os.path.join(str(tmp_path), "jBahrsClipGenerator")

def test_get_app_data_path_no_env(monkeypatch, tmp_path):
    monkeypatch.setattr(os, "getenv", lambda key, default=None: None if key == "APPDATA" else default)
    monkeypatch.setattr(os.path, "expanduser", lambda path: str(tmp_path))
    app_data = get_app_data_path()
    assert os.path.exists(app_data)

def test_get_config_path(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "getenv", lambda key, default=None: str(tmp_path) if key == "APPDATA" else default)
    cfg_path = get_config_path()
    assert cfg_path == os.path.join(str(tmp_path), "jBahrsClipGenerator", "config.json")

def test_inject_bin_to_path(monkeypatch, tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    monkeypatch.setattr("src.utils.paths.get_bin_dir", lambda: str(fake_bin))
    monkeypatch.setenv("PATH", "some_existing_path")

    inject_bin_to_path()
    assert str(fake_bin) in os.environ["PATH"]

    # Call again, should not duplicate
    old_path = os.environ["PATH"]
    inject_bin_to_path()
    assert os.environ["PATH"] == old_path

def test_inject_bin_to_path_nonexistent(monkeypatch, tmp_path):
    fake_bin = tmp_path / "nonexistent_bin"
    monkeypatch.setattr("src.utils.paths.get_bin_dir", lambda: str(fake_bin))
    monkeypatch.setenv("PATH", "orig_path")
    inject_bin_to_path()
    assert os.environ["PATH"] == "orig_path"

def test_get_ffmpeg_path_local(tmp_path, monkeypatch):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    ffmpeg_name = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
    fake_ffmpeg = fake_bin / ffmpeg_name
    fake_ffmpeg.write_text("binary")

    monkeypatch.setattr("src.utils.paths.get_bin_dir", lambda: str(fake_bin))
    path = get_ffmpeg_path()
    assert path == str(fake_ffmpeg)

def test_get_ffmpeg_path_system_which(tmp_path, monkeypatch):
    fake_bin = tmp_path / "empty_bin"
    fake_bin.mkdir()
    monkeypatch.setattr("src.utils.paths.get_bin_dir", lambda: str(fake_bin))
    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/ffmpeg" if cmd == "ffmpeg" else None)

    path = get_ffmpeg_path()
    assert path == "/usr/bin/ffmpeg"

def test_get_ffmpeg_path_fallback(tmp_path, monkeypatch):
    fake_bin = tmp_path / "empty_bin"
    fake_bin.mkdir()
    monkeypatch.setattr("src.utils.paths.get_bin_dir", lambda: str(fake_bin))
    monkeypatch.setattr(shutil, "which", lambda cmd: None)

    path = get_ffmpeg_path()
    assert path == "ffmpeg"

def test_get_ffprobe_path_local(tmp_path, monkeypatch):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    ffprobe_name = "ffprobe.exe" if sys.platform == "win32" else "ffprobe"
    fake_ffprobe = fake_bin / ffprobe_name
    fake_ffprobe.write_text("binary")

    monkeypatch.setattr("src.utils.paths.get_bin_dir", lambda: str(fake_bin))
    path = get_ffprobe_path()
    assert path == str(fake_ffprobe)

def test_get_ffprobe_path_system_which(tmp_path, monkeypatch):
    fake_bin = tmp_path / "empty_bin"
    fake_bin.mkdir()
    monkeypatch.setattr("src.utils.paths.get_bin_dir", lambda: str(fake_bin))
    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/ffprobe" if cmd == "ffprobe" else None)

    path = get_ffprobe_path()
    assert path == "/usr/bin/ffprobe"

def test_get_ffprobe_path_fallback(tmp_path, monkeypatch):
    fake_bin = tmp_path / "empty_bin"
    fake_bin.mkdir()
    monkeypatch.setattr("src.utils.paths.get_bin_dir", lambda: str(fake_bin))
    monkeypatch.setattr(shutil, "which", lambda cmd: None)

    path = get_ffprobe_path()
    assert path == "ffprobe"
