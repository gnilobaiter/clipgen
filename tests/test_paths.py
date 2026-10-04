import os
import shutil
import sys

from src.utils.paths import (
    get_project_root,
    get_bin_dir,
    get_app_data_path,
    get_config_dir,
    get_data_dir,
    get_logs_dir,
    get_models_dir,
    setup_model_env,
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

def test_data_dir_is_inside_the_project_not_the_user_profile():
    data_dir = get_data_dir()  # the real function (conftest only replaces the module attribute used by the app)
    assert data_dir == os.path.join(get_project_root(), "data")
    assert os.path.isabs(data_dir) and "AppData" not in data_dir.split(os.sep)


def test_app_data_is_the_data_dir_and_config_is_in_the_project_root():
    from src.utils import paths  # conftest points the dirs at temp folders for every test
    assert get_app_data_path() == paths.get_data_dir()  # transcript cache, logs and the token file live in data/
    assert paths.get_models_dir() == os.path.join(paths.get_data_dir(), "models")  # downloaded AI models in data/models/
    assert get_config_path() == os.path.join(paths.get_config_dir(), "config.json")  # the config is next to the Makefile
    assert paths.get_config_dir() != paths.get_data_dir()


def test_logs_live_in_the_project_root_not_in_data(tmp_path, monkeypatch):
    monkeypatch.setattr("src.utils.paths.get_project_root", lambda: str(tmp_path))
    logs = get_logs_dir()  # the real function, working on a temp "project"
    assert logs == str(tmp_path / "logs") and (tmp_path / "logs").is_dir()
    assert get_data_dir() != logs and not (tmp_path / "data" / "logs").exists()


def test_real_config_dir_is_the_project_root():
    assert get_config_dir() == get_project_root()  # the real function: conftest only swaps the module attribute


def _config_setup(tmp_path, monkeypatch):
    root = tmp_path / "root"
    legacy = tmp_path / "appdata" / "jBahrsClipGenerator" / "config.json"
    for d in (root, legacy.parent):
        d.mkdir(parents=True)
    monkeypatch.setattr("src.utils.paths.get_config_dir", lambda: str(root))
    monkeypatch.setattr("src.utils.paths.get_legacy_config_path", lambda: str(legacy))
    return root, legacy


def test_config_is_copied_once_from_the_old_appdata_location(tmp_path, monkeypatch):
    root, legacy = _config_setup(tmp_path, monkeypatch)
    legacy.write_text('{"openai": {"api_key": "k"}}', encoding="utf-8")
    path = get_config_path()
    assert path == str(root / "config.json") and open(path, encoding="utf-8").read() == legacy.read_text(encoding="utf-8")
    assert legacy.exists()  # the old file is never deleted

    (root / "config.json").write_text("{}", encoding="utf-8")
    get_config_path()
    assert (root / "config.json").read_text(encoding="utf-8") == "{}"  # an existing config is never overwritten


def test_no_config_anywhere_just_returns_the_root_path(tmp_path, monkeypatch):
    root, _legacy = _config_setup(tmp_path, monkeypatch)
    assert get_config_path() == str(root / "config.json") and not (root / "config.json").exists()


def test_data_dir_keeps_itself_out_of_git_and_creates_models(tmp_path, monkeypatch):
    monkeypatch.setattr("src.utils.paths.get_project_root", lambda: str(tmp_path))
    data = get_data_dir()  # the real functions, working on a temp "project"
    assert data == str(tmp_path / "data")
    assert (tmp_path / "data" / ".gitignore").read_text().strip() == "*"  # several GB of models must never be committed
    monkeypatch.setattr("src.utils.paths.get_data_dir", lambda: data)
    assert get_models_dir() == str(tmp_path / "data" / "models") and (tmp_path / "data" / "models").is_dir()


def test_setup_model_env_points_caches_into_data_models(tmp_path, monkeypatch):
    monkeypatch.setattr("src.utils.paths.get_models_dir", lambda: str(tmp_path))
    for key in ("HF_HOME", "TORCH_HOME", "NLTK_DATA"):
        monkeypatch.delenv(key, raising=False)
    setup_model_env()
    assert os.environ["HF_HOME"] == str(tmp_path / "huggingface")
    assert os.environ["TORCH_HOME"] == str(tmp_path / "torch")
    assert os.environ["NLTK_DATA"] == str(tmp_path / "nltk")
    monkeypatch.setenv("HF_HOME", "custom")
    setup_model_env()
    assert os.environ["HF_HOME"] == str(tmp_path / "huggingface")  # a global setting does not move the models out of data/models


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
