import os
import shutil
import sys

def get_project_root() -> str:
    """Returns the absolute path to the project root directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, "..", ".."))

def get_bin_dir() -> str:
    """Returns the absolute path to the local bin directory inside the project."""
    return os.path.join(get_project_root(), "bin")

def get_data_dir() -> str:
    """Returns <project root>/data: the downloaded AI models (data/models), the transcript cache and the optional
    HuggingFace token file (git-ignored). With config.json and logs/ in the project root, nothing is kept in the user profile."""
    data_dir = os.path.join(get_project_root(), "data")
    os.makedirs(data_dir, exist_ok=True)
    guard = os.path.join(data_dir, ".gitignore")
    if not os.path.exists(guard):  # several GB of models: keep them out of git even if the root .gitignore changes
        try:
            with open(guard, "w", encoding="utf-8") as f:
                f.write("*\n")
        except OSError:
            pass
    return data_dir

def get_models_dir() -> str:
    """Returns <project root>/data/models: the downloaded AI models (Whisper, HuggingFace / pyannote, torch, nltk, YAMNet)."""
    models_dir = os.path.join(get_data_dir(), "models")
    os.makedirs(models_dir, exist_ok=True)
    return models_dir

def get_app_data_path() -> str:
    """Returns the application data directory (<project root>/data): transcript cache and the token file."""
    return get_data_dir()

def get_logs_dir() -> str:
    """Returns <project root>/logs (git-ignored): execution and crash logs."""
    logs_dir = os.path.join(get_project_root(), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    return logs_dir

def get_legacy_config_path() -> str:
    """Config location of older releases (%APPDATA%/jBahrsClipGenerator/config.json). Read-only: used for a one-time copy."""
    base = os.getenv("APPDATA") or os.path.expanduser("~")
    return os.path.join(str(base), "jBahrsClipGenerator", "config.json")

def get_config_dir() -> str:
    """Folder of config.json: the project root (git-ignored)."""
    return get_project_root()

def get_config_path() -> str:
    """Returns the absolute path to <project root>/config.json (API keys, HuggingFace token, settings, prompts).
    A config left in %APPDATA% by an older release is copied here once (that old file is never deleted)."""
    path = os.path.join(get_config_dir(), "config.json")
    if not os.path.exists(path):
        legacy = get_legacy_config_path()
        try:
            if os.path.isfile(legacy):
                shutil.copy2(legacy, path)
        except OSError:
            pass
    return path

def setup_model_env() -> None:
    """Points HuggingFace / torch / nltk caches into data/models (must run before whisperx / transformers are imported).
    Overrides a global HF_HOME / TORCH_HOME / NLTK_DATA on purpose: all downloaded models must live in data/models."""
    models_dir = get_models_dir()
    os.environ["HF_HOME"] = os.path.join(models_dir, "huggingface")
    os.environ["TORCH_HOME"] = os.path.join(models_dir, "torch")
    os.environ["NLTK_DATA"] = os.path.join(models_dir, "nltk")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

def inject_bin_to_path() -> None:
    """Prepends the project's local bin/ folder to the system PATH environment variable."""
    bin_dir = get_bin_dir()
    if os.path.exists(bin_dir):
        current_path = os.environ.get("PATH", "")
        if bin_dir not in current_path:
            os.environ["PATH"] = bin_dir + os.pathsep + current_path

def get_ffmpeg_path() -> str:
    """Resolves ffmpeg binary path prioritizing local bin/ folder then system PATH."""
    inject_bin_to_path()
    bin_ffmpeg = os.path.join(get_bin_dir(), "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg")
    if os.path.exists(bin_ffmpeg):
        return bin_ffmpeg
    which_path = shutil.which("ffmpeg")
    return which_path if which_path else "ffmpeg"

def get_ffprobe_path() -> str:
    """Resolves ffprobe binary path prioritizing local bin/ folder then system PATH."""
    inject_bin_to_path()
    bin_ffprobe = os.path.join(get_bin_dir(), "ffprobe.exe" if sys.platform == "win32" else "ffprobe")
    if os.path.exists(bin_ffprobe):
        return bin_ffprobe
    which_path = shutil.which("ffprobe")
    return which_path if which_path else "ffprobe"
