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

def get_app_data_path() -> str:
    """Returns the application data directory (%APPDATA%/jBahrsClipGenerator on Windows)."""
    app_data = os.getenv("APPDATA")
    if not app_data:
        app_data = os.path.expanduser("~")
    config_dir = os.path.join(str(app_data), "jBahrsClipGenerator")
    if not os.path.exists(config_dir):
        os.makedirs(config_dir, exist_ok=True)
    return config_dir

def get_config_path() -> str:
    """Returns the absolute path to the config.json file in APPDATA."""
    return os.path.join(get_app_data_path(), "config.json")

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
