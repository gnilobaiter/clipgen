import copy
from unittest.mock import MagicMock, patch
import pytest

from src.core.config import get_default_config


class MockWidget:
    """Headless Tkinter / CustomTkinter widget mockup that avoids creating OS windows."""
    def __init__(self, master=None, **kwargs):
        self.master = master
        self.kwargs = kwargs
        self.children = []
        self._text = kwargs.get("text", "")
        self._value = kwargs.get("values", [""])[0] if "values" in kwargs else ""
        self._var = kwargs.get("variable", None)
        self._mapped = True
        self._grid_info = {}
        if master and hasattr(master, "children") and isinstance(master.children, list):
            master.children.append(self)

    def grid(self, **kwargs):
        self._grid_info = kwargs
        self._mapped = True
        return self

    def pack(self, **kwargs):
        self._mapped = True
        return self

    def grid_forget(self):
        self._mapped = False

    def pack_forget(self):
        self._mapped = False

    def grid_propagate(self, *args):
        pass

    def pack_propagate(self, *args):
        pass

    def grid_rowconfigure(self, *args, **kwargs):
        pass

    def grid_columnconfigure(self, *args, **kwargs):
        pass

    def winfo_exists(self):
        return True

    def configure(self, **kwargs):
        self.kwargs.update(kwargs)
        if "text" in kwargs:
            self._text = kwargs["text"]

    def cget(self, key):
        if key == "text":
            return self._text
        return self.kwargs.get(key)

    def get(self, *args):
        if self._var:
            return self._var.get()
        return self._text or self._value

    def insert(self, idx, text, *tags):
        self._text = (self._text or "") + str(text)

    def delete(self, start, end=None):
        self._text = ""

    def set(self, val):
        self._value = val
        if self._var:
            self._var.set(val)

    def select(self):
        if self._var:
            self._var.set(True)

    def deselect(self):
        if self._var:
            self._var.set(False)

    def winfo_ismapped(self):
        return self._mapped

    def winfo_children(self):
        return getattr(self, "children", [])

    def destroy(self):
        pass

    def see(self, *args):
        pass

    def tag_config(self, *args, **kwargs):
        pass

    def bind(self, *args, **kwargs):
        pass

    def unbind(self, *args, **kwargs):
        pass

    def focus_set(self):
        pass

    def start(self, *args):
        pass

    def stop(self, *args):
        pass


class MockCTkRoot(MockWidget):
    """Headless CustomTkinter root window mockup."""
    def __init__(self, *args, **kwargs):
        MockWidget.__init__(self, *args, **kwargs)
        self._title = ""
        self._geometry = ""

    def title(self, val=None):
        if val is not None:
            self._title = val
        return self._title

    def geometry(self, val=None):
        if val is not None:
            self._geometry = val
        return self._geometry

    def minsize(self, *args):
        pass

    def protocol(self, *args, **kwargs):
        pass

    def after(self, _delay, callback, *args):
        if callback:
            callback(*args)

    def update(self):
        pass

    def update_idletasks(self):
        pass

    def withdraw(self):
        pass

    def deiconify(self):
        pass


class MockVar:
    """Mock for Tkinter StringVar/BooleanVar/IntVar."""
    def __init__(self, value=None, *args, **kwargs):
        self._value = value

    def get(self):
        return self._value

    def set(self, val):
        self._value = val


@pytest.fixture(autouse=True)
def isolate_sound_classifier(tmp_path, monkeypatch):
    """Tests must never download the YAMNet model, touch the real app-data folder or start a GPU worker.
    Tests that exercise the classifier override these attributes themselves."""
    from src.core import sound_classifier

    def no_network(_url, **_kwargs):
        raise OSError("network access is disabled in tests")

    monkeypatch.setattr(sound_classifier, "get_app_data_path", lambda: str(tmp_path / "sound_appdata"))
    monkeypatch.setattr(sound_classifier.urllib.request, "urlopen", no_network)
    monkeypatch.setattr(sound_classifier, "_cpu_session", None)


@pytest.fixture(scope="session")
def sample_config():
    """Returns a clean copy of default configuration."""
    return copy.deepcopy(get_default_config())


@pytest.fixture
def mock_app_data(tmp_path, monkeypatch):
    """Isolates all appdata paths, transcripts, and config to a temporary sandbox."""
    app_dir = tmp_path / "appdata" / "jBahrsClipGenerator"
    app_dir.mkdir(parents=True, exist_ok=True)
    (app_dir / "transcripts").mkdir(exist_ok=True)
    (app_dir / "logs").mkdir(exist_ok=True)

    monkeypatch.setattr("src.utils.paths.get_app_data_path", lambda: str(app_dir))
    monkeypatch.setattr("src.ui.app.get_app_data_path", lambda: str(app_dir))
    monkeypatch.setattr("src.core.editor.get_app_data_path", lambda: str(app_dir))
    return app_dir


class MockInputDialog:
    def __init__(self, text="", title="", *args, **kwargs):
        self.text = text
        self.title = title

    def get_input(self):
        return ""


@pytest.fixture
def headless_app(mock_app_data, monkeypatch):
    """Creates a completely headless ClipGenApp in memory without opening any OS windows."""
    import customtkinter as ctk
    import tkinter as tk

    assert mock_app_data.exists()

    ctk_mocks = {
        "CTkFrame": MockWidget,
        "CTkScrollableFrame": MockWidget,
        "CTkButton": MockWidget,
        "CTkLabel": MockWidget,
        "CTkEntry": MockWidget,
        "CTkTextbox": MockWidget,
        "CTkOptionMenu": MockWidget,
        "CTkComboBox": MockWidget,
        "CTkCheckBox": MockWidget,
        "CTkRadioButton": MockWidget,
        "CTkProgressBar": MockWidget,
        "CTkSlider": MockWidget,
        "CTkSwitch": MockWidget,
        "CTkSegmentedButton": MockWidget,
        "CTkTabview": MockWidget,
        "CTkInputDialog": MockInputDialog,
        "CTkFont": MagicMock,
        "CTkImage": MagicMock,
        "StringVar": MockVar,
        "BooleanVar": MockVar,
        "IntVar": MockVar,
    }
    tk_mocks = {
        "StringVar": MockVar,
        "BooleanVar": MockVar,
        "IntVar": MockVar,
    }

    monkeypatch.setattr("src.ui.app.ClipGenApp.refresh_model_list", lambda self: None)

    with patch.multiple(ctk, **ctk_mocks), patch.multiple(tk, **tk_mocks):
        from src.ui.app import ClipGenApp, THEME, config_manager

        class HeadlessClipGenApp(MockCTkRoot, ClipGenApp):
            def __init__(self):
                MockCTkRoot.__init__(self)
                self.title("Clip Generator — AI Highlight Workstation")
                self.geometry("1180x880")
                self.minsize(1000, 680)
                self.configure(fg_color=THEME["bg_window"])
                self.protocol("WM_DELETE_WINDOW", self.minimize_to_tray)
                self.tray_icon = None
                self.config = config_manager.load_config()
                self.metadata_cache = {}
                self.cancel_requested = False
                self._init_logging()
                self.grid_rowconfigure(0, weight=1)
                self.grid_columnconfigure(1, weight=1)
                self._setup_sidebar()
                self._setup_manual_frame()
                self._setup_prompt_frame()
                self._setup_settings_frame()
                self._setup_gallery_frame()
                self.load_prompt_data()
                self.show_manual_frame()

        app = HeadlessClipGenApp()
        yield app
        app.destroy()
