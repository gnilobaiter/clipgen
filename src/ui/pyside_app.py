import datetime
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import subprocess
import sys
import threading
from typing import List, Optional
import urllib.parse
import urllib.request

from PySide6 import QtCore, QtGui, QtWidgets

from src.core import config as config_manager
from src.core import editor
from src.services import model_fetcher
from src.utils.hardware import get_hardware_status
from src.utils.paths import get_app_data_path, inject_bin_to_path

inject_bin_to_path()

# ==============================================================================
# MODERN OBSIDIAN DARK THEME (QSS) — 100% CLEAN & TRANSPARENT LABELS
# ==============================================================================
QSS_STYLESHEET = """
QMainWindow {
    background-color: #080c14;
}

#CentralWidget {
    background-color: #080c14;
}

QWidget {
    color: #e2e8f0;
    font-family: "Segoe UI Variable Text", "Segoe UI", "Inter", -apple-system, sans-serif;
    font-size: 13px;
}

/* ALL Labels, Radios, Checkboxes MUST be transparent so no dark/square boxes appear behind text */
QLabel {
    background: transparent;
    background-color: transparent;
}

QCheckBox, QRadioButton {
    background: transparent;
    background-color: transparent;
    color: #cbd5e1;
    spacing: 8px;
    font-size: 13px;
}

QFrame {
    background: transparent;
    background-color: transparent;
    border: none;
}

/* Cards & Surfaces */
QFrame.card, QFrame[class="card"] {
    background-color: #0f1726;
    border: 1px solid #1c273c;
    border-radius: 10px;
}

QFrame.provider-grid, QFrame[class="provider-grid"] {
    background-color: #0b0f19;
    border: 1px solid #1c273c;
    border-radius: 8px;
}

QFrame.hw-pill, QFrame[class="hw-pill"] {
    background-color: #090d16;
    border: 1px solid #182234;
    border-radius: 6px;
}

QScrollArea, QScrollArea > QWidget > QWidget {
    background: transparent;
    background-color: transparent;
}

/* Sidebar styling */
#SidebarFrame {
    background-color: #0c111c;
    border-right: 1px solid #182234;
}

#SidebarLogo {
    color: #ffffff;
    font-size: 17px;
    font-weight: 800;
    letter-spacing: 0.6px;
    background: transparent;
}

#SidebarBadge {
    background-color: rgba(56, 189, 248, 0.12);
    color: #38bdf8;
    font-size: 10px;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 4px;
    border: 1px solid rgba(56, 189, 248, 0.25);
    letter-spacing: 0.5px;
}

QPushButton.nav-btn, QPushButton[class="nav-btn"] {
    background-color: transparent;
    color: #94a3b8;
    border: none;
    border-left: 3px solid transparent;
    border-radius: 6px;
    padding: 10px 14px;
    text-align: left;
    font-size: 13px;
    font-weight: 500;
}

QPushButton.nav-btn:hover, QPushButton[class="nav-btn"]:hover {
    background-color: rgba(255, 255, 255, 0.04);
    color: #f8fafc;
}

QPushButton.nav-btn:checked, QPushButton[class="nav-btn"]:checked {
    background-color: rgba(59, 130, 246, 0.12);
    border-left: 3px solid #3b82f6;
    color: #60a5fa;
    font-weight: 600;
}

/* Provider Tab Selector Pills */
QPushButton.provider-tab, QPushButton[class="provider-tab"] {
    background-color: #0b0f19;
    color: #94a3b8;
    border: 1px solid #1e293b;
    border-radius: 6px;
    padding: 8px 14px;
    font-size: 12px;
    font-weight: 600;
}

QPushButton.provider-tab:hover, QPushButton[class="provider-tab"]:hover {
    background-color: #141c2c;
    color: #f8fafc;
    border-color: #334155;
}

QPushButton.provider-tab:checked, QPushButton[class="provider-tab"]:checked {
    background-color: #2563eb;
    color: #ffffff;
    border-color: #3b82f6;
}

/* Headings */
QLabel.page-title, QLabel[class="page-title"] {
    color: #ffffff;
    font-size: 20px;
    font-weight: 700;
    letter-spacing: -0.2px;
    background: transparent;
}

QLabel.page-subtitle, QLabel[class="page-subtitle"] {
    color: #64748b;
    font-size: 12px;
    background: transparent;
}

QLabel.card-title, QLabel[class="card-title"] {
    color: #f1f5f9;
    font-size: 14px;
    font-weight: 700;
    background: transparent;
}

QLabel.field-label, QLabel[class="field-label"] {
    color: #94a3b8;
    font-weight: 600;
    font-size: 12px;
    background: transparent;
}

/* Inputs & Form Controls */
QLineEdit, QPlainTextEdit, QTextEdit {
    background-color: #080c14;
    border: 1px solid #1e293b;
    border-radius: 6px;
    color: #f8fafc;
    padding: 8px 12px;
    font-size: 13px;
    selection-background-color: #2563eb;
}

QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus {
    border: 1px solid #3b82f6;
    background-color: #0a0f1a;
}

QLineEdit::placeholder {
    color: #475569;
}

/* Dropdowns / ComboBox */
QComboBox {
    background-color: #080c14;
    border: 1px solid #1e293b;
    border-radius: 6px;
    color: #f1f5f9;
    padding: 7px 12px;
    font-size: 13px;
    font-weight: 500;
}

QComboBox:hover {
    border-color: #3b82f6;
}

QComboBox::drop-down {
    border: none;
    width: 24px;
}

QComboBox QAbstractItemView {
    background-color: #0f1726;
    border: 1px solid #1e293b;
    color: #f1f5f9;
    selection-background-color: #2563eb;
    selection-color: #ffffff;
    padding: 4px;
    border-radius: 6px;
}

/* Push Buttons */
QPushButton {
    font-family: "Segoe UI Variable Text", "Segoe UI", "Inter", -apple-system, sans-serif;
    font-size: 13px;
    font-weight: 500;
}

QPushButton.primary-btn, QPushButton[class="primary-btn"] {
    background-color: #2563eb;
    color: #ffffff;
    border: 1px solid #3b82f6;
    border-radius: 6px;
    padding: 9px 18px;
    font-size: 13px;
    font-weight: 600;
    text-align: center;
}

QPushButton.primary-btn:hover, QPushButton[class="primary-btn"]:hover {
    background-color: #1d4ed8;
    border-color: #60a5fa;
}

QPushButton.primary-btn:disabled, QPushButton[class="primary-btn"]:disabled {
    background-color: #111827;
    border-color: #1f293d;
    color: #475569;
}

QPushButton.secondary-btn, QPushButton[class="secondary-btn"] {
    background-color: #131b2a;
    color: #cbd5e1;
    border: 1px solid #202c3f;
    border-radius: 6px;
    padding: 7px 14px;
    font-size: 12px;
    font-weight: 500;
}

QPushButton.secondary-btn:hover, QPushButton[class="secondary-btn"]:hover {
    background-color: #1a253a;
    border-color: #3b82f6;
    color: #ffffff;
}

QPushButton.success-btn, QPushButton[class="success-btn"] {
    background-color: #059669;
    color: #ffffff;
    border: 1px solid #10b981;
    border-radius: 6px;
    padding: 7px 14px;
    font-weight: 600;
}

QPushButton.success-btn:hover, QPushButton[class="success-btn"]:hover {
    background-color: #047857;
}

QPushButton.danger-btn, QPushButton[class="danger-btn"] {
    background-color: #be123c;
    color: #ffffff;
    border: 1px solid #e11d48;
    border-radius: 6px;
    padding: 7px 14px;
    font-weight: 600;
}

QPushButton.danger-btn:hover, QPushButton[class="danger-btn"]:hover {
    background-color: #9f1239;
}

/* Checkboxes & Radios */
QCheckBox::indicator, QRadioButton::indicator {
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 1px solid #2d3748;
    background-color: #080c14;
}

QRadioButton::indicator {
    border-radius: 8px;
}

QCheckBox::indicator:hover, QRadioButton::indicator:hover {
    border-color: #3b82f6;
}

QCheckBox::indicator:checked, QRadioButton::indicator:checked {
    background-color: #2563eb;
    border-color: #3b82f6;
}

/* Progress Bar */
QProgressBar {
    background-color: #0b0f19;
    border: 1px solid #1e293b;
    border-radius: 3px;
    height: 5px;
    text-align: center;
}

QProgressBar::chunk {
    background-color: #3b82f6;
    border-radius: 2px;
}

/* Scrollbars */
QScrollBar:vertical {
    background-color: #080c14;
    width: 8px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background-color: #1e293b;
    min-height: 24px;
    border-radius: 4px;
}

QScrollBar::handle:vertical:hover {
    background-color: #334155;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

/* Splitter */
QSplitter::handle {
    background-color: #182234;
    width: 1px;
}

/* List Widget (Gallery) */
QListWidget {
    background-color: #090d16;
    border: 1px solid #1c273c;
    border-radius: 8px;
    color: #f1f5f9;
    padding: 6px;
}

QListWidget::item {
    background-color: #0f1726;
    border: 1px solid #1c273c;
    border-radius: 6px;
    padding: 10px 12px;
    margin-bottom: 6px;
}

QListWidget::item:hover {
    background-color: #162033;
    border-color: #2b3a55;
}

QListWidget::item:selected {
    background-color: #1e2d47;
    border: 1px solid #3b82f6;
    color: #ffffff;
}
"""


class WorkerSignals(QtCore.QObject):
    log_message = QtCore.Signal(str, str)
    status_changed = QtCore.Signal(str, str)
    finished = QtCore.Signal(bool)
    model_list_updated = QtCore.Signal(str, list)


class ClipExtractionWorker(QtCore.QThread):
    def __init__(self, input_val: str, profile: str, signals: WorkerSignals, parent=None):
        super().__init__(parent)
        self.input_val = input_val
        self.profile = profile
        self.signals = signals
        self.is_cancelled = False

    def cancel(self):
        self.is_cancelled = True

    def run(self):
        self.is_cancelled = False
        queue = [item.strip() for item in self.input_val.split(";") if item.strip()]

        if not queue:
            self.signals.log_message.emit("❌ No valid files selected in queue.", "manual")
            self.signals.finished.emit(False)
            return

        total_files = len(queue)
        self.signals.log_message.emit(f"🎬 Starting batch highlight extraction ({total_files} file(s))...", "manual")

        overall_success = True
        for idx, file_path in enumerate(queue, 1):
            if self.is_cancelled:
                self.signals.log_message.emit("🛑 Batch processing aborted by user.", "manual")
                overall_success = False
                break

            self.signals.log_message.emit("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", "manual")
            self.signals.log_message.emit(f"▶️ [{idx}/{total_files}] Processing: {os.path.basename(file_path)}", "manual")

            if not os.path.exists(file_path):
                self.signals.log_message.emit(f"❌ File not found: {file_path}", "manual")
                overall_success = False
                continue

            def thread_logger(msg: str):
                self.signals.log_message.emit(msg, "manual")

            try:
                success = editor.process_video(file_path, prompt_profile=self.profile, logger=thread_logger)
                if not success:
                    overall_success = False
            except Exception as e:
                self.signals.log_message.emit(f"❌ Unhandled Exception: {e}", "manual")
                overall_success = False

        if not self.is_cancelled:
            if overall_success:
                self.signals.log_message.emit("✨ All queued videos processed successfully!", "manual")
            else:
                self.signals.log_message.emit("⚠️ Batch completed with some warnings/errors.", "manual")

        self.signals.finished.emit(overall_success)


class ClipGenPySideApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Clip Generator — AI Highlight Workstation")
        self.resize(1180, 880)
        self.setMinimumSize(1000, 680)

        # Apply QSS
        self.setStyleSheet(QSS_STYLESHEET)

        # Configuration & Cache
        self.config = config_manager.load_config()
        self.selected_mp4_path = ""
        self.worker: Optional[ClipExtractionWorker] = None

        # Setup Logging
        self._init_logging()

        # Build UI & Wire Signals
        self._init_signals()
        self._build_ui()

    def _init_signals(self):
        self.signals = WorkerSignals()
        self.signals.log_message.connect(self.log_to_console)
        self.signals.status_changed.connect(self._update_status_ui)
        self.signals.finished.connect(self._on_processing_finished)
        self.signals.model_list_updated.connect(self._on_model_list_updated)

    def _init_logging(self):
        app_data = get_app_data_path()
        log_dir = os.path.join(app_data, "logs")
        os.makedirs(log_dir, exist_ok=True)

        formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

        manual_path = os.path.join(log_dir, "processor.log")
        self.manual_logger = logging.getLogger("pyside_manual_logger")
        self.manual_logger.setLevel(logging.INFO)
        self.manual_logger.propagate = False
        if not self.manual_logger.handlers:
            manual_handler = RotatingFileHandler(manual_path, maxBytes=1024*1024, backupCount=9, encoding='utf-8')
            manual_handler.setFormatter(formatter)
            self.manual_logger.addHandler(manual_handler)

    def _build_ui(self):
        central_widget = QtWidgets.QWidget(self)
        central_widget.setObjectName("CentralWidget")
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 1. Sidebar
        sidebar_frame = QtWidgets.QFrame(central_widget)
        sidebar_frame.setObjectName("SidebarFrame")
        sidebar_frame.setFixedWidth(220)
        sidebar_layout = QtWidgets.QVBoxLayout(sidebar_frame)
        sidebar_layout.setContentsMargins(14, 20, 14, 18)
        sidebar_layout.setSpacing(6)

        # Brand Header
        brand_layout = QtWidgets.QHBoxLayout()
        logo_label = QtWidgets.QLabel("⚡ CLIPGEN", sidebar_frame)
        logo_label.setObjectName("SidebarLogo")
        badge_label = QtWidgets.QLabel("PRO", sidebar_frame)
        badge_label.setObjectName("SidebarBadge")
        brand_layout.addWidget(logo_label)
        brand_layout.addStretch()
        brand_layout.addWidget(badge_label)
        sidebar_layout.addLayout(brand_layout)

        sub_label = QtWidgets.QLabel("AI Highlight Studio", sidebar_frame)
        sub_label.setStyleSheet("color: #475569; font-size: 11px; margin-bottom: 12px;")
        sidebar_layout.addWidget(sub_label)

        # Navigation Buttons
        self.nav_btn_group = QtWidgets.QButtonGroup(self)
        self.nav_btn_group.setExclusive(True)

        self.nav_extractor_btn = QtWidgets.QPushButton("  ⚡  Clip Extractor", sidebar_frame)
        self.nav_prompt_btn = QtWidgets.QPushButton("  📝  Prompt Strategy", sidebar_frame)
        self.nav_settings_btn = QtWidgets.QPushButton("  ⚙️  Configuration", sidebar_frame)
        self.nav_gallery_btn = QtWidgets.QPushButton("  🖼️  Clip Gallery", sidebar_frame)

        for i, btn in enumerate([self.nav_extractor_btn, self.nav_prompt_btn, self.nav_settings_btn, self.nav_gallery_btn]):
            btn.setCheckable(True)
            btn.setProperty("class", "nav-btn")
            btn.setCursor(QtCore.Qt.PointingHandCursor)
            self.nav_btn_group.addButton(btn, i)
            sidebar_layout.addWidget(btn)

        self.nav_btn_group.idClicked.connect(self._on_nav_clicked)
        self.nav_extractor_btn.setChecked(True)

        sidebar_layout.addStretch()

        # Hardware Status Pill
        hw_frame = QtWidgets.QFrame(sidebar_frame)
        hw_frame.setProperty("class", "hw-pill")
        hw_layout = QtWidgets.QVBoxLayout(hw_frame)
        hw_layout.setContentsMargins(8, 6, 8, 6)
        hw_layout.setSpacing(2)

        hw_status = get_hardware_status()
        hw_text = f"🚀 GPU: {hw_status['gpu_name'][:17]}..." if hw_status.get("cuda_available") else "⏳ Engine: CPU"
        self.hw_label = QtWidgets.QLabel(hw_text, hw_frame)
        self.hw_label.setStyleSheet("color: #10b981; font-size: 11px; font-weight: 600;")
        hw_layout.addWidget(self.hw_label)

        version_label = QtWidgets.QLabel("v2.2.0 • Standalone", hw_frame)
        version_label.setStyleSheet("color: #475569; font-size: 10px;")
        hw_layout.addWidget(version_label)

        sidebar_layout.addWidget(hw_frame)
        main_layout.addWidget(sidebar_frame)

        # 2. Main Stacked Content Pages
        self.stacked_widget = QtWidgets.QStackedWidget(central_widget)
        main_layout.addWidget(self.stacked_widget, 1)

        # Build Pages
        self.page_extractor = self._build_extractor_page()
        self.stacked_widget.addWidget(self.page_extractor)

        self.page_prompt = self._build_prompt_page()
        self.stacked_widget.addWidget(self.page_prompt)

        self.page_settings = self._build_settings_page()
        self.stacked_widget.addWidget(self.page_settings)

        self.page_gallery = self._build_gallery_page()
        self.stacked_widget.addWidget(self.page_gallery)

        # Initialize profile dropdowns across all tabs
        self._refresh_profile_combos()
        self._load_active_prompt_text()

    def _on_nav_clicked(self, page_index: int):
        self.stacked_widget.setCurrentIndex(page_index)
        if page_index == 3:
            self.populate_gallery()

    # ==============================================================================
    # PAGE 0: CLIP EXTRACTOR
    # ==============================================================================
    def _build_extractor_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(28, 24, 28, 20)
        layout.setSpacing(14)

        # Header
        header_layout = QtWidgets.QVBoxLayout()
        header_layout.setSpacing(2)
        title = QtWidgets.QLabel("AI Clip Extractor & Workstation", page)
        title.setProperty("class", "page-title")
        subtitle = QtWidgets.QLabel("Queue raw gaming VODs, transcribe audio with Whisper, and extract viral moments.", page)
        subtitle.setProperty("class", "page-subtitle")
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        layout.addLayout(header_layout)

        # Controls Card
        card = QtWidgets.QFrame(page)
        card.setProperty("class", "card")
        card_layout = QtWidgets.QGridLayout(card)
        card_layout.setContentsMargins(18, 16, 18, 16)
        card_layout.setSpacing(12)

        # Source Video Input
        video_label = QtWidgets.QLabel("Source Video File(s):", card)
        video_label.setProperty("class", "field-label")
        self.video_input = QtWidgets.QLineEdit(card)
        self.video_input.setPlaceholderText("Select video file (MP4, MKV, MOV, TS) or multiple separated by semicolon...")
        
        self.browse_video_btn = QtWidgets.QPushButton("Browse...", card)
        self.browse_video_btn.setProperty("class", "secondary-btn")
        self.browse_video_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.browse_video_btn.clicked.connect(self._browse_video)

        card_layout.addWidget(video_label, 0, 0)
        card_layout.addWidget(self.video_input, 0, 1)
        card_layout.addWidget(self.browse_video_btn, 0, 2)

        # Prompt Profile Selector
        prompt_label = QtWidgets.QLabel("Strategy Profile:", card)
        prompt_label.setProperty("class", "field-label")
        self.extractor_profile_combo = QtWidgets.QComboBox(card)
        self.extractor_profile_combo.setCursor(QtCore.Qt.PointingHandCursor)
        self.extractor_profile_combo.currentTextChanged.connect(self._on_extractor_profile_changed)

        self.edit_preset_btn = QtWidgets.QPushButton("✏️ Customize Prompts...", card)
        self.edit_preset_btn.setProperty("class", "secondary-btn")
        self.edit_preset_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.edit_preset_btn.clicked.connect(self._go_to_prompt_editor)

        card_layout.addWidget(prompt_label, 1, 0)
        card_layout.addWidget(self.extractor_profile_combo, 1, 1)
        card_layout.addWidget(self.edit_preset_btn, 1, 2)

        # Action Buttons
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.setSpacing(10)

        self.start_extract_btn = QtWidgets.QPushButton("⚡ Start AI Highlight Extraction", card)
        self.start_extract_btn.setProperty("class", "primary-btn")
        self.start_extract_btn.setMinimumHeight(42)
        self.start_extract_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.start_extract_btn.clicked.connect(self._start_extraction)

        self.cancel_extract_btn = QtWidgets.QPushButton("🛑 Cancel", card)
        self.cancel_extract_btn.setProperty("class", "danger-btn")
        self.cancel_extract_btn.setMinimumHeight(42)
        self.cancel_extract_btn.setEnabled(False)
        self.cancel_extract_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.cancel_extract_btn.clicked.connect(self._cancel_extraction)

        btn_layout.addWidget(self.start_extract_btn, 1)
        btn_layout.addWidget(self.cancel_extract_btn)
        card_layout.addLayout(btn_layout, 2, 0, 1, 3)

        layout.addWidget(card)

        # Status Bar & Progress
        status_bar_layout = QtWidgets.QHBoxLayout()
        self.status_label = QtWidgets.QLabel("● Status: Ready", page)
        self.status_label.setStyleSheet("color: #10b981; font-weight: 600;")
        status_bar_layout.addWidget(self.status_label)
        status_bar_layout.addStretch()

        self.progress_bar = QtWidgets.QProgressBar(page)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.hide()
        status_bar_layout.addWidget(self.progress_bar)

        layout.addLayout(status_bar_layout)

        # Terminal / Log Viewer Card
        console_card = QtWidgets.QFrame(page)
        console_card.setProperty("class", "card")
        console_layout = QtWidgets.QVBoxLayout(console_card)
        console_layout.setContentsMargins(14, 12, 14, 12)
        console_layout.setSpacing(8)

        console_header = QtWidgets.QHBoxLayout()
        console_title = QtWidgets.QLabel("💻 Execution Stream Logs", console_card)
        console_title.setProperty("class", "card-title")
        
        clear_btn = QtWidgets.QPushButton("Clear", console_card)
        clear_btn.setProperty("class", "secondary-btn")
        clear_btn.setFixedHeight(26)
        clear_btn.setCursor(QtCore.Qt.PointingHandCursor)
        clear_btn.clicked.connect(self._clear_console)

        console_header.addWidget(console_title)
        console_header.addStretch()
        console_header.addWidget(clear_btn)
        console_layout.addLayout(console_header)

        self.console_text = QtWidgets.QPlainTextEdit(console_card)
        self.console_text.setReadOnly(True)
        self.console_text.setStyleSheet("""
            background-color: #05080e;
            color: #f1f5f9;
            font-family: 'Consolas', monospace;
            font-size: 12px;
            border: 1px solid #141c2c;
            border-radius: 6px;
        """)
        console_layout.addWidget(self.console_text, 1)

        layout.addWidget(console_card, 1)
        return page

    def _browse_video(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select Video File(s)", "", "Video Files (*.mp4 *.mkv *.avi *.mov *.flv *.ts *.m4v);;All Files (*)"
        )
        if paths:
            self.video_input.setText(";".join(paths))

    def _clear_console(self):
        self.console_text.clear()

    def _start_extraction(self):
        input_val = self.video_input.text().strip()
        if not input_val:
            self.log_to_console("❌ Input Required: Please select a valid video file.", "manual")
            return

        profile = self.extractor_profile_combo.currentText()
        self.start_extract_btn.setEnabled(False)
        self.browse_video_btn.setEnabled(False)
        self.cancel_extract_btn.setEnabled(True)
        self.progress_bar.show()
        self._clear_console()
        self._update_status_ui("Processing video...", "accent_primary")

        self.worker = ClipExtractionWorker(input_val, profile, self.signals)
        self.worker.start()

    def _cancel_extraction(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.cancel_extract_btn.setEnabled(False)
            self._update_status_ui("Cancelling...", "accent_danger")

    def _on_processing_finished(self, success: bool):
        self.start_extract_btn.setEnabled(True)
        self.browse_video_btn.setEnabled(True)
        self.cancel_extract_btn.setEnabled(False)
        self.progress_bar.hide()
        if success:
            self._update_status_ui("Ready (Job completed successfully)", "accent_success")
            self._send_discord_alert("Queue Execution Complete")
        else:
            self._update_status_ui("Ready (Job halted or failed)", "accent_danger")
        self.populate_gallery()

    def _update_status_ui(self, status: str, color_key: str):
        colors = {
            "accent_success": "#10b981",
            "accent_danger": "#ef4444",
            "accent_primary": "#3b82f6",
            "text_secondary": "#94a3b8"
        }
        color = colors.get(color_key, "#10b981")
        self.status_label.setText(f"● Status: {status}")
        self.status_label.setStyleSheet(f"color: {color}; font-weight: 600;")

    def log_to_console(self, text: str, source: str = "system"):
        clean_msg = text.strip()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if clean_msg.startswith("[") and "]" in clean_msg[:25]:
            display_text = clean_msg
        else:
            display_text = f"[{timestamp}] {clean_msg}"

        color_hex = "#f1f5f9"
        if "❌" in clean_msg or "error" in clean_msg.lower():
            color_hex = "#f87171"
        elif "✅" in clean_msg or "✨" in clean_msg or "🏁" in clean_msg:
            color_hex = "#34d399"
        elif "🧠" in clean_msg or "🌌" in clean_msg or "🤖" in clean_msg or "🎯" in clean_msg or "🚀" in clean_msg or "⚡" in clean_msg:
            color_hex = "#38bdf8"
        elif "✂️" in clean_msg or "🎞️" in clean_msg or "📸" in clean_msg or "📱" in clean_msg:
            color_hex = "#fbbf24"

        html = f'<span style="color:{color_hex};">{display_text}</span>'
        
        # Smooth autoscroll without viewport jumping
        scrollbar = self.console_text.verticalScrollBar()
        was_at_bottom = scrollbar.value() >= (scrollbar.maximum() - 25)

        cursor = self.console_text.textCursor()
        cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
        self.console_text.setTextCursor(cursor)

        self.console_text.appendHtml(html)

        if was_at_bottom:
            scrollbar.setValue(scrollbar.maximum())

        if hasattr(self, 'manual_logger'):
            self.manual_logger.info(clean_msg)

    # ==============================================================================
    # PAGE 1: PROMPT STRATEGY EDITOR
    # ==============================================================================
    def _build_prompt_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(28, 24, 28, 20)
        layout.setSpacing(14)

        # Header
        title = QtWidgets.QLabel("AI Prompt & Strategy Editor", page)
        title.setProperty("class", "page-title")
        subtitle = QtWidgets.QLabel("Create and edit prompt templates, virality thresholds, and comedic timing instructions.", page)
        subtitle.setProperty("class", "page-subtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        # Profile Selector Bar
        bar = QtWidgets.QFrame(page)
        bar.setProperty("class", "card")
        bar_layout = QtWidgets.QHBoxLayout(bar)
        bar_layout.setContentsMargins(16, 10, 16, 10)
        bar_layout.setSpacing(10)

        p_label = QtWidgets.QLabel("Preset to Edit:", bar)
        p_label.setProperty("class", "field-label")
        self.prompt_profile_combo = QtWidgets.QComboBox(bar)
        self.prompt_profile_combo.setFixedWidth(240)
        self.prompt_profile_combo.currentTextChanged.connect(self._on_prompt_profile_changed)

        new_btn = QtWidgets.QPushButton("➕ New Profile", bar)
        new_btn.setProperty("class", "success-btn")
        new_btn.setCursor(QtCore.Qt.PointingHandCursor)
        new_btn.clicked.connect(self._create_prompt_profile)

        self.delete_profile_btn = QtWidgets.QPushButton("🗑️ Delete Preset", bar)
        self.delete_profile_btn.setProperty("class", "danger-btn")
        self.delete_profile_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.delete_profile_btn.clicked.connect(self._delete_prompt_profile)

        bar_layout.addWidget(p_label)
        bar_layout.addWidget(self.prompt_profile_combo)
        bar_layout.addWidget(new_btn)
        bar_layout.addStretch()
        bar_layout.addWidget(self.delete_profile_btn)
        layout.addWidget(bar)

        # Editor Card
        editor_card = QtWidgets.QFrame(page)
        editor_card.setProperty("class", "card")
        editor_layout = QtWidgets.QVBoxLayout(editor_card)
        editor_layout.setContentsMargins(16, 16, 16, 16)
        editor_layout.setSpacing(12)

        self.prompt_edit_box = QtWidgets.QPlainTextEdit(editor_card)
        self.prompt_edit_box.setStyleSheet("""
            background-color: #080c14;
            color: #f8fafc;
            font-family: 'Consolas', monospace;
            font-size: 13px;
            border: 1px solid #1c273c;
            border-radius: 6px;
            line-height: 1.4;
        """)
        editor_layout.addWidget(self.prompt_edit_box, 1)

        # Footer
        footer_layout = QtWidgets.QHBoxLayout()
        self.prompt_hint_label = QtWidgets.QLabel("💡 Tip: Output format must remain strictly JSON with start_time, end_time, and virality_score fields.", editor_card)
        self.prompt_hint_label.setStyleSheet("color: #64748b; font-size: 11px;")
        
        self.save_prompt_btn = QtWidgets.QPushButton("💾 Save Prompt", editor_card)
        self.save_prompt_btn.setProperty("class", "primary-btn")
        self.save_prompt_btn.setFixedWidth(140)
        self.save_prompt_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.save_prompt_btn.clicked.connect(self._save_prompt_profile)

        footer_layout.addWidget(self.prompt_hint_label)
        footer_layout.addStretch()
        footer_layout.addWidget(self.save_prompt_btn)
        editor_layout.addLayout(footer_layout)

        layout.addWidget(editor_card, 1)
        return page

    def _go_to_prompt_editor(self):
        self.nav_prompt_btn.setChecked(True)
        self.stacked_widget.setCurrentIndex(1)

    def _refresh_profile_combos(self):
        profiles_dict = self.config.setdefault("prompts", {}).setdefault("profiles", {})
        default_prompt_text = config_manager.get_default_config()["prompts"]["profiles"]["Default"]
        
        if "Default" not in profiles_dict or not str(profiles_dict.get("Default", "")).strip():
            profiles_dict["Default"] = default_prompt_text
            
        profiles = list(profiles_dict.keys())
        active = self.config.get("prompts", {}).get("active_profile", "Default")
        if active not in profiles:
            active = "Default"
            self.config["prompts"]["active_profile"] = active

        if hasattr(self, 'extractor_profile_combo'):
            self.extractor_profile_combo.blockSignals(True)
            self.extractor_profile_combo.clear()
            self.extractor_profile_combo.addItems(profiles)
            self.extractor_profile_combo.setCurrentText(active)
            self.extractor_profile_combo.blockSignals(False)

        if hasattr(self, 'prompt_profile_combo'):
            self.prompt_profile_combo.blockSignals(True)
            self.prompt_profile_combo.clear()
            self.prompt_profile_combo.addItems(profiles)
            self.prompt_profile_combo.setCurrentText(active)
            self.prompt_profile_combo.blockSignals(False)

    def _load_active_prompt_text(self):
        active = self.config.get("prompts", {}).get("active_profile", "Default")
        self._on_prompt_profile_changed(active)

    def _on_extractor_profile_changed(self, profile_name: str):
        if not profile_name:
            return
        self.config.setdefault("prompts", {})["active_profile"] = profile_name
        config_manager.save_config(self.config)
        if hasattr(self, 'prompt_profile_combo'):
            self.prompt_profile_combo.blockSignals(True)
            self.prompt_profile_combo.setCurrentText(profile_name)
            self.prompt_profile_combo.blockSignals(False)
        self._on_prompt_profile_changed(profile_name)

    def _on_prompt_profile_changed(self, profile_name: str):
        if not profile_name:
            return
        self.config.setdefault("prompts", {})["active_profile"] = profile_name
        config_manager.save_config(self.config)
        text = self.config.get("prompts", {}).get("profiles", {}).get(profile_name, "")
        self.prompt_edit_box.setPlainText(text)
        
        if hasattr(self, 'extractor_profile_combo'):
            self.extractor_profile_combo.blockSignals(True)
            self.extractor_profile_combo.setCurrentText(profile_name)
            self.extractor_profile_combo.blockSignals(False)

        if profile_name == "Default":
            self.prompt_edit_box.setReadOnly(True)
            self.prompt_edit_box.setStyleSheet("""
                background-color: #060910;
                color: #64748b;
                font-family: 'Consolas', monospace;
                font-size: 13px;
                border: 1px solid #141c2c;
                border-radius: 6px;
                line-height: 1.4;
            """)
            if hasattr(self, 'save_prompt_btn'):
                self.save_prompt_btn.setEnabled(False)
                self.save_prompt_btn.setToolTip("Default profile is protected. Click '➕ New Profile' to customize.")
            if hasattr(self, 'delete_profile_btn'):
                self.delete_profile_btn.setEnabled(False)
                self.delete_profile_btn.setToolTip("Default built-in profile cannot be deleted.")
            if hasattr(self, 'prompt_hint_label'):
                self.prompt_hint_label.setText("🔒 Default profile is built-in (read-only). Click '➕ New Profile' to create and edit your own strategy.")
        else:
            self.prompt_edit_box.setReadOnly(False)
            self.prompt_edit_box.setStyleSheet("""
                background-color: #080c14;
                color: #f8fafc;
                font-family: 'Consolas', monospace;
                font-size: 13px;
                border: 1px solid #1c273c;
                border-radius: 6px;
                line-height: 1.4;
            """)
            if hasattr(self, 'save_prompt_btn'):
                self.save_prompt_btn.setEnabled(True)
                self.save_prompt_btn.setToolTip("")
            if hasattr(self, 'delete_profile_btn'):
                self.delete_profile_btn.setEnabled(True)
                self.delete_profile_btn.setToolTip("")
            if hasattr(self, 'prompt_hint_label'):
                self.prompt_hint_label.setText("💡 Tip: Output format must remain strictly JSON with start_time, end_time, and virality_score fields.")

    def _create_prompt_profile(self):
        name, ok = QtWidgets.QInputDialog.getText(self, "New Profile", "Enter name for new prompt profile:")
        if ok and name.strip():
            clean_name = name.strip()
            if clean_name.lower() == "default":
                QtWidgets.QMessageBox.warning(self, "Reserved Name", "The name 'Default' is reserved for the built-in profile.")
                return
            if clean_name in self.config["prompts"]["profiles"]:
                QtWidgets.QMessageBox.warning(self, "Profile Exists", f"Profile '{clean_name}' already exists.")
                return
            default_text = self.config["prompts"]["profiles"].get("Default", config_manager.get_default_config()["prompts"]["profiles"]["Default"])
            self.config["prompts"]["profiles"][clean_name] = default_text
            self.config["prompts"]["active_profile"] = clean_name
            config_manager.save_config(self.config)
            self._refresh_profile_combos()
            self._load_active_prompt_text()
            self.log_to_console(f"📝 Created new prompt profile: '{clean_name}'", "system")

    def _save_prompt_profile(self):
        active = self.prompt_profile_combo.currentText().strip()
        if not active:
            return
        if active == "Default":
            QtWidgets.QMessageBox.warning(
                self, "Protected Profile",
                "The Default profile is protected and cannot be modified.\nClick '➕ New Profile' to create and save your custom prompt."
            )
            return
        self.config["prompts"]["profiles"][active] = self.prompt_edit_box.toPlainText().strip()
        config_manager.save_config(self.config)
        self.log_to_console(f"✅ Saved prompt profile: '{active}'", "system")
        QtWidgets.QMessageBox.information(self, "Saved", f"Prompt profile '{active}' saved successfully!")

    def _delete_prompt_profile(self):
        active = self.prompt_profile_combo.currentText().strip()
        if not active:
            return
        if active == "Default":
            QtWidgets.QMessageBox.warning(self, "Protected Profile", "The Default built-in profile cannot be deleted.")
            return

        profiles = self.config.get("prompts", {}).get("profiles", {})
        if active not in profiles:
            return

        reply = QtWidgets.QMessageBox.question(
            self, "Confirm Delete", f"Are you sure you want to delete profile '{active}'?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        if reply == QtWidgets.QMessageBox.Yes:
            del self.config["prompts"]["profiles"][active]
            self.config["prompts"]["active_profile"] = "Default"
            config_manager.save_config(self.config)
            self._refresh_profile_combos()
            self._load_active_prompt_text()
            self.log_to_console(f"🗑️ Deleted prompt profile: '{active}'", "system")

    # ==============================================================================
    # PAGE 2: APPLICATION CONFIGURATION (MODERN UN-NESTED HUB)
    # ==============================================================================
    def _build_settings_page(self) -> QtWidgets.QWidget:
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)

        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(16)

        # Title
        title = QtWidgets.QLabel("Application Configuration", container)
        title.setProperty("class", "page-title")
        subtitle = QtWidgets.QLabel("Configure AI LLM engines, Whisper audio transcription, and video hardware encoders.", container)
        subtitle.setProperty("class", "page-subtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        # 1. AI Cloud Providers Hub (Sleek Tabbed Segmented Controller)
        ai_card = QtWidgets.QFrame(container)
        ai_card.setProperty("class", "card")
        ai_layout = QtWidgets.QVBoxLayout(ai_card)
        ai_layout.setContentsMargins(18, 16, 18, 18)
        ai_layout.setSpacing(14)

        ai_header = QtWidgets.QHBoxLayout()
        ai_title = QtWidgets.QLabel("AI Cloud Engines & Credentials", ai_card)
        ai_title.setProperty("class", "card-title")
        ai_header.addWidget(ai_title)
        ai_header.addStretch()
        ai_layout.addLayout(ai_header)

        # Provider Selector Bar (Horizontal Segmented Pills)
        prov_tabs_layout = QtWidgets.QHBoxLayout()
        prov_tabs_layout.setSpacing(8)

        self.provider_tab_group = QtWidgets.QButtonGroup(self)
        self.provider_tab_group.setExclusive(True)

        providers_meta = [
            ("openai", "🤖 OpenAI / Custom", 0),
            ("deepseek", "⚡ DeepSeek", 1),
            ("anthropic", "🧠 Anthropic Claude", 2),
            ("xai", "🌌 xAI / Grok", 3),
            ("google", "💎 Google Gemini", 4),
        ]

        self.provider_tabs = []
        for prov_id, label, idx in providers_meta:
            tab_btn = QtWidgets.QPushButton(label, ai_card)
            tab_btn.setProperty("class", "provider-tab")
            tab_btn.setCheckable(True)
            tab_btn.setCursor(QtCore.Qt.PointingHandCursor)
            tab_btn.setProperty("prov_id", prov_id)
            self.provider_tab_group.addButton(tab_btn, idx)
            prov_tabs_layout.addWidget(tab_btn)
            self.provider_tabs.append(tab_btn)

        ai_layout.addLayout(prov_tabs_layout)

        # Stacked Widget for Provider Form Panels
        self.provider_stack = QtWidgets.QStackedWidget(ai_card)
        active_prov = self.config.get("active_ai_provider", "openai")

        self.provider_btn_group = QtWidgets.QButtonGroup(self)
        self.provider_btn_group.setExclusive(True)

        # 1. OpenAI Panel
        openai_panel, self.openai_frame, self.openai_radio, self.openai_key_edit, self.openai_model_combo, self.openai_base_url_edit = self._create_provider_panel(
            "OpenAI & Compatible Endpoints", "openai", active_prov,
            self.config.get("openai", {}).get("api_key", ""),
            model_fetcher.MAIN_MODELS["openai"],
            self.config.get("openai_model", "gpt-5.5"),
            self.config.get("openai", {}).get("base_url", ""),
            self._test_openai_key,
            is_openai=True
        )
        self.provider_stack.addWidget(openai_panel)

        # 2. DeepSeek Panel
        deepseek_panel, self.deepseek_frame, self.deepseek_radio, self.deepseek_key_edit, self.deepseek_model_combo, _ = self._create_provider_panel(
            "DeepSeek Cloud Intelligence", "deepseek", active_prov,
            self.config.get("deepseek", {}).get("api_key", ""),
            model_fetcher.MAIN_MODELS["deepseek"],
            self.config.get("deepseek_model", "deepseek-v4-flash"),
            "",
            self._test_deepseek_key
        )
        self.provider_stack.addWidget(deepseek_panel)

        # 3. Anthropic Panel
        anthropic_panel, self.anthropic_frame, self.anthropic_radio, self.anthropic_key_edit, self.anthropic_model_combo, _ = self._create_provider_panel(
            "Anthropic Claude Engine", "anthropic", active_prov,
            self.config.get("anthropic", {}).get("api_key", ""),
            model_fetcher.MAIN_MODELS["anthropic"],
            self.config.get("anthropic_model", "claude-sonnet-5"),
            "",
            self._test_anthropic_key
        )
        self.provider_stack.addWidget(anthropic_panel)

        # 4. xAI Panel
        xai_panel, self.xai_frame, self.xai_radio, self.xai_key_edit, self.xai_model_combo, _ = self._create_provider_panel(
            "xAI Grok Intelligence", "xai", active_prov,
            self.config.get("xai", {}).get("api_key", ""),
            model_fetcher.MAIN_MODELS["xai"],
            self.config.get("xai_model", "grok-4.3"),
            "",
            self._test_xai_key
        )
        self.provider_stack.addWidget(xai_panel)

        # 5. Google Gemini Panel
        google_panel, self.google_frame, self.google_radio, self.google_key_edit, self.google_model_combo, _ = self._create_provider_panel(
            "Google Gemini (2M Long Context)", "google", active_prov,
            self.config.get("google", {}).get("api_key", ""),
            model_fetcher.MAIN_MODELS["google"],
            self.config.get("google_model", "gemini-3.5-flash"),
            "",
            self._test_google_key
        )
        self.provider_stack.addWidget(google_panel)

        ai_layout.addWidget(self.provider_stack)

        # Select initial active provider tab
        tab_indices = {"openai": 0, "deepseek": 1, "anthropic": 2, "xai": 3, "google": 4}
        initial_idx = tab_indices.get(active_prov, 0)
        self.provider_tabs[initial_idx].setChecked(True)
        self.provider_stack.setCurrentIndex(initial_idx)

        self.provider_tab_group.idClicked.connect(lambda idx: self.provider_stack.setCurrentIndex(idx))

        layout.addWidget(ai_card)

        # 2. Local Whisper & Audio Intelligence Card
        whisper_card = QtWidgets.QFrame(container)
        whisper_card.setProperty("class", "card")
        whisper_layout = QtWidgets.QGridLayout(whisper_card)
        whisper_layout.setContentsMargins(18, 16, 18, 16)
        whisper_layout.setSpacing(12)

        whisper_title = QtWidgets.QLabel("Local Whisper & Audio Intelligence", whisper_card)
        whisper_title.setProperty("class", "card-title")
        whisper_layout.addWidget(whisper_title, 0, 0, 1, 2)

        # Whisper Model
        w_model_label = QtWidgets.QLabel("Whisper Transcribe Model:", whisper_card)
        w_model_label.setProperty("class", "field-label")
        self.whisper_model_combo = QtWidgets.QComboBox(whisper_card)
        saved_whisper_model = self.config.get("openai", {}).get("whisper_model", "medium")
        self.whisper_model_combo.addItems(editor.get_whisper_models(saved_whisper_model))
        self.whisper_model_combo.setCurrentText(saved_whisper_model)

        whisper_layout.addWidget(w_model_label, 1, 0)
        whisper_layout.addWidget(self.whisper_model_combo, 1, 1)

        # Audio Language
        w_lang_label = QtWidgets.QLabel("Spoken Language:", whisper_card)
        w_lang_label.setProperty("class", "field-label")
        self.whisper_lang_combo = QtWidgets.QComboBox(whisper_card)
        self.whisper_lang_combo.addItems([
            "Auto-Detect", "English", "Russian", "Spanish", "French", "German", "Italian",
            "Portuguese", "Japanese", "Korean", "Chinese", "Ukrainian", "Polish", "Turkish"
        ])
        self.whisper_lang_combo.setCurrentText(self.config.get("openai", {}).get("whisper_language", "Auto-Detect"))

        whisper_layout.addWidget(w_lang_label, 2, 0)
        whisper_layout.addWidget(self.whisper_lang_combo, 2, 1)

        # Audio Switches
        self.downmix_checkbox = QtWidgets.QCheckBox("Downmix Multi-Track OBS Audio (amix filter)", whisper_card)
        self.downmix_checkbox.setChecked(self.config.get("settings", {}).get("audio_downmix", True))
        whisper_layout.addWidget(self.downmix_checkbox, 3, 0)

        self.peak_checkbox = QtWidgets.QCheckBox("Audio Peak & Screaming Loudness Mapping", whisper_card)
        self.peak_checkbox.setChecked(self.config.get("settings", {}).get("audio_peak_detection", True))
        whisper_layout.addWidget(self.peak_checkbox, 3, 1)

        self.combat_checkbox = QtWidgets.QCheckBox("Gunfight & Combat Transient Spikes Detection", whisper_card)
        self.combat_checkbox.setChecked(self.config.get("settings", {}).get("combat_detection", True))
        whisper_layout.addWidget(self.combat_checkbox, 4, 0, 1, 2)

        clips_label = QtWidgets.QLabel("Target clips per hour of video (0 = no limit):", whisper_card)
        clips_label.setProperty("class", "field-label")
        self.clips_per_hour_spin = QtWidgets.QSpinBox(whisper_card)
        self.clips_per_hour_spin.setRange(0, 60)
        self.clips_per_hour_spin.setValue(int(self.config.get("settings", {}).get("clips_per_hour", 12)))
        whisper_layout.addWidget(clips_label, 5, 0)
        whisper_layout.addWidget(self.clips_per_hour_spin, 5, 1)

        layout.addWidget(whisper_card)

        # 3. Video Export & Hardware Acceleration Card
        proc_card = QtWidgets.QFrame(container)
        proc_card.setProperty("class", "card")
        proc_layout = QtWidgets.QGridLayout(proc_card)
        proc_layout.setContentsMargins(18, 16, 18, 16)
        proc_layout.setSpacing(12)

        proc_title = QtWidgets.QLabel("Video Export & Hardware Acceleration", proc_card)
        proc_title.setProperty("class", "card-title")
        proc_layout.addWidget(proc_title, 0, 0, 1, 2)

        self.hw_encode_checkbox = QtWidgets.QCheckBox("GPU Hardware Acceleration (NVENC / AMF)", proc_card)
        self.hw_encode_checkbox.setChecked(self.config.get("settings", {}).get("hardware_encoding", False))
        proc_layout.addWidget(self.hw_encode_checkbox, 1, 0)

        self.vr_checkbox = QtWidgets.QCheckBox("Apply VR Anti-Shake Deshake Filter", proc_card)
        self.vr_checkbox.setChecked(self.config.get("settings", {}).get("vr_stabilization", False))
        proc_layout.addWidget(self.vr_checkbox, 1, 1)

        self.vertical_checkbox = QtWidgets.QCheckBox("Generate Vertical 9:16 Shorts", proc_card)
        self.vertical_checkbox.setChecked(self.config.get("settings", {}).get("vertical_export", False))
        proc_layout.addWidget(self.vertical_checkbox, 2, 0)

        self.vertical_mode_combo = QtWidgets.QComboBox(proc_card)
        self.vertical_mode_combo.addItems([
            "Standard Center Crop", "Left-Third (Facecam)", "Right-Third",
            "Blurred Background (Portrait)", "Custom Coordinates"
        ])
        self.vertical_mode_combo.setCurrentText(self.config.get("settings", {}).get("vertical_mode", "Standard Center Crop"))
        proc_layout.addWidget(self.vertical_mode_combo, 2, 1)

        layout.addWidget(proc_card)

        # 4. Storage & Integrations Card
        storage_card = QtWidgets.QFrame(container)
        storage_card.setProperty("class", "card")
        storage_layout = QtWidgets.QGridLayout(storage_card)
        storage_layout.setContentsMargins(18, 16, 18, 16)
        storage_layout.setSpacing(12)

        storage_title = QtWidgets.QLabel("Storage & Discord Notifications", storage_card)
        storage_title.setProperty("class", "card-title")
        storage_layout.addWidget(storage_title, 0, 0, 1, 3)

        clips_dir_label = QtWidgets.QLabel("Generated Clips Folder:", storage_card)
        clips_dir_label.setProperty("class", "field-label")
        self.clips_dir_edit = QtWidgets.QLineEdit(storage_card)
        self.clips_dir_edit.setText(self.config.get("settings", {}).get("clips_dir", ""))
        self.clips_dir_edit.setPlaceholderText("C:\\Videos\\Clips")
        browse_clips_btn = QtWidgets.QPushButton("Browse...", storage_card)
        browse_clips_btn.setProperty("class", "secondary-btn")
        browse_clips_btn.setCursor(QtCore.Qt.PointingHandCursor)
        browse_clips_btn.clicked.connect(self._browse_clips_dir)

        storage_layout.addWidget(clips_dir_label, 1, 0)
        storage_layout.addWidget(self.clips_dir_edit, 1, 1)
        storage_layout.addWidget(browse_clips_btn, 1, 2)

        discord_label = QtWidgets.QLabel("Discord Webhook URL:", storage_card)
        discord_label.setProperty("class", "field-label")
        self.discord_edit = QtWidgets.QLineEdit(storage_card)
        self.discord_edit.setText(self.config.get("integrations", {}).get("discord_webhook", ""))
        self.discord_edit.setPlaceholderText("https://discord.com/api/webhooks/...")
        test_discord_btn = QtWidgets.QPushButton("Test Alert", storage_card)
        test_discord_btn.setProperty("class", "secondary-btn")
        test_discord_btn.setCursor(QtCore.Qt.PointingHandCursor)
        test_discord_btn.clicked.connect(self._test_discord_webhook)

        storage_layout.addWidget(discord_label, 2, 0)
        storage_layout.addWidget(self.discord_edit, 2, 1)
        storage_layout.addWidget(test_discord_btn, 2, 2)

        layout.addWidget(storage_card)

        # Save Button
        save_layout = QtWidgets.QHBoxLayout()
        save_layout.addStretch()
        self.save_settings_btn = QtWidgets.QPushButton("💾 Save Configuration", container)
        self.save_settings_btn.setProperty("class", "primary-btn")
        self.save_settings_btn.setMinimumHeight(42)
        self.save_settings_btn.setFixedWidth(180)
        self.save_settings_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.save_settings_btn.clicked.connect(self._save_settings)
        save_layout.addWidget(self.save_settings_btn)

        layout.addLayout(save_layout)
        scroll_area.setWidget(container)
        return scroll_area

    def _create_provider_panel(self, title: str, prov_id: str, active_prov: str, api_key: str, models: List[str], current_model: str, base_url: str, test_func, is_openai: bool = False):
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 8, 0, 4)
        layout.setSpacing(10)

        # Top Bar with Radio
        top_bar = QtWidgets.QHBoxLayout()
        radio = QtWidgets.QRadioButton(f"Use {title} as Primary Engine", panel)
        radio.setChecked(active_prov == prov_id)
        radio.setProperty("prov_id", prov_id)
        radio.setStyleSheet("font-weight: 600; color: #f8fafc;")
        self.provider_btn_group.addButton(radio)
        top_bar.addWidget(radio)
        top_bar.addStretch()
        layout.addLayout(top_bar)

        # Form Grid
        grid_frame = QtWidgets.QFrame(panel)
        grid_frame.setProperty("class", "provider-grid")
        grid = QtWidgets.QGridLayout(grid_frame)
        grid.setContentsMargins(14, 14, 14, 14)
        grid.setSpacing(10)

        # Row 0: API Key
        key_label = QtWidgets.QLabel("API Key:", grid_frame)
        key_label.setProperty("class", "field-label")
        key_edit = QtWidgets.QLineEdit(grid_frame)
        key_edit.setEchoMode(QtWidgets.QLineEdit.Password)
        key_edit.setText(api_key)
        key_edit.setPlaceholderText("Enter provider API Key...")

        test_btn = QtWidgets.QPushButton("⚡ Test Key", grid_frame)
        test_btn.setProperty("class", "secondary-btn")
        test_btn.setCursor(QtCore.Qt.PointingHandCursor)
        test_btn.clicked.connect(test_func)

        grid.addWidget(key_label, 0, 0)
        grid.addWidget(key_edit, 0, 1)
        grid.addWidget(test_btn, 0, 2)

        # Row 1: Model Dropdown
        model_label = QtWidgets.QLabel("Chat Model:", grid_frame)
        model_label.setProperty("class", "field-label")
        model_combo = QtWidgets.QComboBox(grid_frame)
        model_combo.addItems(models)
        if current_model in models:
            model_combo.setCurrentText(current_model)

        grid.addWidget(model_label, 1, 0)
        grid.addWidget(model_combo, 1, 1, 1, 2)

        base_url_edit = None
        if is_openai:
            base_label = QtWidgets.QLabel("Base URL:", grid_frame)
            base_label.setProperty("class", "field-label")
            base_url_edit = QtWidgets.QLineEdit(grid_frame)
            base_url_edit.setPlaceholderText("https://api.openai.com/v1 (or OpenRouter/Local vLLM)")
            base_url_edit.setText(base_url)
            grid.addWidget(base_label, 2, 0)
            grid.addWidget(base_url_edit, 2, 1, 1, 2)

        layout.addWidget(grid_frame)
        return panel, grid_frame, radio, key_edit, model_combo, base_url_edit

    # ==============================================================================
    # PAGE 3: EXTRACTED CLIP GALLERY
    # ==============================================================================
    def _build_gallery_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(28, 24, 28, 20)
        layout.setSpacing(14)

        # Header
        title = QtWidgets.QLabel("Extracted Clip Gallery", page)
        title.setProperty("class", "page-title")
        subtitle = QtWidgets.QLabel("Browse generated highlight clips, inspect AI viral reasoning, and play or delete videos.", page)
        subtitle.setProperty("class", "page-subtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        # Filters Bar
        f_bar = QtWidgets.QFrame(page)
        f_bar.setProperty("class", "card")
        f_layout = QtWidgets.QHBoxLayout(f_bar)
        f_layout.setContentsMargins(14, 10, 14, 10)
        f_layout.setSpacing(12)

        sort_label = QtWidgets.QLabel("Sort By:", f_bar)
        sort_label.setProperty("class", "field-label")
        self.gallery_sort_combo = QtWidgets.QComboBox(f_bar)
        self.gallery_sort_combo.addItems(["Date (Newest)", "Date (Oldest)", "Virality (High)", "Virality (Low)"])
        self.gallery_sort_combo.currentTextChanged.connect(lambda _: self.populate_gallery())

        type_label = QtWidgets.QLabel("Type:", f_bar)
        type_label.setProperty("class", "field-label")
        self.gallery_type_combo = QtWidgets.QComboBox(f_bar)
        self.gallery_type_combo.addItems(["All", "Horizontal (16:9)", "Vertical (9:16)"])
        self.gallery_type_combo.currentTextChanged.connect(lambda _: self.populate_gallery())

        score_label = QtWidgets.QLabel("Min Score:", f_bar)
        score_label.setProperty("class", "field-label")
        self.gallery_score_combo = QtWidgets.QComboBox(f_bar)
        self.gallery_score_combo.addItems(["All", "7+", "8+", "9+"])
        self.gallery_score_combo.currentTextChanged.connect(lambda _: self.populate_gallery())

        refresh_btn = QtWidgets.QPushButton("🔄 Refresh", f_bar)
        refresh_btn.setProperty("class", "secondary-btn")
        refresh_btn.setCursor(QtCore.Qt.PointingHandCursor)
        refresh_btn.clicked.connect(self.populate_gallery)

        delete_all_btn = QtWidgets.QPushButton("🗑️ Delete All Clips", f_bar)
        delete_all_btn.setProperty("class", "danger-btn")
        delete_all_btn.setCursor(QtCore.Qt.PointingHandCursor)
        delete_all_btn.clicked.connect(self._delete_all_clips)

        f_layout.addWidget(sort_label)
        f_layout.addWidget(self.gallery_sort_combo)
        f_layout.addWidget(type_label)
        f_layout.addWidget(self.gallery_type_combo)
        f_layout.addWidget(score_label)
        f_layout.addWidget(self.gallery_score_combo)
        f_layout.addWidget(refresh_btn)
        f_layout.addStretch()
        f_layout.addWidget(delete_all_btn)
        layout.addWidget(f_bar)

        # Splitter Layout (Left: Clip List, Right: Details)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, page)

        self.clip_list_widget = QtWidgets.QListWidget(splitter)
        self.clip_list_widget.currentItemChanged.connect(self._on_clip_selected)
        splitter.addWidget(self.clip_list_widget)

        # Details Panel
        self.details_card = QtWidgets.QFrame(splitter)
        self.details_card.setProperty("class", "card")
        d_layout = QtWidgets.QVBoxLayout(self.details_card)
        d_layout.setContentsMargins(18, 16, 18, 16)
        d_layout.setSpacing(12)

        self.detail_title = QtWidgets.QLabel("Select a clip from the list to preview", self.details_card)
        self.detail_title.setStyleSheet("font-size: 15px; font-weight: 700; color: #ffffff;")
        self.detail_title.setWordWrap(True)
        d_layout.addWidget(self.detail_title)

        self.detail_score = QtWidgets.QLabel("Virality Score: --", self.details_card)
        self.detail_score.setStyleSheet("color: #38bdf8; font-weight: 600; font-size: 13px;")
        d_layout.addWidget(self.detail_score)

        # Thumbnail Box
        self.detail_thumb_label = QtWidgets.QLabel(self.details_card)
        self.detail_thumb_label.setAlignment(QtCore.Qt.AlignCenter)
        self.detail_thumb_label.setMinimumHeight(220)
        self.detail_thumb_label.setStyleSheet("background-color: #060910; border: 1px solid #141c2c; border-radius: 6px; color: #475569;")
        d_layout.addWidget(self.detail_thumb_label)

        # AI Reasoning Box
        self.detail_reasoning_box = QtWidgets.QPlainTextEdit(self.details_card)
        self.detail_reasoning_box.setReadOnly(True)
        self.detail_reasoning_box.setPlaceholderText("AI highlight explanation will appear here.")
        self.detail_reasoning_box.setStyleSheet("background-color: #080c14; color: #f1f5f9; font-size: 13px; border: 1px solid #1c273c; border-radius: 6px;")
        d_layout.addWidget(self.detail_reasoning_box, 1)

        # Action Buttons
        act_layout = QtWidgets.QHBoxLayout()
        self.play_clip_btn = QtWidgets.QPushButton("▶️ Play Video Highlight", self.details_card)
        self.play_clip_btn.setProperty("class", "primary-btn")
        self.play_clip_btn.setMinimumHeight(40)
        self.play_clip_btn.setEnabled(False)
        self.play_clip_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.play_clip_btn.clicked.connect(self._play_selected_clip)

        self.open_explorer_btn = QtWidgets.QPushButton("📁 Show in Explorer", self.details_card)
        self.open_explorer_btn.setProperty("class", "secondary-btn")
        self.open_explorer_btn.setMinimumHeight(40)
        self.open_explorer_btn.setEnabled(False)
        self.open_explorer_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.open_explorer_btn.clicked.connect(self._reveal_selected_clip)

        act_layout.addWidget(self.play_clip_btn, 1)
        act_layout.addWidget(self.open_explorer_btn, 1)
        d_layout.addLayout(act_layout)

        splitter.addWidget(self.details_card)
        splitter.setSizes([420, 600])

        layout.addWidget(splitter, 1)
        return page

    def populate_gallery(self):
        self.clip_list_widget.clear()
        clips_dir = self.config.get("settings", {}).get("clips_dir", "")
        if not clips_dir or not os.path.exists(clips_dir):
            item = QtWidgets.QListWidgetItem("Clips destination folder not configured or empty.")
            item.setFlags(QtCore.Qt.NoItemFlags)
            self.clip_list_widget.addItem(item)
            return

        sort_mode = self.gallery_sort_combo.currentText()
        type_filter = self.gallery_type_combo.currentText()
        score_filter = self.gallery_score_combo.currentText()
        min_score = 0 if score_filter == "All" else int(score_filter.replace("+", ""))

        clip_data = []
        try:
            with os.scandir(clips_dir) as entries:
                for entry in entries:
                    if entry.is_file() and entry.name.endswith(".mp4"):
                        f = entry.name
                        ctime = entry.stat().st_ctime
                        score = 0
                        if "_score" in f:
                            try:
                                score_part = f.rpartition("_score")[2]
                                score_part = score_part.partition("_vertical")[0] if "_vertical" in score_part else score_part.partition(".mp4")[0]
                                score = float(score_part)
                            except ValueError:
                                pass
                        if score == 0:
                            json_path = os.path.join(clips_dir, f.replace("_vertical.mp4", ".mp4").replace(".mp4", ".json"))
                            if os.path.exists(json_path):
                                try:
                                    with open(json_path, 'r', encoding='utf-8') as jf:
                                        jdata = json.load(jf)
                                        score = float(jdata.get("virality_score", 0))
                                except Exception:
                                    pass
                        clip_data.append({"filename": f, "ctime": ctime, "score": score})
        except Exception as e:
            self.log_to_console(f"❌ Gallery Scan Error: {e}", "system")

        if sort_mode == "Date (Newest)":
            clip_data.sort(key=lambda x: x["ctime"], reverse=True)
        elif sort_mode == "Date (Oldest)":
            clip_data.sort(key=lambda x: x["ctime"])
        elif sort_mode == "Virality (High)":
            clip_data.sort(key=lambda x: (x["score"], x["ctime"]), reverse=True)
        elif sort_mode == "Virality (Low)":
            clip_data.sort(key=lambda x: (x["score"], -x["ctime"]))

        count = 0
        for item_data in clip_data:
            f = item_data["filename"]
            score = item_data["score"]

            if type_filter == "Horizontal (16:9)" and "_vertical" in f:
                continue
            if type_filter == "Vertical (9:16)" and "_vertical" not in f:
                continue
            if min_score > 0 and score < min_score:
                continue

            score_badge = f"🔥 {score:.0f}/10" if score > 0 else "⚡ Clip"
            date_str = datetime.datetime.fromtimestamp(item_data["ctime"]).strftime("%b %d, %H:%M")
            display_text = f"[{score_badge}] {f} ({date_str})"
            
            item = QtWidgets.QListWidgetItem(display_text)
            item.setData(QtCore.Qt.UserRole, f)
            self.clip_list_widget.addItem(item)
            count += 1

        if count == 0:
            item = QtWidgets.QListWidgetItem("No clips match the selected filters.")
            item.setFlags(QtCore.Qt.NoItemFlags)
            self.clip_list_widget.addItem(item)

    def _on_clip_selected(self, current: QtWidgets.QListWidgetItem, _previous: QtWidgets.QListWidgetItem = None):
        if not current:
            return

        filename = current.data(QtCore.Qt.UserRole)
        if not filename:
            return

        clips_dir = self.config.get("settings", {}).get("clips_dir", "")
        mp4_path = os.path.join(clips_dir, filename)
        json_path = os.path.join(clips_dir, filename.replace("_vertical.mp4", ".mp4").replace(".mp4", ".json"))
        thumb_path = os.path.join(clips_dir, filename.replace("_vertical.mp4", ".mp4").replace(".mp4", ".jpg"))

        self.detail_title.setText(filename)

        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    score = data.get("virality_score", "N/A")
                    reasoning = data.get("reasoning", "No explanation available.")
                    self.detail_score.setText(f"Virality Score: {score}/10")
                    self.detail_reasoning_box.setPlainText(reasoning)
            except Exception:
                self.detail_score.setText("Virality Score: N/A")
                self.detail_reasoning_box.setPlainText("Error reading metadata JSON.")
        else:
            self.detail_score.setText("Virality Score: N/A")
            self.detail_reasoning_box.setPlainText("No metadata sidecar found.")

        if os.path.exists(thumb_path):
            pixmap = QtGui.QPixmap(thumb_path)
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(520, 240, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
                self.detail_thumb_label.setPixmap(scaled_pixmap)
                self.detail_thumb_label.setText("")
            else:
                self.detail_thumb_label.setPixmap(QtGui.QPixmap())
                self.detail_thumb_label.setText("Invalid thumbnail file")
        else:
            self.detail_thumb_label.setPixmap(QtGui.QPixmap())
            self.detail_thumb_label.setText("No thumbnail generated")

        self.selected_mp4_path = mp4_path
        self.play_clip_btn.setEnabled(True)
        self.open_explorer_btn.setEnabled(True)

    def _play_selected_clip(self):
        if hasattr(self, 'selected_mp4_path') and self.selected_mp4_path:
            self._open_video_file(self.selected_mp4_path)

    def _reveal_selected_clip(self):
        if hasattr(self, 'selected_mp4_path') and self.selected_mp4_path:
            self._reveal_in_explorer(self.selected_mp4_path)

    def _open_video_file(self, file_path: str):
        if os.path.exists(file_path):
            if hasattr(os, 'startfile'):
                os.startfile(os.path.abspath(file_path))
            else:
                QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(os.path.abspath(file_path)))

    def _reveal_in_explorer(self, file_path: str):
        if os.path.exists(file_path):
            subprocess.run(['explorer', '/select,', os.path.abspath(file_path)], timeout=5)

    def _delete_all_clips(self):
        clips_dir = self.config.get("settings", {}).get("clips_dir", "")
        if not clips_dir or not os.path.exists(clips_dir):
            return

        reply = QtWidgets.QMessageBox.question(
            self, "Confirm Batch Delete",
            "Are you sure you want to permanently delete ALL clips, thumbnails, and JSON metadata in the destination folder?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        if reply == QtWidgets.QMessageBox.Yes:
            count = 0
            for f in os.listdir(clips_dir):
                if f.endswith((".mp4", ".json", ".jpg")):
                    try:
                        os.remove(os.path.join(clips_dir, f))
                        count += 1
                    except Exception:
                        pass
            self.log_to_console(f"🗑️ Deleted {count} file(s) from clips directory.", "system")
            self.populate_gallery()

    # ==============================================================================
    # SETTINGS & API HELPERS
    # ==============================================================================
    def _browse_clips_dir(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Clips Destination Directory")
        if folder:
            self.clips_dir_edit.setText(folder)

    def _save_settings(self):
        active_btn = self.provider_btn_group.checkedButton()
        if active_btn:
            self.config["active_ai_provider"] = active_btn.property("prov_id")

        self.config.setdefault("openai", {})["api_key"] = self.openai_key_edit.text().strip()
        self.config["openai_model"] = self.openai_model_combo.currentText()
        if self.openai_base_url_edit:
            self.config["openai"]["base_url"] = self.openai_base_url_edit.text().strip()

        self.config.setdefault("deepseek", {})["api_key"] = self.deepseek_key_edit.text().strip()
        self.config["deepseek_model"] = self.deepseek_model_combo.currentText()

        self.config.setdefault("anthropic", {})["api_key"] = self.anthropic_key_edit.text().strip()
        self.config["anthropic_model"] = self.anthropic_model_combo.currentText()

        self.config.setdefault("xai", {})["api_key"] = self.xai_key_edit.text().strip()
        self.config["xai_model"] = self.xai_model_combo.currentText()

        self.config.setdefault("google", {})["api_key"] = self.google_key_edit.text().strip()
        self.config["google_model"] = self.google_model_combo.currentText()

        self.config.setdefault("openai", {})["whisper_model"] = self.whisper_model_combo.currentText()
        self.config.setdefault("openai", {})["whisper_language"] = self.whisper_lang_combo.currentText()

        settings = self.config.setdefault("settings", {})
        settings["audio_downmix"] = self.downmix_checkbox.isChecked()
        settings["audio_peak_detection"] = self.peak_checkbox.isChecked()
        settings["combat_detection"] = self.combat_checkbox.isChecked()
        settings["clips_per_hour"] = self.clips_per_hour_spin.value()
        settings["hardware_encoding"] = self.hw_encode_checkbox.isChecked()
        settings["vr_stabilization"] = self.vr_checkbox.isChecked()
        settings["vertical_export"] = self.vertical_checkbox.isChecked()
        settings["vertical_mode"] = self.vertical_mode_combo.currentText()
        settings["clips_dir"] = self.clips_dir_edit.text().strip()

        self.config.setdefault("integrations", {})["discord_webhook"] = self.discord_edit.text().strip()

        config_manager.save_config(self.config)
        self.log_to_console("💾 Configuration saved successfully.", "system")
        QtWidgets.QMessageBox.information(self, "Settings Saved", "Configuration saved successfully!")

    def _test_openai_key(self):
        key = self.openai_key_edit.text().strip()
        base_url = self.openai_base_url_edit.text().strip() if self.openai_base_url_edit else ""
        self._test_api_key_async("openai", key, base_url)

    def _test_deepseek_key(self):
        key = self.deepseek_key_edit.text().strip()
        self._test_api_key_async("deepseek", key)

    def _test_anthropic_key(self):
        key = self.anthropic_key_edit.text().strip()
        self._test_api_key_async("anthropic", key)

    def _test_xai_key(self):
        key = self.xai_key_edit.text().strip()
        self._test_api_key_async("xai", key)

    def _test_google_key(self):
        key = self.google_key_edit.text().strip()
        self._test_api_key_async("google", key)

    def _test_api_key_async(self, provider: str, api_key: str, base_url: str = ""):
        if not api_key:
            QtWidgets.QMessageBox.warning(self, "API Key Missing", f"Please enter a valid {provider.capitalize()} API key.")
            return

        self.log_to_console(f"🔍 Testing {provider.capitalize()} API credentials...", "system")

        def task():
            if provider == "openai":
                models = model_fetcher.fetch_openai_models(api_key, base_url=base_url if base_url else None)
            elif provider == "deepseek":
                models = model_fetcher.fetch_deepseek_models(api_key)
            elif provider == "anthropic":
                models = model_fetcher.fetch_anthropic_models(api_key)
            elif provider == "xai":
                models = model_fetcher.fetch_xai_models(api_key)
            elif provider == "google":
                models = model_fetcher.fetch_google_models(api_key)
            else:
                models = []

            if models:
                self.signals.model_list_updated.emit(provider, models)
                self.signals.log_message.emit(f"✅ {provider.capitalize()} Connection Successful! Fetched {len(models)} models.", "system")
            else:
                self.signals.log_message.emit(f"❌ {provider.capitalize()} Connection Failed! Check API Key or endpoint.", "system")

        threading.Thread(target=task, daemon=True).start()

    def _on_model_list_updated(self, provider: str, models: list):
        combos = {
            "openai": self.openai_model_combo,
            "deepseek": self.deepseek_model_combo,
            "anthropic": self.anthropic_model_combo,
            "xai": self.xai_model_combo,
            "google": self.google_model_combo,
        }
        combo = combos.get(provider)
        if combo and models:
            curr = combo.currentText()
            combo.clear()
            combo.addItems(models)
            if curr in models:
                combo.setCurrentText(curr)
            QtWidgets.QMessageBox.information(self, "Connection Verified", f"Successfully connected to {provider.capitalize()}!\nAvailable models updated.")

    def _test_discord_webhook(self):
        url = self.discord_edit.text().strip()
        if not url:
            QtWidgets.QMessageBox.warning(self, "Missing URL", "Please enter a Discord Webhook URL.")
            return
        self._send_discord_alert("🧪 Test Webhook Notification from Clip Generator")
        QtWidgets.QMessageBox.information(self, "Discord Alert", "Test notification sent! Check your Discord channel.")

    def _send_discord_alert(self, message: str):
        webhook_url = self.config.get("integrations", {}).get("discord_webhook", "").strip()
        if not webhook_url:
            return

        def send():
            try:
                payload = json.dumps({"content": f"🎬 **jBahr Clip Generator**: {message}"}).encode("utf-8")
                req = urllib.request.Request(
                    webhook_url,
                    data=payload,
                    headers={"Content-Type": "application/json", "User-Agent": "jBahrClipGen/2.2"}
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if resp.status not in (200, 204):
                        self.signals.log_message.emit(f"⚠️ Discord Webhook response status: {resp.status}", "system")
            except Exception as e:
                self.signals.log_message.emit(f"⚠️ Discord Webhook Failed: {e}", "system")

        threading.Thread(target=send, daemon=True).start()


def run_app():
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = ClipGenPySideApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_app()
