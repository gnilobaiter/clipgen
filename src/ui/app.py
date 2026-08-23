import datetime
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import subprocess
import sys
import threading
from tkinter import messagebox
import urllib.parse
import urllib.request
import webbrowser

if sys.stdout is None:
    sys.stdout = open(os.devnull, 'w')
if sys.stderr is None:
    sys.stderr = open(os.devnull, 'w')

import customtkinter as ctk
from customtkinter import filedialog
from PIL import Image
import pystray

from src.core import config as config_manager
from src.core import editor
from src.services import model_fetcher
from src.utils.hardware import log_hardware_info
from src.utils.paths import get_app_data_path, inject_bin_to_path

inject_bin_to_path()

# ==============================================================================
# MODERN DESIGN TOKENS & COLOR PALETTE (Deep Slate / Obsidian)
# ==============================================================================
THEME = {
    "bg_window": "#0a0c10",          # Main deep canvas background
    "bg_sidebar": "#101318",         # Distinct dark sidebar
    "bg_card": "#141820",            # Elevated card surface
    "bg_card_alt": "#1a202c",        # Secondary / nested card surface
    "bg_input": "#0d1015",           # Sunken input background
    "bg_console": "#080a0d",         # Terminal console background
    
    "border_subtle": "#1f2633",      # Card and container borders
    "border_input": "#252e3d",       # Input borders
    "border_active": "#3b82f6",      # Active focused border
    
    "accent_primary": "#2563eb",     # Electric Blue primary CTA
    "accent_primary_hover": "#1d4ed8",
    "accent_success": "#059669",     # Emerald Green
    "accent_success_hover": "#047857",
    "accent_danger": "#e11d48",      # Crimson / Rose Red
    "accent_danger_hover": "#be123c",
    "accent_purple": "#7c3aed",      # Violet accent
    "accent_purple_hover": "#6d28d9",
    "accent_amber": "#d97706",       # Warning amber
    "accent_amber_hover": "#b45309",
    
    "text_primary": "#f1f5f9",       # High-contrast crisp white/slate
    "text_secondary": "#94a3b8",     # Readable muted slate
    "text_muted": "#64748b",         # Subtle tertiary text
    "text_dim": "#475569",           # Very faint labels
}

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class ClipGenApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Clip Generator — AI Highlight Workstation")
        self.geometry("1180x880")
        self.minsize(1000, 680)
        self.configure(fg_color=THEME["bg_window"])
        
        self.protocol('WM_DELETE_WINDOW', self.minimize_to_tray)
        self.tray_icon = None
        
        self.config = config_manager.load_config()
        self.metadata_cache = {}
        self.cancel_requested = False

        self._init_logging()
        
        # Grid layout for responsive fluid window resizing
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        
        self._setup_sidebar()
        self._setup_manual_frame()
        self._setup_prompt_frame()
        self._setup_settings_frame()
        self._setup_gallery_frame()
        
        self.load_prompt_data()
        self.show_manual_frame()
        
        # Hardware diagnostics & background model list population
        log_hardware_info(lambda msg: self.log_to_console(msg, source="system"))
        threading.Thread(target=self.refresh_model_list, daemon=True).start()

    def _setup_sidebar(self):
        self.sidebar_frame = ctk.CTkFrame(
            self, width=230, corner_radius=0, 
            fg_color=THEME["bg_sidebar"], 
            border_width=1, border_color=THEME["border_subtle"]
        )
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_propagate(False)
        self.sidebar_frame.grid_rowconfigure(11, weight=1)

        # Brand / Logo Header
        self.brand_container = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.brand_container.grid(row=0, column=0, padx=18, pady=(24, 20), sticky="ew")
        
        self.logo_label = ctk.CTkLabel(
            self.brand_container, text="🎬 CLIPGEN", 
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=THEME["text_primary"]
        )
        self.logo_label.pack(anchor="w")
        
        self.sub_logo_label = ctk.CTkLabel(
            self.brand_container, text="AI Highlight Studio", 
            font=ctk.CTkFont(size=11), text_color=THEME["accent_primary"]
        )
        self.sub_logo_label.pack(anchor="w", pady=(1, 0))

        # Main Navigation Items
        self.nav_manual_btn = ctk.CTkButton(
            self.sidebar_frame, cursor="hand2", text="  ⚡ Clip Extractor", 
            anchor="w", height=42, corner_radius=8, font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="transparent", hover_color=THEME["bg_card_alt"], 
            text_color=THEME["text_primary"], command=self.show_manual_frame
        )
        self.nav_manual_btn.grid(row=1, column=0, padx=14, pady=4, sticky="ew")

        self.nav_prompt_btn = ctk.CTkButton(
            self.sidebar_frame, cursor="hand2", text="  📝 Prompt Editor", 
            anchor="w", height=42, corner_radius=8, font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="transparent", hover_color=THEME["bg_card_alt"], 
            text_color=THEME["text_primary"], command=self.show_prompt_frame
        )
        self.nav_prompt_btn.grid(row=2, column=0, padx=14, pady=4, sticky="ew")

        self.nav_settings_btn = ctk.CTkButton(
            self.sidebar_frame, cursor="hand2", text="  ⚙️ Configuration", 
            anchor="w", height=42, corner_radius=8, font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="transparent", hover_color=THEME["bg_card_alt"], 
            text_color=THEME["text_primary"], command=self.show_settings_frame
        )
        self.nav_settings_btn.grid(row=3, column=0, padx=14, pady=4, sticky="ew")

        self.nav_gallery_btn = ctk.CTkButton(
            self.sidebar_frame, cursor="hand2", text="  🖼️ Clip Gallery", 
            anchor="w", height=42, corner_radius=8, font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="transparent", hover_color=THEME["bg_card_alt"], 
            text_color=THEME["text_primary"], command=self.show_gallery_frame
        )
        self.nav_gallery_btn.grid(row=4, column=0, padx=14, pady=4, sticky="ew")

        # Divider
        self.sidebar_div = ctk.CTkFrame(self.sidebar_frame, height=1, fg_color=THEME["border_subtle"])
        self.sidebar_div.grid(row=5, column=0, padx=18, pady=16, sticky="ew")

        # Shortcuts Section
        self.quick_access_label = ctk.CTkLabel(
            self.sidebar_frame, text="QUICK SHORTCUTS", 
            font=ctk.CTkFont(size=10, weight="bold"), text_color=THEME["text_muted"]
        )
        self.quick_access_label.grid(row=6, column=0, padx=18, pady=(0, 6), sticky="w")

        self.open_clips_btn = ctk.CTkButton(
            self.sidebar_frame, cursor="hand2", text="📁 Output Folder", 
            anchor="w", height=34, corner_radius=6, font=ctk.CTkFont(size=12),
            fg_color=THEME["bg_card"], hover_color=THEME["bg_card_alt"], 
            border_width=1, border_color=THEME["border_subtle"],
            text_color=THEME["text_secondary"], command=lambda: self.open_local_folder("clips_dir", self.open_clips_btn)
        )
        self.open_clips_btn.grid(row=7, column=0, padx=14, pady=3, sticky="ew")

        self.open_logs_btn = ctk.CTkButton(
            self.sidebar_frame, cursor="hand2", text="📜 Crash Logs", 
            anchor="w", height=34, corner_radius=6, font=ctk.CTkFont(size=12),
            fg_color=THEME["bg_card"], hover_color=THEME["bg_card_alt"], 
            border_width=1, border_color=THEME["border_subtle"],
            text_color=THEME["text_secondary"], command=self.open_logs
        )
        self.open_logs_btn.grid(row=8, column=0, padx=14, pady=3, sticky="ew")

        self.open_readme_btn = ctk.CTkButton(
            self.sidebar_frame, cursor="hand2", text="📖 Documentation", 
            anchor="w", height=34, corner_radius=6, font=ctk.CTkFont(size=12),
            fg_color=THEME["bg_card"], hover_color=THEME["bg_card_alt"], 
            border_width=1, border_color=THEME["border_subtle"],
            text_color=THEME["text_secondary"], command=lambda: self.open_readme(self.open_readme_btn)
        )
        self.open_readme_btn.grid(row=9, column=0, padx=14, pady=3, sticky="ew")

        # Community Links & Version Footer
        self.footer_container = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.footer_container.grid(row=11, column=0, padx=14, pady=16, sticky="sew")

        self.discord_btn = ctk.CTkButton(
            self.footer_container, cursor="hand2", text="💬 Join Discord", 
            height=34, corner_radius=6, font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#5865F2", hover_color="#4752C4", text_color="#FFFFFF",
            command=lambda: webbrowser.open("https://discord.gg/uUF8J9Zqwz")
        )
        self.discord_btn.pack(fill="x", pady=(0, 6))

        self.github_btn = ctk.CTkButton(
            self.footer_container, cursor="hand2", text="🌐 GitHub Repo", 
            height=32, corner_radius=6, font=ctk.CTkFont(size=11),
            fg_color=THEME["bg_card"], hover_color=THEME["bg_card_alt"],
            border_width=1, border_color=THEME["border_subtle"],
            text_color=THEME["text_secondary"], command=lambda: webbrowser.open("https://github.com/jBahrVR/jBahrs-Clip-Generator")
        )
        self.github_btn.pack(fill="x", pady=(0, 10))

        self.version_badge = ctk.CTkLabel(
            self.footer_container, text="v1.2.1 Pro Edition", 
            font=ctk.CTkFont(size=10), text_color=THEME["text_dim"]
        )
        self.version_badge.pack(anchor="center")

    def _setup_manual_frame(self):
        self.manual_frame = ctk.CTkFrame(self, fg_color=THEME["bg_window"])
        self.manual_frame.grid_columnconfigure(0, weight=1)
        self.manual_frame.grid_rowconfigure(4, weight=1)
        
        # Header
        self.manual_header = ctk.CTkFrame(self.manual_frame, fg_color="transparent")
        self.manual_header.grid(row=0, column=0, padx=32, pady=(28, 12), sticky="ew")
        
        self.manual_title = ctk.CTkLabel(
            self.manual_header, text="Video Highlight Extractor", 
            font=ctk.CTkFont(size=26, weight="bold"), text_color=THEME["text_primary"]
        )
        self.manual_title.pack(anchor="w")
        
        self.manual_subtitle = ctk.CTkLabel(
            self.manual_header, text="Select local gameplay recordings, VODs, or OBS clips for AI highlight detection & rendering.", 
            font=ctk.CTkFont(size=13), text_color=THEME["text_secondary"]
        )
        self.manual_subtitle.pack(anchor="w", pady=(3, 0))

        # Input Card Container
        self.input_card = ctk.CTkFrame(
            self.manual_frame, corner_radius=12, 
            fg_color=THEME["bg_card"], border_width=1, border_color=THEME["border_subtle"]
        )
        self.input_card.grid(row=1, column=0, padx=32, pady=(0, 14), sticky="ew")
        self.input_card.grid_columnconfigure(0, weight=1)

        self.file_input = ctk.CTkEntry(
            self.input_card, placeholder_text="Select or enter local video file path(s)...", 
            height=44, corner_radius=8, font=ctk.CTkFont(size=13),
            fg_color=THEME["bg_input"], border_width=1, border_color=THEME["border_input"],
            text_color=THEME["text_primary"]
        )
        self.file_input.grid(row=0, column=0, padx=16, pady=16, sticky="ew")
        self.file_input.bind("<Return>", self.start_manual_process)

        self.local_file_btn = ctk.CTkButton(
            self.input_card, cursor="hand2", text="📂 Browse", 
            height=44, width=105, corner_radius=8, font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=THEME["bg_card_alt"], hover_color=THEME["border_subtle"],
            border_width=1, border_color=THEME["border_input"],
            text_color=THEME["text_primary"], command=self.browse_local_file
        )
        self.local_file_btn.grid(row=0, column=1, padx=(0, 8), pady=16)

        self.process_btn = ctk.CTkButton(
            self.input_card, cursor="hand2", text="⚡ Process Files", 
            height=44, width=130, corner_radius=8, font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=THEME["accent_primary"], hover_color=THEME["accent_primary_hover"],
            text_color="#FFFFFF", command=self.start_manual_process
        )
        self.process_btn.grid(row=0, column=2, padx=(0, 8), pady=16)

        self.cancel_btn = ctk.CTkButton(
            self.input_card, cursor="hand2", text="🛑 Cancel", 
            height=44, width=90, corner_radius=8, font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=THEME["accent_danger"], hover_color=THEME["accent_danger_hover"],
            text_color="#FFFFFF", state="disabled", command=self.cancel_manual_process
        )
        self.cancel_btn.grid(row=0, column=3, padx=(0, 16), pady=16)

        # Status & Progress Row
        self.status_row = ctk.CTkFrame(self.manual_frame, fg_color="transparent")
        self.status_row.grid(row=2, column=0, padx=32, pady=(0, 6), sticky="ew")
        
        self.manual_status_label = ctk.CTkLabel(
            self.status_row, text="● Status: Ready", 
            font=ctk.CTkFont(size=13, weight="bold"), text_color=THEME["accent_success"]
        )
        self.manual_status_label.pack(side="left")

        self.manual_progress = ctk.CTkProgressBar(
            self.manual_frame, mode="indeterminate", height=6, corner_radius=3,
            fg_color=THEME["bg_card"], progress_color=THEME["accent_primary"]
        )
        self.manual_progress.grid(row=3, column=0, padx=32, pady=(0, 14), sticky="ew")
        self.manual_progress.set(0)

        # Live Console Card
        self.console_card = ctk.CTkFrame(
            self.manual_frame, corner_radius=12, 
            fg_color=THEME["bg_card"], border_width=1, border_color=THEME["border_subtle"]
        )
        self.console_card.grid(row=4, column=0, padx=32, pady=(0, 24), sticky="nsew")
        self.console_card.grid_columnconfigure(0, weight=1)
        self.console_card.grid_rowconfigure(1, weight=1)

        # Terminal Header Bar
        self.console_header = ctk.CTkFrame(self.console_card, height=36, fg_color="transparent")
        self.console_header.grid(row=0, column=0, padx=16, pady=(12, 6), sticky="ew")
        
        self.terminal_badge = ctk.CTkLabel(
            self.console_header, text="💻 Execution Stream Logs", 
            font=ctk.CTkFont(size=12, weight="bold"), text_color=THEME["text_secondary"]
        )
        self.terminal_badge.pack(side="left")

        self.clear_console_btn = ctk.CTkButton(
            self.console_header, cursor="hand2", text="Clear Output", 
            height=26, width=85, corner_radius=5, font=ctk.CTkFont(size=11),
            fg_color=THEME["bg_card_alt"], hover_color=THEME["border_subtle"],
            text_color=THEME["text_secondary"], command=self.clear_console
        )
        self.clear_console_btn.pack(side="right")

        self.console_box = ctk.CTkTextbox(
            self.console_card, state="disabled", 
            fg_color=THEME["bg_console"], corner_radius=8,
            font=ctk.CTkFont(family="Consolas", size=12), text_color=THEME["text_primary"]
        )
        self.console_box.grid(row=1, column=0, padx=16, pady=(0, 16), sticky="nsew")
        
        self.console_box.tag_config("error", foreground="#f87171")
        self.console_box.tag_config("success", foreground="#34d399")
        self.console_box.tag_config("ai", foreground="#38bdf8")
        self.console_box.tag_config("ffmpeg", foreground="#fbbf24")

    def _setup_prompt_frame(self):
        self.prompt_frame = ctk.CTkFrame(self, fg_color=THEME["bg_window"])
        self.prompt_frame.grid_columnconfigure(0, weight=1)
        self.prompt_frame.grid_rowconfigure(2, weight=1)
        
        # Header
        self.prompt_header = ctk.CTkFrame(self.prompt_frame, fg_color="transparent")
        self.prompt_header.grid(row=0, column=0, padx=32, pady=(28, 12), sticky="ew")
        
        self.prompt_title = ctk.CTkLabel(
            self.prompt_header, text="AI Prompt & Strategy Manager", 
            font=ctk.CTkFont(size=26, weight="bold"), text_color=THEME["text_primary"]
        )
        self.prompt_title.pack(anchor="w")

        self.prompt_subtitle = ctk.CTkLabel(
            self.prompt_header, text="Fine-tune highlight extraction rules, virality thresholds, and comedic timing profiles.", 
            font=ctk.CTkFont(size=13), text_color=THEME["text_secondary"]
        )
        self.prompt_subtitle.pack(anchor="w", pady=(3, 0))

        # Top Profile Selector Bar
        self.prompt_select_card = ctk.CTkFrame(
            self.prompt_frame, corner_radius=12, 
            fg_color=THEME["bg_card"], border_width=1, border_color=THEME["border_subtle"]
        )
        self.prompt_select_card.grid(row=1, column=0, padx=32, pady=(0, 14), sticky="ew")
        
        ctk.CTkLabel(
            self.prompt_select_card, text="Active Profile:", 
            font=ctk.CTkFont(size=13, weight="bold"), text_color=THEME["text_primary"]
        ).pack(side="left", padx=(16, 8), pady=14)

        self.profile_dropdown = ctk.CTkOptionMenu(
            self.prompt_select_card, cursor="hand2", height=36, width=220, corner_radius=8,
            fg_color=THEME["bg_card_alt"], button_color=THEME["border_input"],
            text_color=THEME["text_primary"], command=self.on_profile_change
        )
        self.profile_dropdown.pack(side="left", padx=8)

        self.new_profile_btn = ctk.CTkButton(
            self.prompt_select_card, cursor="hand2", text="➕ New Profile", 
            height=36, corner_radius=8, font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=THEME["accent_success"], hover_color=THEME["accent_success_hover"],
            text_color="#FFFFFF", command=self.create_new_profile
        )
        self.new_profile_btn.pack(side="left", padx=6)

        self.delete_profile_btn = ctk.CTkButton(
            self.prompt_select_card, cursor="hand2", text="🗑️ Delete", 
            height=36, width=90, corner_radius=8, font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=THEME["accent_danger"], hover_color=THEME["accent_danger_hover"],
            text_color="#FFFFFF", command=self.delete_profile
        )
        self.delete_profile_btn.pack(side="right", padx=16)

        # Editor Area
        self.prompt_editor_card = ctk.CTkFrame(
            self.prompt_frame, corner_radius=12, 
            fg_color=THEME["bg_card"], border_width=1, border_color=THEME["border_subtle"]
        )
        self.prompt_editor_card.grid(row=2, column=0, padx=32, pady=(0, 24), sticky="nsew")
        self.prompt_editor_card.grid_columnconfigure(0, weight=1)
        self.prompt_editor_card.grid_rowconfigure(0, weight=1)

        self.prompt_textbox = ctk.CTkTextbox(
            self.prompt_editor_card, font=ctk.CTkFont(size=13), wrap="word",
            fg_color=THEME["bg_input"], corner_radius=8, text_color=THEME["text_primary"]
        )
        self.prompt_textbox.grid(row=0, column=0, padx=16, pady=16, sticky="nsew")

        self.prompt_footer_bar = ctk.CTkFrame(self.prompt_editor_card, fg_color="transparent")
        self.prompt_footer_bar.grid(row=1, column=0, padx=16, pady=(0, 16), sticky="ew")

        self.prompt_hint = ctk.CTkLabel(
            self.prompt_footer_bar, text="💡 Tip: Output format must remain strictly JSON with start_time, end_time, and virality_score fields.",
            font=ctk.CTkFont(size=11), text_color=THEME["text_muted"]
        )
        self.prompt_hint.pack(side="left")

        self.save_prompt_btn = ctk.CTkButton(
            self.prompt_footer_bar, cursor="hand2", text="💾 Save Prompt", 
            height=40, width=140, corner_radius=8, font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=THEME["accent_primary"], hover_color=THEME["accent_primary_hover"],
            text_color="#FFFFFF", command=self.save_current_prompt
        )
        self.save_prompt_btn.pack(side="right")

    def _setup_settings_frame(self):
        self.settings_frame = ctk.CTkScrollableFrame(
            self, fg_color=THEME["bg_window"],
            scrollbar_button_color=THEME["border_subtle"],
            scrollbar_button_hover_color=THEME["border_input"]
        )
        self.settings_frame.grid_columnconfigure(0, weight=1)
        
        # Header
        self.settings_header = ctk.CTkFrame(self.settings_frame, fg_color="transparent")
        self.settings_header.grid(row=0, column=0, padx=32, pady=(28, 12), sticky="ew")
        
        self.settings_title = ctk.CTkLabel(
            self.settings_header, text="Application Configuration", 
            font=ctk.CTkFont(size=26, weight="bold"), text_color=THEME["text_primary"]
        )
        self.settings_title.pack(anchor="w")

        self.settings_subtitle = ctk.CTkLabel(
            self.settings_header, text="Configure AI cloud providers, local Whisper models, and hardware export encoders.", 
            font=ctk.CTkFont(size=13), text_color=THEME["text_secondary"]
        )
        self.settings_subtitle.pack(anchor="w", pady=(3, 0))

        # --- Card 1: AI Engines & API Keys ---
        self.api_card = ctk.CTkFrame(
            self.settings_frame, corner_radius=12, 
            fg_color=THEME["bg_card"], border_width=1, border_color=THEME["border_subtle"]
        )
        self.api_card.grid(row=1, column=0, padx=32, pady=(0, 16), sticky="ew")
        self.api_card.grid_columnconfigure(0, minsize=220)
        self.api_card.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            self.api_card, text="AI Cloud Engines & Credentials", 
            font=ctk.CTkFont(size=15, weight="bold"), text_color=THEME["text_primary"]
        ).grid(row=0, column=0, columnspan=3, padx=20, pady=(18, 12), sticky="w")

        self.active_provider_var = ctk.StringVar(value=self.config.get("active_ai_provider", "openai"))
        
        # 1. OpenAI
        self.openai_frame = ctk.CTkFrame(self.api_card, fg_color=THEME["bg_card_alt"], corner_radius=8)
        self.openai_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=6, padx=16)
        self.openai_frame.grid_columnconfigure(0, minsize=200)
        self.openai_frame.grid_columnconfigure(1, weight=1)

        self.openai_radio = ctk.CTkRadioButton(
            self.openai_frame, cursor="hand2", text="OpenAI / Custom Endpoint:", 
            font=ctk.CTkFont(size=13, weight="bold"), text_color=THEME["text_primary"],
            variable=self.active_provider_var, value="openai"
        )
        self.openai_radio.grid(row=0, column=0, padx=16, pady=(12, 6), sticky="w")
        
        self.openai_entry = ctk.CTkEntry(
            self.openai_frame, show="•", height=34, placeholder_text="sk-proj-...",
            fg_color=THEME["bg_input"], border_color=THEME["border_input"], text_color=THEME["text_primary"]
        )
        self.openai_entry.grid(row=0, column=1, padx=(0, 10), pady=(12, 6), sticky="ew")
        self.openai_entry.insert(0, self.config.get('openai', {}).get('api_key', ''))
        self.openai_entry.bind("<KeyRelease>", lambda e: self._update_model_visibility())
        
        self.test_openai_btn = ctk.CTkButton(
            self.openai_frame, cursor="hand2", text="Test Key", width=85, height=34,
            corner_radius=6, font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=THEME["accent_primary"], hover_color=THEME["accent_primary_hover"],
            command=self.test_openai_key
        )
        self.test_openai_btn.grid(row=0, column=2, padx=(0, 16), pady=(12, 6), sticky="e")

        self.openai_model_label = ctk.CTkLabel(self.openai_frame, text="AI Model:", font=ctk.CTkFont(size=12, weight="bold"), text_color=THEME["text_secondary"])
        self.openai_model_menu = ctk.CTkComboBox(self.openai_frame, cursor="hand2", values=model_fetcher.MAIN_MODELS["openai"], height=32, width=260)
        self.openai_model_menu.set(self.config.get("openai_model", "gpt-4o"))
        
        self.base_url_label = ctk.CTkLabel(self.openai_frame, text="Base URL:", font=ctk.CTkFont(size=11), text_color=THEME["text_muted"])
        self.base_url_label.grid(row=2, column=0, padx=(36, 10), pady=(4, 12), sticky="w")
        self.base_url_entry = ctk.CTkEntry(self.openai_frame, height=30, placeholder_text="https://openrouter.ai/api/v1 (Optional)", fg_color=THEME["bg_input"], border_color=THEME["border_input"], text_color=THEME["text_primary"])
        self.base_url_entry.grid(row=2, column=1, columnspan=2, padx=(0, 16), pady=(4, 12), sticky="ew")
        self.base_url_entry.insert(0, self.config.get('openai', {}).get('base_url', ''))

        # 2. DeepSeek
        self.deepseek_frame = ctk.CTkFrame(self.api_card, fg_color=THEME["bg_card_alt"], corner_radius=8)
        self.deepseek_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=6, padx=16)
        self.deepseek_frame.grid_columnconfigure(0, minsize=200)
        self.deepseek_frame.grid_columnconfigure(1, weight=1)

        self.deepseek_radio = ctk.CTkRadioButton(
            self.deepseek_frame, cursor="hand2", text="DeepSeek Engine:", 
            font=ctk.CTkFont(size=13, weight="bold"), text_color=THEME["text_primary"],
            variable=self.active_provider_var, value="deepseek"
        )
        self.deepseek_radio.grid(row=0, column=0, padx=16, pady=12, sticky="w")
        
        self.deepseek_entry = ctk.CTkEntry(
            self.deepseek_frame, show="•", height=34, placeholder_text="sk-...",
            fg_color=THEME["bg_input"], border_color=THEME["border_input"], text_color=THEME["text_primary"]
        )
        self.deepseek_entry.grid(row=0, column=1, padx=(0, 10), pady=12, sticky="ew")
        self.deepseek_entry.insert(0, self.config.get('deepseek', {}).get('api_key', ''))
        self.deepseek_entry.bind("<KeyRelease>", lambda e: self._update_model_visibility())
        
        self.test_deepseek_btn = ctk.CTkButton(
            self.deepseek_frame, cursor="hand2", text="Test Key", width=85, height=34,
            corner_radius=6, font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=THEME["accent_primary"], hover_color=THEME["accent_primary_hover"],
            command=self.test_deepseek_key
        )
        self.test_deepseek_btn.grid(row=0, column=2, padx=(0, 16), pady=12, sticky="e")

        self.deepseek_model_label = ctk.CTkLabel(self.deepseek_frame, text="AI Model:", font=ctk.CTkFont(size=12, weight="bold"), text_color=THEME["text_secondary"])
        self.deepseek_model_menu = ctk.CTkComboBox(self.deepseek_frame, cursor="hand2", values=model_fetcher.MAIN_MODELS["deepseek"], height=32, width=260)
        self.deepseek_model_menu.set(self.config.get("deepseek_model", "deepseek-v4-flash"))

        # 3. Anthropic
        self.anthropic_frame = ctk.CTkFrame(self.api_card, fg_color=THEME["bg_card_alt"], corner_radius=8)
        self.anthropic_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=6, padx=16)
        self.anthropic_frame.grid_columnconfigure(0, minsize=200)
        self.anthropic_frame.grid_columnconfigure(1, weight=1)

        self.anthropic_radio = ctk.CTkRadioButton(
            self.anthropic_frame, cursor="hand2", text="Anthropic Claude:", 
            font=ctk.CTkFont(size=13, weight="bold"), text_color=THEME["text_primary"],
            variable=self.active_provider_var, value="anthropic"
        )
        self.anthropic_radio.grid(row=0, column=0, padx=16, pady=12, sticky="w")
        
        self.anthropic_entry = ctk.CTkEntry(
            self.anthropic_frame, show="•", height=34, placeholder_text="sk-ant-...",
            fg_color=THEME["bg_input"], border_color=THEME["border_input"], text_color=THEME["text_primary"]
        )
        self.anthropic_entry.grid(row=0, column=1, padx=(0, 10), pady=12, sticky="ew")
        self.anthropic_entry.insert(0, self.config.get('anthropic', {}).get('api_key', ''))
        self.anthropic_entry.bind("<KeyRelease>", lambda e: self._update_model_visibility())
        
        self.test_anthropic_btn = ctk.CTkButton(
            self.anthropic_frame, cursor="hand2", text="Test Key", width=85, height=34,
            corner_radius=6, font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=THEME["accent_primary"], hover_color=THEME["accent_primary_hover"],
            command=self.test_anthropic_key
        )
        self.test_anthropic_btn.grid(row=0, column=2, padx=(0, 16), pady=12, sticky="e")

        self.anthropic_model_label = ctk.CTkLabel(self.anthropic_frame, text="AI Model:", font=ctk.CTkFont(size=12, weight="bold"), text_color=THEME["text_secondary"])
        self.anthropic_model_menu = ctk.CTkComboBox(self.anthropic_frame, cursor="hand2", values=model_fetcher.MAIN_MODELS["anthropic"], height=32, width=260)
        self.anthropic_model_menu.set(self.config.get("anthropic_model", "claude-3-5-sonnet-latest"))

        # 4. xAI Grok
        self.xai_frame = ctk.CTkFrame(self.api_card, fg_color=THEME["bg_card_alt"], corner_radius=8)
        self.xai_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=6, padx=16)
        self.xai_frame.grid_columnconfigure(0, minsize=200)
        self.xai_frame.grid_columnconfigure(1, weight=1)

        self.grok_radio = ctk.CTkRadioButton(
            self.xai_frame, cursor="hand2", text="xAI / Grok:", 
            font=ctk.CTkFont(size=13, weight="bold"), text_color=THEME["text_primary"],
            variable=self.active_provider_var, value="xai"
        )
        self.grok_radio.grid(row=0, column=0, padx=16, pady=12, sticky="w")
        
        self.grok_entry = ctk.CTkEntry(
            self.xai_frame, show="•", height=34, placeholder_text="xai-...",
            fg_color=THEME["bg_input"], border_color=THEME["border_input"], text_color=THEME["text_primary"]
        )
        self.grok_entry.grid(row=0, column=1, padx=(0, 10), pady=12, sticky="ew")
        self.grok_entry.insert(0, self.config.get('xai', {}).get('api_key', ''))
        self.grok_entry.bind("<KeyRelease>", lambda e: self._update_model_visibility())
        
        self.test_grok_btn = ctk.CTkButton(
            self.xai_frame, cursor="hand2", text="Test Key", width=85, height=34,
            corner_radius=6, font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=THEME["accent_primary"], hover_color=THEME["accent_primary_hover"],
            command=self.test_grok_key
        )
        self.test_grok_btn.grid(row=0, column=2, padx=(0, 16), pady=12, sticky="e")

        self.xai_model_label = ctk.CTkLabel(self.xai_frame, text="AI Model:", font=ctk.CTkFont(size=12, weight="bold"), text_color=THEME["text_secondary"])
        self.xai_model_menu = ctk.CTkComboBox(self.xai_frame, cursor="hand2", values=model_fetcher.MAIN_MODELS["xai"], height=32, width=260)
        self.xai_model_menu.set(self.config.get("xai_model", "grok-2-latest"))

        # 5. Google Gemini
        self.google_frame = ctk.CTkFrame(self.api_card, fg_color=THEME["bg_card_alt"], corner_radius=8)
        self.google_frame.grid(row=5, column=0, columnspan=3, sticky="ew", pady=6, padx=16)
        self.google_frame.grid_columnconfigure(0, minsize=200)
        self.google_frame.grid_columnconfigure(1, weight=1)

        self.google_radio = ctk.CTkRadioButton(
            self.google_frame, cursor="hand2", text="Google Gemini (2M):", 
            font=ctk.CTkFont(size=13, weight="bold"), text_color=THEME["text_primary"],
            variable=self.active_provider_var, value="google"
        )
        self.google_radio.grid(row=0, column=0, padx=16, pady=12, sticky="w")
        
        self.google_entry = ctk.CTkEntry(
            self.google_frame, show="•", height=34, placeholder_text="AIzaSy...",
            fg_color=THEME["bg_input"], border_color=THEME["border_input"], text_color=THEME["text_primary"]
        )
        self.google_entry.grid(row=0, column=1, padx=(0, 10), pady=12, sticky="ew")
        self.google_entry.insert(0, self.config.get('google', {}).get('api_key', ''))
        self.google_entry.bind("<KeyRelease>", lambda e: self._update_model_visibility())
        
        self.test_google_btn = ctk.CTkButton(
            self.google_frame, cursor="hand2", text="Test Key", width=85, height=34,
            corner_radius=6, font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=THEME["accent_primary"], hover_color=THEME["accent_primary_hover"],
            command=self.test_google_key
        )
        self.test_google_btn.grid(row=0, column=2, padx=(0, 16), pady=12, sticky="e")

        self.google_model_label = ctk.CTkLabel(self.google_frame, text="AI Model:", font=ctk.CTkFont(size=12, weight="bold"), text_color=THEME["text_secondary"])
        self.google_model_menu = ctk.CTkComboBox(self.google_frame, cursor="hand2", values=model_fetcher.MAIN_MODELS["google"], height=32, width=260)
        self.google_model_menu.set(self.config.get("google_model", "gemini-2.5-flash"))

        # Whisper Settings
        ctk.CTkLabel(self.api_card, text="Whisper Transcribe Model:", font=ctk.CTkFont(size=13, weight="bold"), text_color=THEME["text_primary"]).grid(row=6, column=0, padx=20, pady=(14, 4), sticky="w")
        self.whisper_menu = ctk.CTkOptionMenu(self.api_card, cursor="hand2", values=["tiny", "base", "small", "medium", "large"], height=34, corner_radius=6, fg_color=THEME["bg_card_alt"], button_color=THEME["border_input"], text_color=THEME["text_primary"])
        self.whisper_menu.grid(row=6, column=1, columnspan=2, padx=(0, 16), pady=(14, 4), sticky="ew")
        self.whisper_menu.set(self.config.get('openai', {}).get('whisper_model', 'base'))
        
        ctk.CTkLabel(self.api_card, text="Audio Spoken Language:", font=ctk.CTkFont(size=13, weight="bold"), text_color=THEME["text_primary"]).grid(row=7, column=0, padx=20, pady=(4, 18), sticky="w")
        self.language_menu = ctk.CTkComboBox(self.api_card, cursor="hand2", values=["Auto-Detect", "English", "Spanish", "French", "German", "Italian", "Portuguese", "Russian", "Japanese", "Korean", "Chinese"], height=34, corner_radius=6, fg_color=THEME["bg_card_alt"], button_color=THEME["border_input"], text_color=THEME["text_primary"])
        self.language_menu.grid(row=7, column=1, columnspan=2, padx=(0, 16), pady=(4, 18), sticky="ew")
        self.language_menu.set(self.config.get('openai', {}).get('whisper_language', 'English'))

        self._update_model_visibility()

        # --- Card 2: Video Processing & Audio Intelligence ---
        self.proc_card = ctk.CTkFrame(
            self.settings_frame, corner_radius=12, 
            fg_color=THEME["bg_card"], border_width=1, border_color=THEME["border_subtle"]
        )
        self.proc_card.grid(row=2, column=0, padx=32, pady=(0, 16), sticky="ew")
        self.proc_card.grid_columnconfigure(0, weight=1)
        self.proc_card.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            self.proc_card, text="Video Processing & Audio Intelligence", 
            font=ctk.CTkFont(size=15, weight="bold"), text_color=THEME["text_primary"]
        ).grid(row=0, column=0, columnspan=2, padx=20, pady=(18, 12), sticky="w")

        self.hardware_switch = ctk.CTkSwitch(self.proc_card, cursor="hand2", text="GPU Hardware Acceleration (NVENC / AMF)", font=ctk.CTkFont(size=13, weight="bold"), text_color=THEME["text_primary"], progress_color=THEME["accent_primary"])
        self.hardware_switch.grid(row=1, column=0, padx=20, pady=8, sticky="w")

        self.downmix_switch = ctk.CTkSwitch(self.proc_card, cursor="hand2", text="Downmix Multi-Track OBS Audio (amix)", font=ctk.CTkFont(size=13, weight="bold"), text_color=THEME["text_primary"], progress_color=THEME["accent_primary"])
        self.downmix_switch.grid(row=1, column=1, padx=20, pady=8, sticky="w")

        self.audio_peak_switch = ctk.CTkSwitch(self.proc_card, cursor="hand2", text="Audio Peak & Screaming Loudness Mapping", font=ctk.CTkFont(size=13, weight="bold"), text_color=THEME["text_primary"], progress_color=THEME["accent_primary"])
        self.audio_peak_switch.grid(row=2, column=0, padx=20, pady=8, sticky="w")

        self.combat_switch = ctk.CTkSwitch(self.proc_card, cursor="hand2", text="Gunfight & Combat Transient Spikes", font=ctk.CTkFont(size=13, weight="bold"), text_color=THEME["text_primary"], progress_color=THEME["accent_primary"])
        self.combat_switch.grid(row=2, column=1, padx=20, pady=8, sticky="w")

        self.stabilize_switch = ctk.CTkSwitch(self.proc_card, cursor="hand2", text="Apply VR Anti-Shake Deshake Filter", font=ctk.CTkFont(size=13, weight="bold"), text_color=THEME["text_primary"], progress_color=THEME["accent_primary"])
        self.stabilize_switch.grid(row=3, column=0, padx=20, pady=8, sticky="w")

        self.vertical_switch = ctk.CTkSwitch(self.proc_card, cursor="hand2", text="Generate Vertical 9:16 Shorts", font=ctk.CTkFont(size=13, weight="bold"), text_color=THEME["text_primary"], progress_color=THEME["accent_primary"])
        self.vertical_switch.grid(row=4, column=0, padx=20, pady=(12, 6), sticky="w")

        self.vertical_mode_menu = ctk.CTkOptionMenu(
            self.proc_card, cursor="hand2", height=32, corner_radius=6,
            values=["Standard Center Crop", "Facecam Top-Left", "Facecam Top-Right", "Facecam Bottom-Left", "Facecam Bottom-Right", "Custom Coordinates"],
            fg_color=THEME["bg_card_alt"], button_color=THEME["border_input"], text_color=THEME["text_primary"]
        )
        self.vertical_mode_menu.grid(row=4, column=1, padx=(0, 20), pady=(12, 6), sticky="ew")

        self.coord_frame = ctk.CTkFrame(self.proc_card, fg_color="transparent")
        self.coord_frame.grid(row=5, column=1, padx=(0, 20), pady=(4, 18), sticky="w")

        for label_text, attr, default in [("X:", "crop_x_entry", "0"), ("Y:", "crop_y_entry", "0"), ("W:", "crop_w_entry", "400"), ("H:", "crop_h_entry", "225")]:
            ctk.CTkLabel(self.coord_frame, text=label_text, font=ctk.CTkFont(size=11), text_color=THEME["text_secondary"]).pack(side="left", padx=(6, 2))
            entry = ctk.CTkEntry(self.coord_frame, width=48, height=28, placeholder_text=default, fg_color=THEME["bg_input"], border_color=THEME["border_input"], text_color=THEME["text_primary"])
            entry.pack(side="left", padx=(0, 8))
            setattr(self, attr, entry)

        # Restore switches
        settings_cfg = self.config.get('settings', {})
        if settings_cfg.get('hardware_encoding', False): self.hardware_switch.select()
        else: self.hardware_switch.deselect()
        if settings_cfg.get('audio_downmix', True): self.downmix_switch.select()
        else: self.downmix_switch.deselect()
        if settings_cfg.get('audio_peak_detection', True): self.audio_peak_switch.select()
        else: self.audio_peak_switch.deselect()
        if settings_cfg.get('combat_detection', True): self.combat_switch.select()
        else: self.combat_switch.deselect()
        if settings_cfg.get('vr_stabilization', False): self.stabilize_switch.select()
        else: self.stabilize_switch.deselect()
        if settings_cfg.get('vertical_export', False): self.vertical_switch.select()
        else: self.vertical_switch.deselect()
        
        self.vertical_mode_menu.set(settings_cfg.get('vertical_mode', 'Standard Center Crop'))
        self.crop_x_entry.insert(0, settings_cfg.get('crop_x', '0'))
        self.crop_y_entry.insert(0, settings_cfg.get('crop_y', '0'))
        self.crop_w_entry.insert(0, settings_cfg.get('crop_w', '400'))
        self.crop_h_entry.insert(0, settings_cfg.get('crop_h', '225'))

        # --- Card 3: Storage & Integrations ---
        self.storage_card = ctk.CTkFrame(
            self.settings_frame, corner_radius=12, 
            fg_color=THEME["bg_card"], border_width=1, border_color=THEME["border_subtle"]
        )
        self.storage_card.grid(row=3, column=0, padx=32, pady=(0, 16), sticky="ew")
        self.storage_card.grid_columnconfigure(0, minsize=220)
        self.storage_card.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            self.storage_card, text="Output Directory & Discord Alerts", 
            font=ctk.CTkFont(size=15, weight="bold"), text_color=THEME["text_primary"]
        ).grid(row=0, column=0, columnspan=3, padx=20, pady=(18, 12), sticky="w")

        ctk.CTkLabel(self.storage_card, text="Generated Clips Folder:", font=ctk.CTkFont(size=13, weight="bold"), text_color=THEME["text_primary"]).grid(row=1, column=0, padx=20, pady=8, sticky="w")
        self.clip_dir_entry = ctk.CTkEntry(self.storage_card, height=34, placeholder_text="C:\\Videos\\Clips", fg_color=THEME["bg_input"], border_color=THEME["border_input"], text_color=THEME["text_primary"])
        self.clip_dir_entry.grid(row=1, column=1, padx=(0, 10), pady=8, sticky="ew")
        self.clip_dir_entry.insert(0, self.config.get('settings', {}).get('clips_dir', ''))
        self.clip_browse_btn = ctk.CTkButton(self.storage_card, cursor="hand2", text="Browse...", width=85, height=34, corner_radius=6, fg_color=THEME["bg_card_alt"], hover_color=THEME["border_subtle"], text_color=THEME["text_primary"], command=lambda: self.browse_folder(self.clip_dir_entry))
        self.clip_browse_btn.grid(row=1, column=2, padx=(0, 20), pady=8, sticky="e")

        ctk.CTkLabel(self.storage_card, text="Discord Webhook URL:", font=ctk.CTkFont(size=13, weight="bold"), text_color=THEME["text_primary"]).grid(row=2, column=0, padx=20, pady=(8, 18), sticky="w")
        self.discord_entry = ctk.CTkEntry(self.storage_card, height=34, placeholder_text="https://discord.com/api/webhooks/...", fg_color=THEME["bg_input"], border_color=THEME["border_input"], text_color=THEME["text_primary"])
        self.discord_entry.grid(row=2, column=1, padx=(0, 10), pady=(8, 18), sticky="ew")
        self.discord_entry.insert(0, self.config.get('integrations', {}).get('discord_webhook', ''))
        self.test_discord_btn = ctk.CTkButton(self.storage_card, cursor="hand2", text="Test Alert", width=85, height=34, corner_radius=6, fg_color=THEME["accent_primary"], hover_color=THEME["accent_primary_hover"], text_color="#FFFFFF", command=self.test_discord_webhook)
        self.test_discord_btn.grid(row=2, column=2, padx=(0, 20), pady=(8, 18), sticky="e")

        # Save Settings Action Bar
        self.settings_action_bar = ctk.CTkFrame(self.settings_frame, fg_color="transparent")
        self.settings_action_bar.grid(row=4, column=0, padx=32, pady=(4, 28), sticky="ew")

        self.save_btn = ctk.CTkButton(
            self.settings_action_bar, cursor="hand2", text="💾 Save Configuration", 
            height=44, width=180, corner_radius=8, font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=THEME["accent_primary"], hover_color=THEME["accent_primary_hover"],
            text_color="#FFFFFF", command=self.save_settings
        )
        self.save_btn.pack(side="right")

    def _setup_gallery_frame(self):
        self.gallery_frame = ctk.CTkFrame(self, fg_color=THEME["bg_window"])
        self.gallery_frame.grid_columnconfigure(1, weight=1)
        self.gallery_frame.grid_rowconfigure(3, weight=1)
        
        # Header
        self.gallery_header = ctk.CTkFrame(self.gallery_frame, fg_color="transparent")
        self.gallery_header.grid(row=0, column=0, columnspan=2, padx=32, pady=(28, 12), sticky="ew")
        
        self.gallery_title = ctk.CTkLabel(
            self.gallery_header, text="Extracted Clip Gallery", 
            font=ctk.CTkFont(size=26, weight="bold"), text_color=THEME["text_primary"]
        )
        self.gallery_title.pack(anchor="w")

        self.gallery_subtitle = ctk.CTkLabel(
            self.gallery_header, text="Inspect generated clips, preview high-res thumbnails, view AI reasoning notes, and manage cuts.", 
            font=ctk.CTkFont(size=13), text_color=THEME["text_secondary"]
        )
        self.gallery_subtitle.pack(anchor="w", pady=(3, 0))

        # Filters & Sorting Toolbar
        self.toolbar_card = ctk.CTkFrame(
            self.gallery_frame, corner_radius=10, 
            fg_color=THEME["bg_card"], border_width=1, border_color=THEME["border_subtle"]
        )
        self.toolbar_card.grid(row=1, column=0, padx=(32, 12), pady=(0, 8), sticky="ew")
        
        ctk.CTkLabel(self.toolbar_card, text="Sort:", font=ctk.CTkFont(size=12, weight="bold"), text_color=THEME["text_secondary"]).pack(side="left", padx=(12, 4), pady=8)
        self.sort_menu = ctk.CTkOptionMenu(
            self.toolbar_card, cursor="hand2", height=30, width=140, corner_radius=6,
            values=["Date (Newest)", "Date (Oldest)", "Virality (High)", "Virality (Low)"],
            fg_color=THEME["bg_card_alt"], button_color=THEME["border_input"], text_color=THEME["text_primary"],
            command=lambda _: self.populate_gallery()
        )
        self.sort_menu.pack(side="left", padx=4, pady=8)
        self.sort_menu.set("Date (Newest)")

        ctk.CTkLabel(self.toolbar_card, text="Type:", font=ctk.CTkFont(size=12, weight="bold"), text_color=THEME["text_secondary"]).pack(side="left", padx=(10, 4), pady=8)
        self.type_filter_menu = ctk.CTkOptionMenu(
            self.toolbar_card, cursor="hand2", height=30, width=95, corner_radius=6,
            values=["All", "Horizontal", "Vertical"],
            fg_color=THEME["bg_card_alt"], button_color=THEME["border_input"], text_color=THEME["text_primary"],
            command=lambda _: self.populate_gallery()
        )
        self.type_filter_menu.pack(side="left", padx=4, pady=8)
        self.type_filter_menu.set("All")

        ctk.CTkLabel(self.toolbar_card, text="Min Score:", font=ctk.CTkFont(size=12, weight="bold"), text_color=THEME["text_secondary"]).pack(side="left", padx=(10, 4), pady=8)
        self.score_filter_menu = ctk.CTkOptionMenu(
            self.toolbar_card, cursor="hand2", height=30, width=75, corner_radius=6,
            values=["All", "3+", "5+", "7+", "8+", "9+"],
            fg_color=THEME["bg_card_alt"], button_color=THEME["border_input"], text_color=THEME["text_primary"],
            command=lambda _: self.populate_gallery()
        )
        self.score_filter_menu.pack(side="left", padx=(4, 12), pady=8)
        self.score_filter_menu.set("All")

        # Left List Container
        self.clip_listbox = ctk.CTkScrollableFrame(
            self.gallery_frame, width=340, corner_radius=12, 
            fg_color=THEME["bg_card"], border_width=1, border_color=THEME["border_subtle"],
            scrollbar_button_color=THEME["border_subtle"], scrollbar_button_hover_color=THEME["border_input"]
        )
        self.clip_listbox.grid(row=2, column=0, rowspan=2, padx=(32, 12), pady=(0, 10), sticky="nsew")

        # Batch Action Bar below List
        self.gallery_actions_frame = ctk.CTkFrame(self.gallery_frame, fg_color="transparent")
        self.gallery_actions_frame.grid(row=4, column=0, padx=(32, 12), pady=(0, 24), sticky="ew")

        self.select_all_var = ctk.BooleanVar(value=False)
        self.select_all_checkbox = ctk.CTkCheckBox(
            self.gallery_actions_frame, cursor="hand2", text="Select All", 
            font=ctk.CTkFont(size=12), text_color=THEME["text_secondary"],
            variable=self.select_all_var, command=self.toggle_select_all,
            fg_color=THEME["accent_primary"], hover_color=THEME["accent_primary_hover"]
        )
        self.select_all_checkbox.pack(side="left", padx=(4, 8))

        self.refresh_gallery_btn = ctk.CTkButton(
            self.gallery_actions_frame, cursor="hand2", text="🔄 Refresh", 
            height=32, width=80, corner_radius=6, font=ctk.CTkFont(size=11),
            fg_color=THEME["bg_card_alt"], hover_color=THEME["border_subtle"],
            text_color=THEME["text_secondary"], command=self.refresh_gallery_action
        )
        self.refresh_gallery_btn.pack(side="left", padx=4)

        self.delete_marked_btn = ctk.CTkButton(
            self.gallery_actions_frame, cursor="hand2", text="🗑️ Delete", 
            height=32, corner_radius=6, font=ctk.CTkFont(size=11, weight="bold"),
            fg_color=THEME["accent_danger"], hover_color=THEME["accent_danger_hover"],
            text_color="#FFFFFF", command=self.confirm_delete_marked
        )
        self.delete_marked_btn.pack(side="right", padx=(4, 0))

        # Right Inspector Card
        self.details_card = ctk.CTkFrame(
            self.gallery_frame, corner_radius=12, 
            fg_color=THEME["bg_card"], border_width=1, border_color=THEME["border_subtle"]
        )
        self.details_card.grid(row=1, column=1, rowspan=4, padx=(0, 32), pady=(0, 24), sticky="nsew")
        self.details_card.grid_columnconfigure(0, weight=1)
        self.details_card.grid_rowconfigure(3, weight=1)

        self.detail_title = ctk.CTkLabel(
            self.details_card, text="Select a clip from the list to preview", 
            font=ctk.CTkFont(size=16, weight="bold"), text_color=THEME["text_primary"],
            anchor="w"
        )
        self.detail_title.grid(row=0, column=0, padx=20, pady=(18, 4), sticky="ew")

        self.detail_score = ctk.CTkLabel(
            self.details_card, text="Virality Score: --/10", 
            font=ctk.CTkFont(size=14, weight="bold"), text_color=THEME["accent_success"],
            anchor="w"
        )
        self.detail_score.grid(row=1, column=0, padx=20, pady=(0, 8), sticky="w")

        # Large Poster Thumbnail Container
        self.thumb_container = ctk.CTkFrame(self.details_card, corner_radius=8, fg_color=THEME["bg_input"], height=240)
        self.thumb_container.grid(row=2, column=0, padx=20, pady=(0, 12), sticky="nsew")
        self.thumb_container.pack_propagate(False)

        self.detail_thumbnail = ctk.CTkLabel(self.thumb_container, text="No Preview Selected", text_color=THEME["text_dim"])
        self.detail_thumbnail.pack(expand=True)

        # AI Justification / Reasoning Textbox
        self.detail_reasoning = ctk.CTkTextbox(
            self.details_card, font=ctk.CTkFont(size=13), wrap="word",
            fg_color=THEME["bg_input"], corner_radius=8, text_color=THEME["text_primary"]
        )
        self.detail_reasoning.grid(row=3, column=0, padx=20, pady=(0, 14), sticky="nsew")
        self.detail_reasoning.insert("1.0", "AI highlight explanation will appear here.")
        self.detail_reasoning.configure(state="disabled")

        # Player Controls Bar
        self.gallery_btns_frame = ctk.CTkFrame(self.details_card, fg_color="transparent")
        self.gallery_btns_frame.grid(row=4, column=0, padx=20, pady=(0, 18), sticky="ew")
        self.gallery_btns_frame.grid_columnconfigure((0, 1), weight=1)

        self.play_clip_btn = ctk.CTkButton(
            self.gallery_btns_frame, cursor="hand2", text="▶️ Play Video Highlight", 
            height=42, corner_radius=8, font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=THEME["accent_primary"], hover_color=THEME["accent_primary_hover"],
            text_color="#FFFFFF", state="disabled"
        )
        self.play_clip_btn.grid(row=0, column=0, padx=(0, 6), sticky="ew")

        self.open_folder_btn = ctk.CTkButton(
            self.gallery_btns_frame, cursor="hand2", text="📁 Show in Explorer", 
            height=42, corner_radius=8, font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=THEME["bg_card_alt"], hover_color=THEME["border_subtle"],
            border_width=1, border_color=THEME["border_input"],
            text_color=THEME["text_primary"], state="disabled"
        )
        self.open_folder_btn.grid(row=0, column=1, padx=(6, 0), sticky="ew")

    # ==============================================================================
    # LOGGING & CONSOLE STREAM
    # ==============================================================================
    def _init_logging(self):
        log_dir = os.path.join(get_app_data_path(), "logs")
        os.makedirs(log_dir, exist_ok=True)
        formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        
        manual_path = os.path.join(log_dir, "processor.log")
        self.manual_logger = logging.getLogger("manual_logger")
        self.manual_logger.setLevel(logging.INFO)
        self.manual_logger.propagate = False
        if not self.manual_logger.handlers:
            manual_handler = RotatingFileHandler(manual_path, maxBytes=1024*1024, backupCount=9, encoding='utf-8')
            manual_handler.setFormatter(formatter)
            self.manual_logger.addHandler(manual_handler)

    def clear_console(self):
        if hasattr(self, 'console_box'):
            self.console_box.configure(state="normal")
            self.console_box.delete("1.0", "end")
            self.console_box.configure(state="disabled")

    def log_to_console(self, text, source="system"):
        tag = None
        clean_msg = text.strip()
        if "❌" in clean_msg or "error" in clean_msg.lower(): tag = "error"
        elif "✅" in clean_msg or "✨" in clean_msg or "🏁" in clean_msg: tag = "success"
        elif "🧠" in clean_msg or "🌌" in clean_msg or "🤖" in clean_msg or "🎯" in clean_msg or "🚀" in clean_msg or "⚡" in clean_msg: tag = "ai"
        elif "✂️" in clean_msg or "🎞️" in clean_msg or "📸" in clean_msg or "📱" in clean_msg: tag = "ffmpeg"
        
        status_clean = clean_msg.rpartition("]")[2].strip() if "]" in clean_msg else clean_msg
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if clean_msg.startswith("[") and "]" in clean_msg[:25]:
            display_text = clean_msg
        else:
            display_text = f"[{timestamp}] {clean_msg}"

        try:
            print(display_text, flush=True)
        except Exception:
            pass

        def update_text():
            if hasattr(self, 'manual_status_label') and source in ["manual", "system"]:
                if tag == "error":
                    self.manual_status_label.configure(text=f"● Status: {status_clean}", text_color=THEME["accent_danger"])
                elif tag == "success":
                    self.manual_status_label.configure(text=f"● Status: {status_clean}", text_color=THEME["accent_success"])
                else:
                    self.manual_status_label.configure(text=f"● Status: {status_clean}", text_color=THEME["accent_primary"])
                    
            if hasattr(self, 'console_box'):
                self.console_box.configure(state="normal")
                if tag:
                    self.console_box.insert("end", display_text + "\n", tag)
                else:
                    self.console_box.insert("end", display_text + "\n")
                self.console_box.configure(state="disabled")
                self.console_box.see("end")
        self.after(0, update_text)
        
        if hasattr(self, 'manual_logger'):
            self.manual_logger.info(clean_msg)
        
        try:
            crash_log_path = os.path.join(get_app_data_path(), "app_crash_log.txt")
            with open(crash_log_path, "a", encoding="utf-8") as f:
                f.write(display_text + "\n")
        except Exception:
            pass

    # ==============================================================================
    # API KEY TESTERS & DISCORD WEBHOOKS
    # ==============================================================================
    def test_openai_key(self):
        key = self.openai_entry.get().strip()
        base_url = self.base_url_entry.get().strip()
        self.test_openai_btn.configure(text="Testing...", fg_color=THEME["accent_amber"])
        def run_test():
            try:
                from openai import OpenAI
                client_args = {"api_key": key if key else "blank_valid_key"}
                if base_url: client_args["base_url"] = base_url
                client = OpenAI(**client_args)
                client.models.list() 
                self.after(0, lambda: self.test_openai_btn.configure(text="✅ Valid!", fg_color=THEME["accent_success"]))
                self.log_to_console("✅ OpenAI / Custom API verified successfully!", source="system")
            except Exception as e:
                self.log_to_console(f"❌ OpenAI Key Test Error: {e}", source="system")
                self.after(0, lambda: self.test_openai_btn.configure(text="❌ Invalid", fg_color=THEME["accent_danger"]))
            self.after(3000, lambda: self.test_openai_btn.configure(text="Test Key", fg_color=THEME["accent_primary"]))
        threading.Thread(target=run_test, daemon=True).start()

    def test_deepseek_key(self):
        key = self.deepseek_entry.get().strip()
        self.test_deepseek_btn.configure(text="Testing...", fg_color=THEME["accent_amber"])
        def run_test():
            try:
                from openai import OpenAI
                client = OpenAI(api_key=key if key else "blank_valid_key", base_url="https://api.deepseek.com")
                client.models.list() 
                self.after(0, lambda: self.test_deepseek_btn.configure(text="✅ Valid!", fg_color=THEME["accent_success"]))
                self.log_to_console("✅ DeepSeek API verified successfully!", source="system")
            except Exception as e:
                self.log_to_console(f"❌ DeepSeek Key Test Error: {e}", source="system")
                self.after(0, lambda: self.test_deepseek_btn.configure(text="❌ Invalid", fg_color=THEME["accent_danger"]))
            self.after(3000, lambda: self.test_deepseek_btn.configure(text="Test Key", fg_color=THEME["accent_primary"]))
        threading.Thread(target=run_test, daemon=True).start()

    def test_anthropic_key(self):
        key = self.anthropic_entry.get().strip()
        self.test_anthropic_btn.configure(text="Testing...", fg_color=THEME["accent_amber"])
        def run_test():
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=key)
                client.models.list() 
                self.after(0, lambda: self.test_anthropic_btn.configure(text="✅ Valid!", fg_color=THEME["accent_success"]))
                self.log_to_console("✅ Anthropic API verified successfully!", source="system")
            except Exception as e:
                self.log_to_console(f"❌ Anthropic Key Test Error: {e}", source="system")
                self.after(0, lambda: self.test_anthropic_btn.configure(text="❌ Invalid", fg_color=THEME["accent_danger"]))
            self.after(3000, lambda: self.test_anthropic_btn.configure(text="Test Key", fg_color=THEME["accent_primary"]))
        threading.Thread(target=run_test, daemon=True).start()

    def test_grok_key(self):
        key = self.grok_entry.get().strip()
        self.test_grok_btn.configure(text="Testing...", fg_color=THEME["accent_amber"])
        def run_test():
            try:
                from openai import OpenAI
                client = OpenAI(api_key=key, base_url="https://api.x.ai/v1")
                client.models.list() 
                self.after(0, lambda: self.test_grok_btn.configure(text="✅ Valid!", fg_color=THEME["accent_success"]))
                self.log_to_console("✅ Grok / xAI API verified successfully!", source="system")
            except Exception as e:
                self.log_to_console(f"❌ Grok Key Test Error: {e}", source="system")
                self.after(0, lambda: self.test_grok_btn.configure(text="❌ Invalid", fg_color=THEME["accent_danger"]))
            self.after(3000, lambda: self.test_grok_btn.configure(text="Test Key", fg_color=THEME["accent_primary"]))
        threading.Thread(target=run_test, daemon=True).start()

    def test_google_key(self):
        key = self.google_entry.get().strip()
        self.test_google_btn.configure(text="Testing...", fg_color=THEME["accent_amber"])
        def run_test():
            try:
                from google import genai
                client = genai.Client(api_key=key)
                list(client.models.list())
                self.after(0, lambda: self.test_google_btn.configure(text="✅ Valid!", fg_color=THEME["accent_success"]))
                self.log_to_console("✅ Google Gemini API verified successfully!", source="system")
            except Exception as e:
                self.log_to_console(f"❌ Google Key Test Error: {e}", source="system")
                self.after(0, lambda: self.test_google_btn.configure(text="❌ Invalid", fg_color=THEME["accent_danger"]))
            self.after(3000, lambda: self.test_google_btn.configure(text="Test Key", fg_color=THEME["accent_primary"]))
        threading.Thread(target=run_test, daemon=True).start()

    def _show_transient_button_state(self, widget, temp_text, temp_color=THEME["accent_amber"], duration=2000):
        if not widget: return
        if widget.cget("text") != temp_text:
            if not hasattr(widget, "_orig_text"):
                widget._orig_text = widget.cget("text")
                widget._orig_color = widget.cget("fg_color")
        widget.configure(text=temp_text, fg_color=temp_color)
        def restore():
            if hasattr(widget, "_orig_text"):
                widget.configure(text=widget._orig_text, fg_color=widget._orig_color)
                delattr(widget, "_orig_text")
                delattr(widget, "_orig_color")
        self.after(duration, restore)

    def test_discord_webhook(self):
        url = self.discord_entry.get().strip()
        if not url:
            self._show_transient_button_state(self.test_discord_btn, "⚠️ No URL")
            return
        self.test_discord_btn.configure(text="Testing...", fg_color=THEME["accent_amber"])
        def run_test():
            try:
                parsed = urllib.parse.urlparse(url)
                if parsed.scheme != "https" or parsed.hostname != "discord.com" or not parsed.path.startswith("/api/webhooks/") or ".." in urllib.parse.unquote(parsed.path):
                    raise ValueError("Invalid Discord URL")
                headers = {"Content-Type": "application/json", "User-Agent": "jBahrsClipGen/1.2.1"}
                data = json.dumps({"content": "✅ **Test Alert from Clip Generator!** The Webhook link is alive."}).encode('utf-8')
                req = urllib.request.Request(url, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=10) as response:
                    if response.status in [200, 204]:
                        self.after(0, lambda: self.test_discord_btn.configure(text="✅ Valid!", fg_color=THEME["accent_success"]))
                        self.after(3000, lambda: self.test_discord_btn.configure(text="Test Alert", fg_color=THEME["accent_primary"]))
                        return
                raise ValueError(f"Bad response: {response.status}")
            except Exception as e:
                self.log_to_console(f"❌ Discord Test Failed: {str(e)}")
                self.after(0, lambda: self.test_discord_btn.configure(text="❌ Invalid", fg_color=THEME["accent_danger"]))
                self.after(3000, lambda: self.test_discord_btn.configure(text="Test Alert", fg_color=THEME["accent_primary"]))
        threading.Thread(target=run_test, daemon=True).start()

    def send_discord_alert(self, title):
        url = self.config.get("integrations", {}).get("discord_webhook", "").strip()
        if not url: return
        def run_alert():
            try:
                parsed = urllib.parse.urlparse(url)
                if parsed.scheme != "https" or parsed.hostname != "discord.com" or not parsed.path.startswith("/api/webhooks/") or ".." in urllib.parse.unquote(parsed.path):
                    return
                payload = {
                    "content": None,
                    "embeds": [{
                        "title": "🎬 Generation Complete!",
                        "description": f"The Application has finished processing your queue.\n**Event:** {title}",
                        "color": 3066993
                    }]
                }
                headers = {"Content-Type": "application/json", "User-Agent": "jBahrsClipGen/1.2.1"}
                data = json.dumps(payload).encode('utf-8')
                req = urllib.request.Request(url, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=10) as _:
                    pass
            except Exception as e:
                self.log_to_console(f"❌ Discord Webhook Failed: {e}")
        threading.Thread(target=run_alert, daemon=True).start()

    def _update_model_visibility(self):
        providers = [
            (self.openai_entry, self.openai_model_label, self.openai_model_menu),
            (self.deepseek_entry, self.deepseek_model_label, self.deepseek_model_menu),
            (self.anthropic_entry, self.anthropic_model_label, self.anthropic_model_menu),
            (self.grok_entry, self.xai_model_label, self.xai_model_menu),
            (self.google_entry, self.google_model_label, self.google_model_menu)
        ]
        for entry, label, menu in providers:
            if entry.get().strip():
                label.grid(row=1, column=0, padx=(36, 10), pady=(0, 10), sticky="w")
                menu.grid(row=1, column=1, padx=(0, 10), pady=(0, 10), sticky="w")
            else:
                label.grid_forget()
                menu.grid_forget()

    def refresh_model_list(self):
        self.log_to_console("🔄 Syncing latest AI cloud model rosters in background...")
        try:
            openai_models = model_fetcher.fetch_openai_models(self.openai_entry.get().strip(), self.base_url_entry.get().strip())
            if openai_models: self.openai_model_menu.configure(values=openai_models)
            
            deepseek_models = model_fetcher.fetch_deepseek_models(self.deepseek_entry.get().strip())
            if deepseek_models: self.deepseek_model_menu.configure(values=deepseek_models)
            
            anthropic_models = model_fetcher.fetch_anthropic_models(self.anthropic_entry.get().strip())
            if anthropic_models: self.anthropic_model_menu.configure(values=anthropic_models)
            
            xai_models = model_fetcher.fetch_xai_models(self.grok_entry.get().strip())
            if xai_models: self.xai_model_menu.configure(values=xai_models)
            
            google_models = model_fetcher.fetch_google_models(self.google_entry.get().strip())
            if google_models: self.google_model_menu.configure(values=google_models)
        except Exception as e:
            self.log_to_console(f"❌ Failed to refresh model list: {e}")

    # ==============================================================================
    # NAVIGATION & FRAME SWITCHING
    # ==============================================================================
    def browse_folder(self, entry_widget):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            entry_widget.delete(0, "end")
            entry_widget.insert(0, folder_selected)

    def open_logs(self):
        log_path = os.path.join(get_app_data_path(), "app_crash_log.txt")
        if os.path.exists(log_path):
            subprocess.run(['explorer', '/select,', log_path], timeout=5)
        else:
            app_data_path = get_app_data_path()
            if hasattr(os, 'startfile'):
                os.startfile(os.path.abspath(app_data_path))

    def open_local_folder(self, key, widget=None):
        path = self.config.get('settings', {}).get(key, "")
        if path and os.path.exists(path) and hasattr(os, 'startfile'):
            os.startfile(os.path.abspath(path))
        else:
            self._show_transient_button_state(widget, "⚠️ Not Found")

    def open_readme(self, widget=None):
        if os.path.exists("README.md") and hasattr(os, 'startfile'): 
            os.startfile(os.path.abspath("README.md"))
        elif os.path.exists("readme.md") and hasattr(os, 'startfile'):
            os.startfile(os.path.abspath("readme.md"))
        else:
            self._show_transient_button_state(widget, "⚠️ Not Found")

    def show_manual_frame(self):
        self._hide_all_frames()
        self.manual_frame.grid(row=0, column=1, sticky="nsew")
        self._highlight_button(self.nav_manual_btn)

    def show_prompt_frame(self):
        self._hide_all_frames()
        self.prompt_frame.grid(row=0, column=1, sticky="nsew")
        self._highlight_button(self.nav_prompt_btn)

    def show_settings_frame(self):
        self._hide_all_frames()
        self.settings_frame.grid(row=0, column=1, sticky="nsew")
        self._highlight_button(self.nav_settings_btn)

    def show_gallery_frame(self):
        self._hide_all_frames()
        self.gallery_frame.grid(row=0, column=1, sticky="nsew")
        self._highlight_button(self.nav_gallery_btn)
        self.populate_gallery()

    def _hide_all_frames(self):
        for f in [self.manual_frame, self.prompt_frame, self.settings_frame, self.gallery_frame]: 
            f.grid_forget()

    def _highlight_button(self, active_button):
        for btn in [self.nav_manual_btn, self.nav_prompt_btn, self.nav_settings_btn, self.nav_gallery_btn]: 
            btn.configure(fg_color="transparent", text_color=THEME["text_secondary"])
        active_button.configure(fg_color=THEME["accent_primary"], text_color="#FFFFFF")

    # ==============================================================================
    # PROMPT MANAGEMENT
    # ==============================================================================
    def load_prompt_data(self):
        profiles = self.config.get("prompts", {}).get("profiles", {})
        if not profiles: return
        p_names = list(profiles.keys())
        self.profile_dropdown.configure(values=p_names)
        active = self.config["prompts"].get("active_profile", p_names[0])
        self.profile_dropdown.set(active)
        self.on_profile_change(active)

    def on_profile_change(self, choice):
        self.config["prompts"]["active_profile"] = choice
        p_text = self.config["prompts"]["profiles"].get(choice, "")
        self.prompt_textbox.delete("1.0", "end")
        self.prompt_textbox.insert("1.0", p_text)

    def create_new_profile(self):
        dialog = ctk.CTkInputDialog(text="Enter a name for your new prompt profile:", title="New Profile")
        new_name = dialog.get_input()
        if new_name is None: return
        if new_name:
            new_name = new_name.strip()
            if new_name and new_name not in self.config["prompts"]["profiles"]:
                self.config["prompts"]["profiles"][new_name] = "You are a specialized Gaming Editor. Your goal is to..."
                self.config["prompts"]["active_profile"] = new_name
                config_manager.save_config(self.config)
                self.load_prompt_data()
                self.log_to_console(f"📝 Created new prompt profile: '{new_name}'")
            elif new_name in self.config["prompts"]["profiles"]:
                self._show_transient_button_state(self.new_profile_btn, "⚠️ Exists")
            else:
                self._show_transient_button_state(self.new_profile_btn, "⚠️ Empty")
        else:
            self._show_transient_button_state(self.new_profile_btn, "⚠️ Empty")

    def save_current_prompt(self):
        active = self.profile_dropdown.get()
        self.config["prompts"]["profiles"][active] = self.prompt_textbox.get("1.0", "end").strip()
        config_manager.save_config(self.config)
        self._show_transient_button_state(self.save_prompt_btn, "✅ Saved!", temp_color=THEME["accent_success"])

    def delete_profile(self):
        active = self.profile_dropdown.get()
        if len(self.config["prompts"]["profiles"]) > 1:
            if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete the profile '{active}'?"):
                del self.config["prompts"]["profiles"][active]
                config_manager.save_config(self.config)
                self.load_prompt_data()
        else:
            messagebox.showwarning("Cannot Delete", "You must have at least one prompt profile.")

    # ==============================================================================
    # CONFIGURATION & SETTINGS
    # ==============================================================================
    def save_settings(self):
        self.config['openai']['api_key'] = self.openai_entry.get()
        self.config['openai']['base_url'] = self.base_url_entry.get()
        if 'deepseek' not in self.config: self.config['deepseek'] = {}
        self.config['deepseek']['api_key'] = self.deepseek_entry.get()
        self.config['anthropic']['api_key'] = self.anthropic_entry.get()
        self.config['xai']['api_key'] = self.grok_entry.get()
        self.config['google']['api_key'] = self.google_entry.get()
        self.config['integrations']['discord_webhook'] = self.discord_entry.get()
        
        self.config['active_ai_provider'] = self.active_provider_var.get()
        self.config['openai_model'] = self.openai_model_menu.get()
        self.config['deepseek_model'] = self.deepseek_model_menu.get()
        self.config['anthropic_model'] = self.anthropic_model_menu.get()
        self.config['xai_model'] = self.xai_model_menu.get()
        self.config['google_model'] = self.google_model_menu.get()
        
        active_prov = self.config['active_ai_provider']
        if active_prov == "openai": self.config['openai']['chat_model'] = self.config['openai_model']
        elif active_prov == "deepseek": self.config['openai']['chat_model'] = self.config['deepseek_model']
        elif active_prov == "anthropic": self.config['openai']['chat_model'] = self.config['anthropic_model']
        elif active_prov == "xai": self.config['openai']['chat_model'] = self.config['xai_model']
        elif active_prov == "google": self.config['openai']['chat_model'] = self.config['google_model']
        
        self.config['openai']['whisper_model'] = self.whisper_menu.get()
        self.config['openai']['whisper_language'] = self.language_menu.get()
        self.config['settings']['clips_dir'] = self.clip_dir_entry.get()
        
        self.config['settings']['vr_stabilization'] = self.stabilize_switch.get() == 1
        self.config['settings']['hardware_encoding'] = self.hardware_switch.get() == 1
        self.config['settings']['audio_downmix'] = self.downmix_switch.get() == 1
        self.config['settings']['audio_peak_detection'] = self.audio_peak_switch.get() == 1
        self.config['settings']['combat_detection'] = self.combat_switch.get() == 1
        self.config['settings']['vertical_export'] = self.vertical_switch.get() == 1
        self.config['settings']['vertical_mode'] = self.vertical_mode_menu.get()
        self.config['settings']['crop_x'] = self.crop_x_entry.get()
        self.config['settings']['crop_y'] = self.crop_y_entry.get()
        self.config['settings']['crop_w'] = self.crop_w_entry.get()
        self.config['settings']['crop_h'] = self.crop_h_entry.get()
        
        config_manager.save_config(self.config)
        self.log_to_console("✅ Settings saved successfully!")
        self._show_transient_button_state(self.save_btn, "✅ Saved!", temp_color=THEME["accent_success"])
        threading.Thread(target=self.refresh_model_list, daemon=True).start()

    # ==============================================================================
    # GALLERY & INSPECTION
    # ==============================================================================
    def refresh_gallery_action(self):
        self.populate_gallery()
        self._show_transient_button_state(self.refresh_gallery_btn, "✅ Refreshed", temp_color=THEME["accent_success"])

    def toggle_select_all(self):
        select_state = self.select_all_var.get()
        if hasattr(self, 'marked_for_deletion'):
            for var in self.marked_for_deletion.values():
                var.set(select_state)

    def populate_gallery(self):
        for widget in self.clip_listbox.winfo_children():
            widget.destroy()
        if hasattr(self, 'select_all_var'):
            self.select_all_var.set(False)
            
        clips_dir = self.config.get('settings', {}).get('clips_dir', '')
        if not clips_dir or not os.path.exists(clips_dir):
            empty_label = ctk.CTkLabel(
                self.clip_listbox, text="Clips destination folder not configured or empty.", 
                font=ctk.CTkFont(size=12, slant="italic"), text_color=THEME["text_muted"]
            )
            empty_label.pack(pady=30)
            return

        sort_mode = self.sort_menu.get()
        self.marked_for_deletion = {}
        clip_data = []

        if not hasattr(self, 'metadata_cache'):
            self.metadata_cache = {}

        try:
            with os.scandir(clips_dir) as entries:
                entries_list = list(entries)
                for entry in entries_list:
                    if entry.name.endswith(".mp4") and entry.is_file():
                        f = entry.name
                        ctime = entry.stat().st_ctime
                        score = 0
                        if "Virality" in sort_mode:
                            if "_score" in f:
                                try:
                                    score_part = f.rpartition("_score")[2]
                                    if "_vertical" in score_part:
                                        score_part = score_part.partition("_vertical")[0]
                                    else:
                                        score_part = score_part.partition(".mp4")[0]
                                    score = float(score_part)
                                except ValueError:
                                    pass
                            if score == 0:
                                json_path = os.path.join(clips_dir, f.replace("_vertical.mp4", ".mp4").replace(".mp4", ".json"))
                                if json_path in self.metadata_cache and self.metadata_cache[json_path][0] == ctime:
                                    score = self.metadata_cache[json_path][1]
                                elif os.path.exists(json_path):
                                    try:
                                        with open(json_path, 'r', encoding='utf-8') as jf:
                                            jdata = json.load(jf)
                                            score = float(jdata.get("virality_score", 0))
                                            self.metadata_cache[json_path] = (ctime, score)
                                    except Exception:
                                        pass
                        clip_data.append({"filename": f, "ctime": ctime, "score": score})
        except Exception as e:
            self.log_to_console(f"❌ Error scanning gallery folder: {e}")

        # Sorting
        if sort_mode == "Date (Newest)":
            clip_data.sort(key=lambda x: x["ctime"], reverse=True)
        elif sort_mode == "Date (Oldest)":
            clip_data.sort(key=lambda x: x["ctime"])
        elif sort_mode == "Virality (High)":
            clip_data.sort(key=lambda x: (x["score"], x["ctime"]), reverse=True)
        elif sort_mode == "Virality (Low)":
            clip_data.sort(key=lambda x: (x["score"], -x["ctime"]))

        # Filtering
        type_filter = self.type_filter_menu.get()
        score_filter = self.score_filter_menu.get()
        min_score = 0
        if score_filter != "All":
            min_score = int(score_filter.replace("+", ""))

        visible_count = 0
        for item in clip_data:
            file = item["filename"]
            is_vertical = "_vertical" in str(file)
            if type_filter == "Horizontal" and is_vertical: continue
            if type_filter == "Vertical" and not is_vertical: continue
            if float(item["score"]) < min_score: continue
            
            visible_count += 1
            row_card = ctk.CTkFrame(
                self.clip_listbox, fg_color=THEME["bg_card_alt"], 
                corner_radius=8, border_width=1, border_color=THEME["border_subtle"]
            )
            row_card.pack(fill="x", pady=3, padx=4)

            self.marked_for_deletion[file] = ctk.BooleanVar(value=False)
            checkbox = ctk.CTkCheckBox(
                row_card, cursor="hand2", text="", 
                variable=self.marked_for_deletion[file], width=20,
                fg_color=THEME["accent_primary"], hover_color=THEME["accent_primary_hover"]
            )
            checkbox.pack(side="left", padx=(10, 4), pady=6)

            display_name = file
            if len(display_name) > 34:
                display_name = display_name[:16] + "..." + display_name[-14:]

            btn = ctk.CTkButton(
                row_card, cursor="hand2", text=display_name, anchor="w",
                fg_color="transparent", hover_color=THEME["border_subtle"],
                font=ctk.CTkFont(size=12), text_color=THEME["text_primary"],
                command=lambda f=file: self.load_clip_details(f, clips_dir)
            )
            btn.pack(side="left", fill="x", expand=True, padx=(0, 6), pady=4)

        if visible_count == 0:
            empty_label = ctk.CTkLabel(
                self.clip_listbox, text="No clips matching active filter parameters.", 
                font=ctk.CTkFont(size=12, slant="italic"), text_color=THEME["text_muted"]
            )
            empty_label.pack(pady=30)

    def confirm_delete_marked(self):
        files_to_delete = [f for f, var in getattr(self, 'marked_for_deletion', {}).items() if var.get()]
        if not files_to_delete:
            self._show_transient_button_state(self.delete_marked_btn, "⚠️ None Selected", temp_color=THEME["accent_amber"])
            return
            
        confirm = messagebox.askyesno(
            "Confirm Deletion", 
            f"Are you sure you want to delete {len(files_to_delete)} marked clip(s)?\n\nAssociated .jpg thumbnails and .json sidecars will also be removed."
        )
        if confirm:
            clips_dir = self.config.get('settings', {}).get('clips_dir', '')
            for f in files_to_delete:
                mp4_path = os.path.join(clips_dir, f)
                json_path = os.path.join(clips_dir, f.replace("_vertical.mp4", ".mp4").replace(".mp4", ".json"))
                jpg_path = os.path.join(clips_dir, f.replace("_vertical.mp4", ".mp4").replace(".mp4", ".jpg"))
                for p in [mp4_path, json_path, jpg_path]:
                    if os.path.exists(p):
                        try: os.remove(p)
                        except Exception as e: self.log_to_console(f"❌ Failed to delete {p}: {e}")
            
            self.populate_gallery()
            self.detail_title.configure(text="Select a clip from the list to preview")
            self.detail_score.configure(text="Virality Score: --/10")
            self.detail_reasoning.configure(state="normal")
            self.detail_reasoning.delete("1.0", "end")
            self.detail_reasoning.configure(state="disabled")
            self.detail_thumbnail.configure(image=None, text="No Preview Selected")
            self.play_clip_btn.configure(state="disabled")
            self.open_folder_btn.configure(state="disabled")

    def load_clip_details(self, filename, directory):
        self.detail_title.configure(text=filename)
        mp4_path = os.path.join(directory, filename)
        
        base_json_name = filename.replace("_vertical.mp4", ".mp4").replace(".mp4", ".json")
        json_path = os.path.join(directory, base_json_name)
        
        self.detail_reasoning.configure(state="normal")
        self.detail_reasoning.delete("1.0", "end")
        
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    score = data.get("virality_score", "N/A")
                    reasoning = data.get("reasoning", "No AI reasoning notes provided.")
                    self.detail_score.configure(text=f"Virality Score: {score}/10")
                    self.detail_reasoning.insert("1.0", reasoning)
            except Exception:
                self.detail_score.configure(text="Score: N/A")
                self.detail_reasoning.insert("1.0", "Error decoding metadata JSON.")
        else:
            self.detail_score.configure(text="Virality Score: N/A")
            self.detail_reasoning.insert("1.0", "No metadata sidecar found for this video clip.")
        self.detail_reasoning.configure(state="disabled")
        
        # Poster Preview
        thumb_name = filename.replace("_vertical.mp4", ".mp4").replace(".mp4", ".jpg")
        thumb_path = os.path.join(directory, thumb_name)
        if os.path.exists(thumb_path):
            try:
                pil_img = Image.open(thumb_path)
                width, height = pil_img.size
                ratio = min(560 / width, 240 / height)
                new_w, new_h = int(width * ratio), int(height * ratio)
                large_clip_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(new_w, new_h))
                self.detail_thumbnail.configure(image=large_clip_img, text="")
            except Exception:
                self.detail_thumbnail.configure(image=None, text="Failed to load thumbnail preview")
        else:
            self.detail_thumbnail.configure(image=None, text="No thumbnail generated")
            
        if hasattr(os, 'startfile'):
            self.play_clip_btn.configure(state="normal", command=lambda: os.startfile(mp4_path))
            self.open_folder_btn.configure(state="normal", command=lambda: subprocess.run(['explorer', '/select,', os.path.abspath(mp4_path)], timeout=5))

    # ==============================================================================
    # BATCH VIDEO HIGHLIGHT PROCESSOR
    # ==============================================================================
    def browse_local_file(self):
        file_paths = filedialog.askopenfilenames(
            title="Select Local Video File(s)",
            filetypes=[("Video Files", "*.mp4 *.mkv *.avi *.mov *.flv *.ts *.m4v")]
        )
        if file_paths:
            self.file_input.delete(0, "end")
            self.file_input.insert(0, ";".join(file_paths))

    def cancel_manual_process(self):
        self.cancel_requested = True
        self.cancel_btn.configure(state="disabled", text="Aborting...")
        self.log_to_console("🛑 Cancellation requested. Terminating active tasks...", source="manual")

    def start_manual_process(self, event=None):
        input_val = self.file_input.get().strip()
        if input_val:
            self.cancel_requested = False
            self.process_btn.configure(state="disabled", text="Processing...")
            self.local_file_btn.configure(state="disabled")
            self.cancel_btn.configure(state="normal", text="🛑 Cancel")
            self.manual_progress.start()
            self.clear_console()
            
            threading.Thread(target=self._process_video_thread, args=(input_val,), daemon=True).start()
        else:
            self._show_transient_button_state(self.process_btn, "❌ Input Required", temp_color=THEME["accent_danger"])

    def _process_video_thread(self, input_val):
        try:
            profile = self.profile_dropdown.get()
            queue = [item.strip() for item in input_val.split(";") if item.strip()]
            
            for index, item in enumerate(queue):
                if self.cancel_requested: break
                
                if len(queue) > 1:
                    self.log_to_console(f"\n📦 BATCH QUEUE: Processing video {index + 1} of {len(queue)}...", source="manual")
                if os.path.exists(item):
                    self.log_to_console(f"📁 Source file detected: {item}", source="manual")
                    editor.process_video(
                        item, prompt_profile=profile, 
                        logger=lambda msg: self.log_to_console(msg, source="manual"), 
                        is_cancelled=lambda: self.cancel_requested
                    )
                else:
                    self.log_to_console(f"❌ File not found: {item}", source="manual")
                    
            if self.cancel_requested:
                self.log_to_console("🛑 Video processing cancelled by user.", source="manual")
            else:
                self.log_to_console("🏁 ALL VIDEO PIPELINE TASKS COMPLETED!", source="manual")
                self.send_discord_alert("Queue Execution Complete")
                self._show_transient_button_state(self.process_btn, "✅ Complete!", temp_color=THEME["accent_success"], duration=3000)
                
        except Exception as e:
            self.log_to_console(f"❌ Execution Error: {e}", source="manual")
        finally:
            self.after(0, lambda: [
                self.process_btn.configure(state="normal"), 
                self.local_file_btn.configure(state="normal"),
                self.cancel_btn.configure(state="disabled", text="🛑 Cancel"),
                self.manual_progress.stop(),
                self.manual_status_label.configure(text="● Status: Ready", text_color=THEME["accent_success"]),
                self.populate_gallery()
            ])

    # ==============================================================================
    # SYSTEM TRAY
    # ==============================================================================
    def minimize_to_tray(self):
        self.withdraw() 
        image = Image.new('RGB', (64, 64), color=(37, 99, 235))
        menu = pystray.Menu(
            pystray.MenuItem('Show Generator', self.show_window),
            pystray.MenuItem('Quit', self.quit_window)
        )
        self.tray_icon = pystray.Icon("jBahrsClipGen", image, "Clip Generator", menu)
        self.tray_icon.run_detached()

    def show_window(self, icon, item):
        if self.tray_icon: 
            self.tray_icon.stop()
        self.after(0, self.deiconify) 

    def quit_window(self, icon, item):
        if self.tray_icon: 
            self.tray_icon.stop()
        self.after(0, self.destroy)   


if __name__ == "__main__":
    app = ClipGenApp()
    app.mainloop()