import json
import os
import shutil
import stat
from typing import Any, Dict, Optional

from src.utils.paths import get_config_path

CONFIG_FILE = get_config_path()
OLD_LOCAL_CONFIG = "config.json"

def get_default_config() -> Dict[str, Any]:
    return {
        "openai": {
            "api_key": "", 
            "chat_model": "gpt-4o", 
            "whisper_model": "medium",
            "whisper_language": "Auto-Detect",
            "base_url": ""
        },
        "deepseek": {
            "api_key": "",
            "base_url": "https://api.deepseek.com"
        },
        "anthropic": {
            "api_key": ""
        },
        "xai": {
            "api_key": ""
        },
        "google": {
            "api_key": ""
        },
        "integrations": {
            "discord_webhook": ""
        },
        "settings": {
            "clips_dir": "",
            "vr_stabilization": False,
            "vertical_export": False,
            "vertical_mode": "Standard Center Crop",
            "crop_x": "0",
            "crop_y": "0",
            "crop_w": "400",
            "crop_h": "225",
            "hardware_encoding": False,
            "audio_downmix": True,
            "audio_peak_detection": True,
            "combat_detection": True
        },
        "active_ai_provider": "openai",
        "openai_model": "gpt-4o",
        "anthropic_model": "claude-3-5-sonnet-latest",
        "google_model": "gemini-3-flash",
        "xai_model": "grok-2-latest",
        "deepseek_model": "deepseek-v4-flash",
        "cached_models": [
            "gpt-4o", "gpt-4o-mini", 
            "gemini-3-flash", "gemini-3-pro",
            "gemini-2.0-flash", "gemini-2.0-pro",
            "claude-3-5-sonnet-latest", "claude-3-5-haiku-latest",
            "grok-2-latest", "grok-2-mini",
            "deepseek-v4-flash", "deepseek-v4-pro"
        ],
        "prompts": {
            "active_profile": "Default", 
            "profiles": {
                "Default": (
                    "You are an elite Video Editor and Content Strategist specializing in high-engagement gaming highlights, viral clips, and compilation reels.\n\n"
                    "You are analyzing the complete, timestamped transcript of a raw gaming VOD. Each line begins with an exact timestamp range like [14.5s - 18.2s], with optional [LOUDNESS: XX%] and [ACTION: COMBAT] tags.\n\n"
                    "### CORE DIRECTIVE: HIGHLIGHT EXTRACTION (SCORE >= 7)\n"
                    "- Extract all genuine highlights that meet Score 7 or higher (Score 7, 8, 9, 10). Capture every moment that truly deserves a clip, but do not lower your quality threshold for mundane gameplay.\n"
                    "- NO CONTIGUOUS TILING: Never slice the transcript into continuous back-to-back chunks (e.g. 30–60s, 60–90s, 90–120s). Every clip must be an isolated highlight with a clear beginning, climax, and end, separated by unclipped baseline content.\n"
                    "- Score 7 threshold: A valid highlight must feature a distinct punchline, notable banter, memorable laugh, clean clutch kill, or funny mishap. Filter out routine chatter, quiet looting, and mundane travel (Scores 1–6).\n\n"
                    "### TWO PRIMARY HIGHLIGHT CATEGORIES:\n\n"
                    "1. COMEDY, BANTER & ROASTING (Typical duration: 15–45 seconds)\n"
                    "- Friend group chemistry, hilarious arguments, clever roasts, absurd logic, dumb decisions with instant karma, panic screaming, dark humor, and uncontrollable laughing fits.\n"
                    "- SMART TRIMMING: Start right where the setup question/premise is uttered (2–4s before the trigger). Do NOT include boring walking or unrelated chatter. Capture setup -> punchline -> climax laughter. End 2–3s after the laugh settles.\n"
                    "- CONTEXT & SPEECH PRESERVATION: NEVER cut in the middle of a word or sentence. Always ensure the opening sentence starts cleanly from its first syllable, and the closing sentence/reaction finishes completely.\n\n"
                    "2. EPIC GAMEPLAY, KILLSTREAKS & CLUTCHES (Typical duration: 25–65 seconds)\n"
                    "- Outstanding skill, multi-man killstreaks, 1vX clutch rounds, insane flick shots, desperate escapes, boss battle climaxes, physics chaos, and catastrophic team wipes.\n"
                    "- SMART TRIMMING: Start 3–5s before combat/action begins. Capture the FULL continuous sequence of action (do not chop a 3-kill streak into separate clips). End 2–4s after the combat resolves, objective succeeds, or players react.\n\n"
                    "### STRICT SELECTION PRINCIPLES:\n"
                    "- NO ADJACENT CHUNKING: Once a highlight resolves, stop clipping immediately. The next clip must be a separate, independent peak event.\n"
                    "- NO CUT OFF SPEECH: Ensure every dialogue phrase inside the clip is whole and complete — never chop sentences in half.\n"
                    "- NO DEAD AIR: Cut all silent wandering, mundane looting, and uninteresting travel.\n"
                    "- 'CLIP IT' OVERRIDE: If any player explicitly commands 'clip it', 'clip that', or equivalent in the transcript's language, extract the moment immediately with virality_score = 10.\n\n"
                    "### SCORING SYSTEM:\n"
                    "- Scores 1–6: Mundane gameplay, boring silence, flat casual talk, weak jokes, routine single kills -> STRICTLY IGNORE (Do NOT include in output).\n"
                    "- Score 7: Solid highlight (genuinely funny punchline, clever roast, clean clutch kill, memorable laugh) -> EXTRACT THIS.\n"
                    "- Score 8: Great highlight (multi-kill streak, loud laughing fit, hilarious team betrayal/fail) -> EXTRACT THIS.\n"
                    "- Score 9: Peak highlight (major 1vX clutch, legendary comedy sequence, escalating disaster) -> EXTRACT THIS.\n"
                    "- Score 10: Iconic / 'clip it' moment / unforgettable peak of the entire stream -> EXTRACT THIS.\n\n"
                    "### LANGUAGE REQUIREMENT FOR OUTPUT:\n"
                    "- The 'reasoning' field MUST be written in the SAME natural language as the transcript (e.g. if transcript is in Russian, write reasoning in Russian; if in English, write in English).\n\n"
                    "### OUTPUT FORMAT:\n"
                    "Return strictly valid JSON with a single 'clips' array. No markdown code blocks, no extra commentary.\n"
                    "Each clip object must contain:\n"
                    "- 'start_time': float (exact timestamp in seconds)\n"
                    "- 'end_time': float (exact timestamp in seconds)\n"
                    "- 'virality_score': integer (7 to 10)\n"
                    "- 'reasoning': concise summary (1-2 sentences in transcript's language explaining why this moment is a highlight)"
                )
            }
        }
    }

def load_config(filepath: Optional[str] = None) -> Dict[str, Any]:
    target_file = filepath or get_config_path()

    if not filepath and os.path.exists(OLD_LOCAL_CONFIG) and not os.path.exists(target_file):
        try:
            shutil.move(OLD_LOCAL_CONFIG, target_file)
        except Exception as e:
            print(f"Failed to migrate old config file: {e}")

    if os.path.exists(target_file):
        with open(target_file, 'r', encoding='utf-8') as f:
            try:
                cfg = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Failed to decode config file: {e}")
                return get_default_config()
            
            # Inject new settings if they don't exist in saved config
            settings = cfg.setdefault("settings", {})
            settings.setdefault("hardware_encoding", False)
            settings.setdefault("audio_downmix", True)
            settings.setdefault("audio_peak_detection", True)
            settings.setdefault("combat_detection", True)
            
            openai_cfg = cfg.setdefault("openai", {})
            openai_cfg.setdefault("base_url", "")
            openai_cfg.setdefault("whisper_model", "medium")
            openai_cfg.setdefault("whisper_language", "Auto-Detect")
            
            cfg.setdefault("deepseek", {"api_key": "", "base_url": "https://api.deepseek.com"})
            cfg.setdefault("anthropic", {"api_key": ""})
            cfg.setdefault("xai", {"api_key": ""})
            cfg.setdefault("integrations", {"discord_webhook": ""})
            
            # Migration/Defaults for new multi-provider UI
            cfg.setdefault("active_ai_provider", "openai")
            cfg.setdefault("openai_model", "gpt-4o")
            cfg.setdefault("deepseek_model", "deepseek-v4-flash")
            cfg.setdefault("anthropic_model", "claude-3-5-sonnet-latest")
            cfg.setdefault("google_model", "gemini-3-flash")
            cfg.setdefault("xai_model", "grok-2-latest")
            
            # Migrate retired DeepSeek models (deepseek-chat, deepseek-reasoner -> deepseek-v4-flash)
            if cfg.get("deepseek_model") in ["deepseek-chat", "deepseek-reasoner"]:
                cfg["deepseek_model"] = "deepseek-v4-flash"
            if cfg.get("openai_model") in ["deepseek-chat", "deepseek-reasoner"]:
                cfg["openai_model"] = "deepseek-v4-flash"
            
            # Migrate old chat_model to the correct field
            old_chat_model = cfg.get("openai", {}).get("chat_model")
            if old_chat_model:
                if "gemini" in old_chat_model:
                    cfg["google_model"] = old_chat_model
                    cfg["active_ai_provider"] = "google"
                elif "claude" in old_chat_model:
                    cfg["anthropic_model"] = old_chat_model
                    cfg["active_ai_provider"] = "anthropic"
                elif "grok" in old_chat_model:
                    cfg["xai_model"] = old_chat_model
                    cfg["active_ai_provider"] = "xai"
                elif "deepseek" in old_chat_model:
                    if old_chat_model in ["deepseek-chat", "deepseek-reasoner"]:
                        cfg["deepseek_model"] = "deepseek-v4-flash"
                    else:
                        cfg["deepseek_model"] = old_chat_model
                    cfg["active_ai_provider"] = "deepseek"
                else:
                    cfg["openai_model"] = old_chat_model
                    cfg["active_ai_provider"] = "openai"
            
            # Update cached_models
            cached_models = cfg.get("cached_models", [])
            if any(m in ["deepseek-chat", "deepseek-reasoner"] for m in cached_models):
                updated_models = [m for m in cached_models if m not in ["deepseek-chat", "deepseek-reasoner"]]
                for model in ["deepseek-v4-flash", "deepseek-v4-pro"]:
                    if model not in updated_models:
                        updated_models.append(model)
                cfg["cached_models"] = updated_models
            
            # Load default profiles for prompt management
            default_prompts = get_default_config()["prompts"]
            prompts = cfg.setdefault("prompts", default_prompts)
            profiles = prompts.setdefault("profiles", default_prompts["profiles"])
            
            # Always sync protected Default profile to the latest built-in codebase prompt
            profiles["Default"] = default_prompts["profiles"]["Default"]
            if prompts.get("active_profile") not in profiles:
                prompts["active_profile"] = "Default"
            
            return cfg
            
    # Default Config
    return get_default_config()

def save_config(config: Dict[str, Any], filepath: Optional[str] = None) -> None:
    target_file = filepath or get_config_path()
    os.makedirs(os.path.dirname(os.path.abspath(target_file)), exist_ok=True)
    
    # Open file descriptor with restrictive permissions
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    mode = stat.S_IRUSR | stat.S_IWUSR

    fd = os.open(target_file, flags, mode)
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    # Ensure existing files also have restricted permissions
    try:
        os.chmod(target_file, mode)
    except OSError:
        pass
