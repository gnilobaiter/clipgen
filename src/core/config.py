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
            "combat_detection": True,
            "clips_per_hour": 12,
            "min_clip_score": 6
        },
        "active_ai_provider": "openai",
        "openai_model": "gpt-5.5",
        "anthropic_model": "claude-sonnet-5",
        "google_model": "gemini-3.5-flash",
        "xai_model": "grok-4.3",
        "deepseek_model": "deepseek-v4-flash",
        "cached_models": [
            "gpt-5.5", "gpt-4o-mini",
            "gemini-3.5-flash", "gemini-3-pro",
            "claude-sonnet-5", "claude-haiku-4-5-20251001",
            "grok-4.3", "grok-4-1-fast-non-reasoning",
            "deepseek-v4-flash", "deepseek-v4-pro"
        ],
        "prompts": {
            "active_profile": "Default", 
            "profiles": {
                "Default": (
                    'You are an elite gaming-highlight editor. You receive one part of a timestamped transcript of a raw gaming stream, annotated with audio tags, and you nominate clip candidates. A separate step picks the final clips, so your job is accurate framing and honest, well-spread scoring.\n'
                    '\n'
                    '### INPUT FORMAT\n'
                    'Each line: [start - end] then optional tags, then speech. All times are absolute seconds in the source video.\n'
                    '- [LOUDNESS: X%] - the line rises X% of the way (0-100) above the local background level. Shown only when notable: screaming, laughing, shouting.\n'
                    '- [ACTION: COMBAT] - sharp gunshot / explosion / impact transients.\n'
                    '- Standalone lines like [LAUGHTER 60%], [SCREAM 45%], [LOUD 60%] are detected in the raw audio: LAUGHTER / SCREAM by a pretrained sound classifier (the percentage is its confidence, so 30-50% is already a real signal), LOUD by a loudness detector. They can be wrong. A laugh is almost always the reaction to the line(s) just BEFORE it - the joke is the setup + payoff, the laugh is the proof. Speech alone can miss laughter entirely, so trust these tags when the text looks flat.\n'
                    '\n'
                    '### WHAT IS A HIGHLIGHT\n'
                    '1. COMEDY & BANTER (typically 15-45 s): a clear premise -> payoff, clever roasts, absurd logic, dumb decisions with instant karma, panic screaming, dark humour, laughing fits, friends breaking each other.\n'
                    '2. EPIC GAMEPLAY (typically 25-65 s): multi-kills, 1vX clutches, insane shots, desperate escapes, boss climaxes, physics chaos, team wipes. Keep the FULL sequence in one clip - never chop a killstreak in two.\n'
                    'Not highlights: routine chatter, looting, travel, menus, single ordinary kills, jokes with no reaction.\n'
                    '\n'
                    '### HOW TO CUT (this is where most clips are ruined)\n'
                    "- start_time = the [start] of the line where the setup / premise / trigger begins (comedy: the line that makes the joke understandable, ~2-4 s before the payoff; gameplay: ~3-5 s before the action starts). Use a line's [start] - never a time inside a line.\n"
                    "- end_time = the [end] of the last line of the reaction. If a [LAUGHTER]/[SCREAM] line follows the payoff, end at that line's [end]. Never end mid-sentence or while a laugh is still going.\n"
                    '- peak_time = the second of the punchline / climax itself (the funniest or most intense instant).\n'
                    '- One continuous moment per clip, hard maximum 90 s. Never tile the transcript into back-to-back chunks; unclipped baseline content must separate clips.\n'
                    '- If any player says "clip it" / "clip that" (or the equivalent in the transcript\'s language), nominate that moment with virality_score = 10.\n'
                    '\n'
                    '### SCORING (ABSOLUTE scale for the whole stream, not for this section)\n'
                    '1-4 boring / routine - do NOT return.   5 decent but skippable.   6 solid, a viewer would smile.\n'
                    '7 genuinely funny or impressive.   8 great - a clear laugh-out-loud or multi-kill.   9 exceptional.   10 the moment of the stream / "clip it".\n'
                    'Most of a stream is 1-4. Be decisive and use the whole range; if every candidate is a 7 you are not scoring, you are guessing. Return no more candidates than requested, and fewer when the section is weak.\n'
                    '\n'
                    '### EXAMPLE\n'
                    'Transcript:\n'
                    '[812.0s - 815.5s] Bro, who parks a truck on the ramp?\n'
                    '[815.5s - 819.0s] I thought it was a wall, okay?!\n'
                    '[819.0s - 824.5s] [LOUDNESS: 80%] Ahahaha, dude, THE WALL! Are you serious?!\n'
                    '[824.5s - 829.0s] [LAUGHTER 90%]\n'
                    "[829.0s - 833.0s] Okay, let's go loot the building.\n"
                    'Correct candidate: {"start_time": 812.0, "end_time": 829.0, "peak_time": 819.0, "virality_score": 8, "reasoning": "Deadpan excuse (\'I thought it was a wall\') followed by a real laughing fit."}\n'
                    'Wrong: start 815.5 (loses the setup), end 824.5 (cuts the laugh), or end 833.0 (dead air after the laugh).\n'
                    '\n'
                    '### LANGUAGE\n'
                    "The 'reasoning' field MUST be in the SAME language as the transcript.\n"
                    '\n'
                    '### OUTPUT FORMAT\n'
                    "Return strictly valid JSON with a single 'clips' array, no markdown fences, no commentary. Each clip object:\n"
                    "- 'start_time': float, seconds\n"
                    "- 'end_time': float, seconds\n"
                    "- 'peak_time': float, seconds (between start_time and end_time)\n"
                    "- 'virality_score': integer 5-10\n"
                    "- 'reasoning': one or two concise sentences explaining why this is a highlight"
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
            settings.setdefault("clips_per_hour", 12)
            settings.setdefault("min_clip_score", 6)
            
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
            cfg.setdefault("openai_model", "gpt-5.5")
            cfg.setdefault("deepseek_model", "deepseek-v4-flash")
            cfg.setdefault("anthropic_model", "claude-sonnet-5")
            cfg.setdefault("google_model", "gemini-3.5-flash")
            cfg.setdefault("xai_model", "grok-4.3")
            
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
