import json
import pytest

from src.core.config import (
    get_default_config,
    load_config,
    save_config,
)


def test_get_default_config():
    config = get_default_config()
    assert isinstance(config, dict)

    expected_keys = [
        "openai", "deepseek", "anthropic", "xai",
        "google", "integrations", "settings", "prompts"
    ]
    for key in expected_keys:
        assert key in config
    assert "youtube" not in config
    assert "twitch" not in config
    assert "auto_scheduler" not in config

    assert config["openai"]["chat_model"] == "gpt-4o"
    assert config["openai"]["whisper_model"] == "medium"
    assert config["openai"]["whisper_language"] == "Auto-Detect"
    assert config["deepseek"]["base_url"] == "https://api.deepseek.com"
    assert config["prompts"]["active_profile"] == "Default"
    assert "Default" in config["prompts"]["profiles"]

    default_prompt = config["prompts"]["profiles"]["Default"]
    assert "COMEDY & BANTER" in default_prompt
    assert "EPIC GAMEPLAY" in default_prompt
    assert "HOW TO CUT" in default_prompt
    assert "peak_time" in default_prompt
    assert "LANGUAGE" in default_prompt
    assert "virality_score" in default_prompt
    assert config["settings"]["clips_per_hour"] == 12
    assert config["settings"]["min_clip_score"] == 6



def test_save_config_creates_file(tmp_path):
    temp_config = tmp_path / "test_config.json"
    config = get_default_config()
    save_config(config, filepath=str(temp_config))
    assert temp_config.exists()
    loaded = json.loads(temp_config.read_text(encoding="utf-8"))
    assert loaded == config


def test_save_config_creates_nested_directories(tmp_path):
    nested_config = tmp_path / "subdir1" / "subdir2" / "config.json"
    config = get_default_config()
    save_config(config, filepath=str(nested_config))
    assert nested_config.exists()


def test_save_config_error_handling(tmp_path, monkeypatch):
    def bad_open(*args, **kwargs):
        raise PermissionError("Access Denied")

    monkeypatch.setattr("os.open", bad_open)
    temp_target = str(tmp_path / "some_path.json")
    with pytest.raises(PermissionError):
        save_config(get_default_config(), filepath=temp_target)


def test_load_config_no_file(tmp_path):
    temp_config = tmp_path / "nonexistent.json"
    config = load_config(filepath=str(temp_config))
    assert config == get_default_config()


def test_load_config_migration(tmp_path, monkeypatch):
    old_cfg = tmp_path / "old_config.json"
    new_cfg = tmp_path / "new_config.json"
    old_cfg.write_text(json.dumps({"test": "data"}), encoding="utf-8")
    monkeypatch.setattr("src.core.config.OLD_LOCAL_CONFIG", str(old_cfg))
    cfg = load_config(filepath=str(new_cfg))
    assert cfg is not None


def test_load_config_invalid_json(tmp_path, monkeypatch):
    bad_cfg = tmp_path / "bad.json"
    bad_cfg.write_text("{invalid_json: true}", encoding="utf-8")
    printed = []
    monkeypatch.setattr("builtins.print", lambda msg: printed.append(msg))
    config = load_config(filepath=str(bad_cfg))
    assert config == get_default_config()
    assert any("Failed to decode config file" in msg for msg in printed)


def test_load_config_deepseek_migration(tmp_path):
    cfg_path = tmp_path / "deepseek_mig.json"
    old_cfg_data = {
        "openai": {"chat_model": "deepseek-chat"},
        "deepseek_model": "deepseek-chat",
        "cached_models": ["gpt-4o", "deepseek-chat", "deepseek-reasoner"]
    }
    cfg_path.write_text(json.dumps(old_cfg_data), encoding="utf-8")

    config = load_config(filepath=str(cfg_path))
    assert config["deepseek_model"] == "deepseek-v4-flash"
    assert config["active_ai_provider"] == "deepseek"
    assert "deepseek-v4-flash" in config["cached_models"]
    assert "deepseek-v4-pro" in config["cached_models"]
    assert "deepseek-chat" not in config["cached_models"]
    assert "deepseek-reasoner" not in config["cached_models"]


def test_load_config_populates_missing_keys(tmp_path):
    cfg_path = tmp_path / "partial.json"
    cfg_path.write_text(json.dumps({"openai": {"api_key": "abc"}}), encoding="utf-8")

    config = load_config(filepath=str(cfg_path))
    assert config["openai"]["api_key"] == "abc"
    assert "settings" in config
    assert "prompts" in config
    assert "deepseek" in config
@pytest.mark.parametrize("old_model,expected_provider,expected_key", [
    ("gemini-1.5-flash", "google", "google_model"),
    ("claude-3-opus", "anthropic", "anthropic_model"),
    ("grok-beta", "xai", "xai_model"),
    ("deepseek-coder", "deepseek", "deepseek_model"),
    ("gpt-4-turbo", "openai", "openai_model"),
])
def test_load_config_old_chat_model_migration(tmp_path, old_model, expected_provider, expected_key):
    cfg_path = tmp_path / "old_chat_model.json"
    old_data = {
        "openai": {"chat_model": old_model}
    }
    cfg_path.write_text(json.dumps(old_data), encoding="utf-8")

    config = load_config(filepath=str(cfg_path))
    assert config["active_ai_provider"] == expected_provider
    assert config[expected_key] == old_model
