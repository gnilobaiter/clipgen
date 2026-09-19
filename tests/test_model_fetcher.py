import pytest
from unittest.mock import MagicMock

from src.services import model_fetcher


@pytest.mark.parametrize("model_id,expected", [
    ("gpt-4o", True),
    ("claude-3-5-sonnet", True),
    ("gemini-1.5-pro", True),
    ("openrouter/google/gemini-pro", True),
    ("deepseek-chat", True),
    ("deepseek-v4-flash", True),
    ("deepseek-v4-pro", True),
    ("text-embedding-3-small", False),
    ("text-embedding-ada-002", False),
    ("vision-only-model", False),
    ("whisper-1", False),
    ("tts-1-hd", False),
    ("dall-e-3", False),
    ("unknown-other", False),
])
def test_is_chat_model(model_id, expected):
    assert model_fetcher.is_chat_model(model_id) is expected


def test_fetch_openai_models_empty_key():
    assert model_fetcher.fetch_openai_models("") == []


def test_fetch_openai_models_missing_package(monkeypatch):
    monkeypatch.setattr("src.services.model_fetcher.OpenAI", None)
    assert model_fetcher.fetch_openai_models("some_key") == []


def test_fetch_openai_models_success(monkeypatch):
    mock_client = MagicMock()
    mock_model1 = MagicMock(id="gpt-4o")
    mock_model2 = MagicMock(id="text-embedding-3")
    mock_client.models.list.return_value = [mock_model1, mock_model2]

    monkeypatch.setattr("src.services.model_fetcher.OpenAI", lambda **kwargs: mock_client)

    models = model_fetcher.fetch_openai_models("fake_key")
    assert models == ["gpt-4o"]


def test_fetch_openai_models_fallback_filter(monkeypatch):
    mock_client = MagicMock()
    mock_model1 = MagicMock(id="custom-gpt-4-finetuned")
    mock_client.models.list.return_value = [mock_model1]

    monkeypatch.setattr("src.services.model_fetcher.OpenAI", lambda **kwargs: mock_client)

    models = model_fetcher.fetch_openai_models("fake_key")
    assert models == ["custom-gpt-4-finetuned"]


def test_fetch_openai_models_error(monkeypatch):
    def bad_client(**kwargs):
        raise RuntimeError("API Offline")

    monkeypatch.setattr("src.services.model_fetcher.OpenAI", bad_client)
    models = model_fetcher.fetch_openai_models("fake_key")
    assert models == []


def test_fetch_deepseek_models_empty_key():
    assert model_fetcher.fetch_deepseek_models("") == model_fetcher.MAIN_MODELS["deepseek"]


def test_fetch_deepseek_models_success(monkeypatch):
    mock_client = MagicMock()
    mock_model1 = MagicMock(id="deepseek-v4-flash")
    mock_model2 = MagicMock(id="deepseek-v4-pro")
    mock_model3 = MagicMock(id="text-embedding-ds")
    mock_client.models.list.return_value = [mock_model1, mock_model2, mock_model3]

    monkeypatch.setattr("src.services.model_fetcher.OpenAI", lambda **kwargs: mock_client)

    models = model_fetcher.fetch_deepseek_models("fake_key")
    assert models == ["deepseek-v4-flash", "deepseek-v4-pro"]


def test_fetch_deepseek_models_error(monkeypatch):
    def bad_client(**kwargs):
        raise RuntimeError("DeepSeek Timeout")

    monkeypatch.setattr("src.services.model_fetcher.OpenAI", bad_client)
    models = model_fetcher.fetch_deepseek_models("fake_key")
    assert models == model_fetcher.MAIN_MODELS["deepseek"]


def test_fetch_anthropic_models_empty_key():
    assert model_fetcher.fetch_anthropic_models("") == []


def test_fetch_anthropic_models_missing_package(monkeypatch):
    monkeypatch.setattr("src.services.model_fetcher.anthropic", None)
    assert model_fetcher.fetch_anthropic_models("some_key") == model_fetcher.MAIN_MODELS["anthropic"]


def test_fetch_anthropic_models_success(monkeypatch):
    mock_client = MagicMock()
    mock_model1 = MagicMock(id="claude-3-5-sonnet-latest")
    mock_client.models.list.return_value = [mock_model1]

    monkeypatch.setattr("anthropic.Anthropic", lambda **kwargs: mock_client)
    models = model_fetcher.fetch_anthropic_models("fake_key")
    assert models == ["claude-3-5-sonnet-latest"]


def test_fetch_anthropic_models_fallback_filter(monkeypatch):
    mock_client = MagicMock()
    mock_model1 = MagicMock(id="custom-claude-3-community")
    mock_client.models.list.return_value = [mock_model1]

    monkeypatch.setattr("anthropic.Anthropic", lambda **kwargs: mock_client)
    models = model_fetcher.fetch_anthropic_models("fake_key")
    assert models == ["custom-claude-3-community"]


def test_fetch_anthropic_models_error(monkeypatch):
    def bad_client(**kwargs):
        raise RuntimeError("Auth Error")

    monkeypatch.setattr("anthropic.Anthropic", bad_client)
    models = model_fetcher.fetch_anthropic_models("fake_key")
    assert models == model_fetcher.MAIN_MODELS["anthropic"]


def test_fetch_google_models_empty_key():
    assert model_fetcher.fetch_google_models("") == []


def test_fetch_google_models_missing_package(monkeypatch):
    monkeypatch.setattr("src.services.model_fetcher.genai", None)
    assert model_fetcher.fetch_google_models("some_key") == []


def test_fetch_google_models_success(monkeypatch):
    mock_client = MagicMock()
    mock_model1 = MagicMock()
    mock_model1.name = "models/gemini-2.5-flash"
    mock_client.models.list.return_value = [mock_model1]

    monkeypatch.setattr("google.genai.Client", lambda **kwargs: mock_client)
    models = model_fetcher.fetch_google_models("fake_key")
    assert models == ["gemini-2.5-flash"]


def test_fetch_google_models_error(monkeypatch):
    def bad_client(**kwargs):
        raise RuntimeError("Google API Error")

    monkeypatch.setattr("google.genai.Client", bad_client)
    models = model_fetcher.fetch_google_models("fake_key")
    assert models == model_fetcher.MAIN_MODELS["google"]


def test_fetch_xai_models_empty_key():
    assert model_fetcher.fetch_xai_models("") == []


def test_fetch_xai_models_success(monkeypatch):
    mock_client = MagicMock()
    mock_model1 = MagicMock(id="grok-4.3")
    mock_client.models.list.return_value = [mock_model1]

    monkeypatch.setattr("src.services.model_fetcher.OpenAI", lambda **kwargs: mock_client)
    models = model_fetcher.fetch_xai_models("fake_key")
    assert models == ["grok-4.3"]


def test_fetch_all_available_models(monkeypatch):
    config = {
        "openai": {"api_key": "sk-op", "base_url": ""},
        "deepseek": {"api_key": "sk-ds"},
        "anthropic": {"api_key": "sk-ant"},
        "xai": {"api_key": "sk-xai"},
        "google": {"api_key": "AIzaSy"}
    }

    monkeypatch.setattr(model_fetcher, "fetch_openai_models", lambda *args, **kwargs: ["gpt-4o"])
    monkeypatch.setattr(model_fetcher, "fetch_deepseek_models", lambda *args, **kwargs: ["deepseek-v4-flash"])
    monkeypatch.setattr(model_fetcher, "fetch_anthropic_models", lambda *args, **kwargs: ["claude-3-5-sonnet-latest"])
    monkeypatch.setattr(model_fetcher, "fetch_xai_models", lambda *args, **kwargs: ["grok-2-latest"])
    monkeypatch.setattr(model_fetcher, "fetch_google_models", lambda *args, **kwargs: ["gemini-2.5-flash"])

    all_models = model_fetcher.fetch_all_available_models(config)
    assert "gpt-4o" in all_models
    assert "deepseek-v4-flash" in all_models
    assert "claude-3-5-sonnet-latest" in all_models
    assert "grok-2-latest" in all_models
    assert "gemini-2.5-flash" in all_models
