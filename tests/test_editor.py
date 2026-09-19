import json
import os
import subprocess
import pytest
from unittest.mock import MagicMock
import numpy as np

from src.core import editor


# ==============================================================================
# SANITIZE & AUDIO STREAMS
# ==============================================================================
def test_sanitize_filename():
    assert editor.sanitize_filename("Awesome/Clip:With*Bad?Chars|<Name>") == "AwesomeClipWithBadCharsName"


def test_count_audio_streams_multi(monkeypatch):
    mock_run = MagicMock()
    mock_run.return_value.returncode = 0
    mock_run.return_value.stdout = "0\n1\n2\n"
    monkeypatch.setattr(subprocess, "run", mock_run)

    assert editor._count_audio_streams("obs_video.mp4") == 3


def test_count_audio_streams_error(monkeypatch):
    def bad_run(*args, **kwargs):
        raise RuntimeError("ffprobe not found")

    monkeypatch.setattr(subprocess, "run", bad_run)
    assert editor._count_audio_streams("obs_video.mp4") == 1


# ==============================================================================
# AUDIO EXTRACTION & ANALYSIS
# ==============================================================================
def test_extract_audio_hidden_success_linux(monkeypatch):
    monkeypatch.setattr("os.name", "posix")

    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_stdout = np.zeros(10, dtype=np.int16).tobytes()
    mock_process.communicate.return_value = (mock_stdout, b"")

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: mock_process)

    file_path = "test_audio.mp4"
    result = editor.extract_audio_hidden(file_path)

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert len(result) == 10


def test_extract_audio_hidden_success_windows(monkeypatch):
    monkeypatch.setattr("os.name", "nt")

    class DummyStartupInfo:
        def __init__(self):
            self.dwFlags = 0
            self.wShowWindow = 0

    monkeypatch.setattr(subprocess, "STARTUPINFO", DummyStartupInfo, raising=False)
    monkeypatch.setattr(subprocess, "STARTF_USESHOWWINDOW", 1, raising=False)
    monkeypatch.setattr(subprocess, "SW_HIDE", 0, raising=False)

    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_stdout = np.zeros(10, dtype=np.int16).tobytes()
    mock_process.communicate.return_value = (mock_stdout, b"")

    recorded_kwargs = {}
    def mock_popen(*args, **kwargs):
        recorded_kwargs.update(kwargs)
        return mock_process

    monkeypatch.setattr(subprocess, "Popen", mock_popen)

    file_path = "test_audio_win.mp4"
    result = editor.extract_audio_hidden(file_path)
    assert isinstance(result, np.ndarray)
    assert "startupinfo" in recorded_kwargs


def test_extract_audio_hidden_failure(monkeypatch):
    mock_process = MagicMock()
    mock_process.returncode = 1
    mock_process.communicate.return_value = (b"", b"Mock FFmpeg Error")

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: mock_process)

    with pytest.raises(RuntimeError, match="FFmpeg audio extraction failed"):
        editor.extract_audio_hidden("bad_file.mp4")


def test_enhance_segments_with_audio_analysis():
    segments = [
        {"start": 0.0, "end": 4.0, "text": "calm dialogue"},
        {"start": 4.0, "end": 6.0, "text": "screaming explosion"}
    ]
    # 6 seconds of 16kHz audio: quiet background, then a loud burst with a sharp spike
    audio = np.full(96000, 0.02, dtype=np.float32)
    audio[64000:96000] = 0.8
    audio[70000:70200] = 1.0

    enhanced = editor.analyze_audio_peaks(
        audio, segments, sample_rate=16000, peak_detection=True, combat_detection=True
    )
    assert enhanced[0]["text"] == "calm dialogue"  # text is never polluted with tags
    assert enhanced[0]["loudness"] < 10
    assert enhanced[1]["loudness"] == 100


def test_analyze_audio_peaks_flags_disabled():
    audio = np.full(16000, 0.1, dtype=np.float32)
    enhanced = editor.analyze_audio_peaks(audio, [{"start": 0.0, "end": 1.0, "text": "x"}], peak_detection=False, combat_detection=False)
    assert "loudness" not in enhanced[0]
    assert enhanced[0]["is_combat"] is False


# ==============================================================================
# JSON PARSING & CACHE TESTS
# ==============================================================================
@pytest.mark.parametrize("raw_input,expected_count,expected_score", [
    ('{"clips": [{"start_time": 10.0, "end_time": 25.0, "virality_score": 9, "reasoning": "Great"}]}', 1, 9),
    ('`json\n{"clips": [{"start_time": 5.0, "end_time": 20.0, "virality_score": 8, "reasoning": "Scare"}]}\n`', 1, 8),
    ('`\n{"clips": [{"start_time": 1.0, "end_time": 15.0, "virality_score": 7, "reasoning": "Action"}]}\n`', 1, 7),
    ('Text before\n`json\n{"clips": [{"start_time": 2.0, "end_time": 18.0, "virality_score": 10, "reasoning": "Clutch"}]}\n`\nText after', 1, 10),
    ('[{"start_time": 3.0, "end_time": 12.0, "virality_score": 6, "reasoning": "Direct list"}]', 1, 6),
])
def test_parse_json_clips_valid(raw_input, expected_count, expected_score):
    clips = editor._parse_json_clips(raw_input)
    assert clips is not None
    assert len(clips) == expected_count
    assert clips[0]["virality_score"] == expected_score


@pytest.mark.parametrize("invalid_input", [
    "",
    "   ",
    None,
    "Not a json string",
    "{invalid json}",
    "`json\n{broken}\n`"
])
def test_parse_json_clips_invalid(invalid_input):
    assert editor._parse_json_clips(invalid_input) is None


def test_transcription_cache_roundtrip(tmp_path):
    cache_file = tmp_path / "test_cache.json"
    segments = [{"start": 0.0, "end": 10.0, "text": "Hello world"}]
    editor._save_cached_segments(str(cache_file), segments)

    assert cache_file.exists()
    loaded = editor._load_cached_segments(str(cache_file))
    assert loaded == segments


def test_transcription_cache_load_nonexistent():
    assert editor._load_cached_segments("nonexistent_path_9999.json") is None


def test_transcription_cache_load_corrupted(tmp_path):
    bad_cache = tmp_path / "bad_cache.json"
    bad_cache.write_text("{corrupt json", encoding="utf-8")
    assert editor._load_cached_segments(str(bad_cache)) is None


def test_get_transcription_cache_path(tmp_path):
    video = tmp_path / "sample.mp4"
    video.write_text("video")
    path = editor._get_transcription_cache_path(str(video), "base", "en", True, True)
    assert path.endswith(".json")


# ==============================================================================
# API KEY VALIDATION TESTS
# ==============================================================================
def test_validate_api_keys_all_providers():
    cfg_openai = {"active_ai_provider": "openai", "openai": {"api_key": "sk-openai"}}
    assert editor._validate_api_keys(cfg_openai, "gpt-4o", None) is True

    cfg_ds = {"active_ai_provider": "deepseek", "deepseek": {"api_key": "sk-deepseek"}}
    assert editor._validate_api_keys(cfg_ds, "deepseek-v4-flash", None) is True

    cfg_ant = {"active_ai_provider": "anthropic", "anthropic": {"api_key": "sk-ant"}}
    assert editor._validate_api_keys(cfg_ant, "claude-3-5-sonnet-latest", None) is True

    cfg_google = {"active_ai_provider": "google", "google": {"api_key": "AIzaSy"}}
    assert editor._validate_api_keys(cfg_google, "gemini-2.5-flash", None) is True

    cfg_xai = {"active_ai_provider": "xai", "xai": {"api_key": "xai-test"}}
    assert editor._validate_api_keys(cfg_xai, "grok-2-latest", None) is True


def test_validate_api_keys_missing():
    logs = []
    cfg_empty = {"active_ai_provider": "openai", "openai": {"api_key": ""}}
    assert editor._validate_api_keys(cfg_empty, "gpt-4o", logs.append) is False
    assert any("OpenAI/Custom API Key not set" in log for log in logs)


# ==============================================================================
# LLM ROUTING TESTS
# ==============================================================================
def test_deepseek_llm_routing(monkeypatch):
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = '{"clips": [{"start_time": 10.0, "end_time": 30.0, "virality_score": 8, "reasoning": "Clutch"}]}'
    mock_client.chat.completions.create.return_value = mock_resp

    monkeypatch.setattr("src.core.editor.OpenAI", lambda **kwargs: mock_client)

    config = {"openai": {"api_key": "ds_key", "base_url": "https://api.deepseek.com"}}
    segments = [{"start": 0.0, "end": 60.0, "text": "test"}]
    clips = editor._generate_clips_with_llm(segments, config, "deepseek-v4-flash", "Prompt", None)

    assert len(clips) == 1
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs.get("extra_body") == {"thinking": {"type": "disabled"}}


def test_openai_retry_on_empty(monkeypatch):
    mock_client = MagicMock()
    mock_empty = MagicMock()
    mock_empty.choices = [MagicMock()]
    mock_empty.choices[0].message.content = ""

    mock_valid = MagicMock()
    mock_valid.choices = [MagicMock()]
    mock_valid.choices[0].message.content = '{"clips": [{"start_time": 5.0, "end_time": 25.0, "virality_score": 9, "reasoning": "Banter"}]}'

    mock_client.chat.completions.create.side_effect = [mock_empty, mock_valid]
    monkeypatch.setattr("src.core.editor.OpenAI", lambda **kwargs: mock_client)
    monkeypatch.setattr("time.sleep", lambda s: None)

    config = {"openai": {"api_key": "key", "base_url": ""}}
    segments = [{"start": 0.0, "end": 60.0, "text": "test"}]
    clips = editor._generate_clips_with_llm(segments, config, "gpt-4o", "Prompt", None)

    assert len(clips) == 1
    assert mock_client.chat.completions.create.call_count == 2


def test_gemini_routing(monkeypatch):
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = '{"clips": [{"start_time": 1.0, "end_time": 15.0, "virality_score": 7, "reasoning": "Action"}]}'
    mock_client.models.generate_content.return_value = mock_resp

    monkeypatch.setattr("google.genai.Client", lambda **kwargs: mock_client)

    config = {"google": {"api_key": "gemini_key"}}
    segments = [{"start": 0.0, "end": 30.0, "text": "speech"}]
    clips = editor._generate_clips_with_llm(segments, config, "gemini-2.5-flash", "Prompt", None)

    assert len(clips) == 1


def test_anthropic_routing(monkeypatch):
    mock_client = MagicMock()
    mock_block = MagicMock()
    mock_block.text = '{"clips": [{"start_time": 2.0, "end_time": 20.0, "virality_score": 8, "reasoning": "Funny"}]}'
    mock_resp = MagicMock(content=[mock_block])
    mock_client.messages.create.return_value = mock_resp

    monkeypatch.setattr("anthropic.Anthropic", lambda **kwargs: mock_client)

    config = {"anthropic": {"api_key": "ant_key"}}
    segments = [{"start": 0.0, "end": 30.0, "text": "speech"}]
    clips = editor._generate_clips_with_llm(segments, config, "claude-3-5-sonnet-latest", "Prompt", None)

    assert len(clips) == 1


def test_xai_grok_routing(monkeypatch):
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = '{"clips": [{"start_time": 3.0, "end_time": 18.0, "virality_score": 9, "reasoning": "Clutch"}]}'
    mock_client.chat.completions.create.return_value = mock_resp

    monkeypatch.setattr("src.core.editor.OpenAI", lambda **kwargs: mock_client)

    config = {"xai": {"api_key": "xai_key"}, "active_ai_provider": "xai"}
    segments = [{"start": 0.0, "end": 30.0, "text": "speech"}]
    clips = editor._generate_clips_with_llm(segments, config, "grok-2-latest", "Prompt", None)

    assert len(clips) == 1


# ==============================================================================
# CROP & SLICE TESTS
# ==============================================================================
@pytest.mark.parametrize("mode,expected_crop", [
    ("Standard Center Crop", "crop=ih*(9/16):ih:(iw-ow)/2:0,scale=1080:1920"),
    ("Left-Third (Facecam)", "crop=ih*(9/16):ih:0:0,scale=1080:1920"),
    ("Right-Third", "crop=ih*(9/16):ih:iw-ow:0,scale=1080:1920"),
    ("Blurred Background (Portrait)", "boxblur=20:5"),
    ("Custom Coordinates", "crop=400:225:10:20,scale=1080:1920"),
])
def test_calculate_crop_filter(mode, expected_crop):
    filt = editor._calculate_crop_filter("9:16", mode, "10", "20", "400", "225")
    assert expected_crop in filt


def test_extract_clips_execution(tmp_path, monkeypatch):
    mock_run = MagicMock()
    mock_run.return_value.returncode = 0
    monkeypatch.setattr(subprocess, "run", mock_run)
    monkeypatch.setattr(os.path, "isfile", lambda p: True)

    clips_data = {
        "clips": [
            {"start_time": 10.0, "end_time": 25.0, "virality_score": 8, "reasoning": "Insane moment"}
        ]
    }

    created = editor.extract_clips(
        input_file="source.mp4",
        clips_data=clips_data,
        output_dir=str(tmp_path),
        logger=lambda msg: None
    )

    assert mock_run.call_count >= 1
    assert len(created) == 1
    json_files = list(tmp_path.glob("*.json"))
    assert len(json_files) == 1
    data = json.loads(json_files[0].read_text(encoding="utf-8"))
    assert data["virality_score"] == 8


# ==============================================================================
# FULL VIDEO PROCESS WORKFLOW TESTS
# ==============================================================================
def test_process_video_nonexistent(tmp_path):
    logs = []
    res = editor.process_video("nonexistent_video.mp4", logger=logs.append)
    assert res is False
    assert any("Source video file not found" in line for line in logs)


def test_process_video_success(tmp_path, monkeypatch):
    video = tmp_path / "gameplay.mp4"
    video.write_text("dummy")

    fake_config = {
        "active_ai_provider": "openai",
        "openai_model": "gpt-4o",
        "settings": {"clips_dir": str(tmp_path / "clips")},
        "prompts": {"profiles": {"Default": "Prompt"}}
    }
    monkeypatch.setattr("src.core.editor.config_manager.load_config", lambda *args, **kwargs: fake_config)
    monkeypatch.setattr("src.core.editor._validate_api_keys", lambda *args, **kwargs: True)
    monkeypatch.setattr("src.core.editor._transcribe_audio_to_segments", lambda *args, **kwargs: [
        {"start": 0.0, "end": 10.0, "text": "POG moment"}
    ])
    monkeypatch.setattr("src.core.editor._generate_clips_with_llm", lambda *args, **kwargs: [
        {"start_time": 0.0, "end_time": 10.0, "virality_score": 9, "reasoning": "Epic"}
    ])
    monkeypatch.setattr("src.core.editor.extract_clips", lambda *args, **kwargs: [
        str(tmp_path / "gameplay_clip_1.mp4")
    ])

    logs = []
    res = editor.process_video(str(video), prompt_profile="Default", logger=logs.append)
    assert res is True
    assert any("Successfully exported" in line for line in logs)


def test_process_video_cancellation(tmp_path, monkeypatch):
    video = tmp_path / "gameplay.mp4"
    video.write_text("dummy")

    fake_config = {
        "active_ai_provider": "openai",
        "openai_model": "gpt-4o",
        "settings": {"clips_dir": str(tmp_path / "clips")},
        "prompts": {"profiles": {"Default": "Prompt"}}
    }
    monkeypatch.setattr("src.core.editor.config_manager.load_config", lambda *args, **kwargs: fake_config)
    monkeypatch.setattr("src.core.editor._validate_api_keys", lambda *args, **kwargs: True)
    monkeypatch.setattr("src.core.editor._transcribe_audio_to_segments", lambda *args, **kwargs: None)

    logs = []
    res = editor.process_video(str(video), prompt_profile="Default", logger=logs.append)
    assert res is False

    logs = []
    editor.process_video(
        str(video),
        logger=logs.append,
        is_cancelled=lambda: True
    )
def test_transcribe_audio_cached_hit(tmp_path, monkeypatch):
    cache_dir = tmp_path / "appdata" / "transcripts"
    cache_dir.mkdir(parents=True)
    monkeypatch.setattr("src.core.editor.get_app_data_path", lambda: str(tmp_path / "appdata"))

    video = tmp_path / "video.mp4"
    video.write_text("dummy")

    cache_file = editor._get_transcription_cache_path(str(video), "base", "en", True, True)
    cached_data = [{"start": 0.0, "end": 5.0, "text": "Cached segment"}]
    editor._save_cached_segments(cache_file, cached_data)

    logs = []
    config = {"openai": {"whisper_model": "base", "whisper_language": "English"}, "settings": {}}
    result = editor._transcribe_audio_to_segments(str(video), config, logger=logs.append, is_cancelled=None)
    assert result == cached_data
    assert any("cached audio segments" in line.lower() for line in logs)


def test_transcribe_audio_model_load_error(tmp_path, monkeypatch):
    monkeypatch.setattr("src.core.editor.get_app_data_path", lambda: str(tmp_path / "appdata"))
    monkeypatch.setattr("src.core.editor.whisper.load_model", MagicMock(side_effect=RuntimeError("Whisper VRAM full")))

    video = tmp_path / "video.mp4"
    video.write_text("dummy")

    logs = []
    config = {"openai": {"whisper_model": "base", "whisper_language": "English"}, "settings": {}}
    result = editor._transcribe_audio_to_segments(str(video), config, logger=logs.append, is_cancelled=None)
    assert result is None
    assert any("Failed to load Whisper" in line for line in logs)


def test_transcribe_audio_extraction_error(tmp_path, monkeypatch):
    monkeypatch.setattr("src.core.editor.get_app_data_path", lambda: str(tmp_path / "appdata"))
    mock_model = MagicMock()
    monkeypatch.setattr("src.core.editor.whisper.load_model", lambda *args, **kwargs: mock_model)
    monkeypatch.setattr("src.core.editor.extract_audio_hidden", MagicMock(side_effect=RuntimeError("Audio corrupted")))

    video = tmp_path / "video.mp4"
    video.write_text("dummy")

    logs = []
    config = {"openai": {"whisper_model": "base", "whisper_language": "English"}, "settings": {}}
    result = editor._transcribe_audio_to_segments(str(video), config, logger=logs.append, is_cancelled=None)
    assert result is None
    assert any("Audio extraction error" in line for line in logs)


def test_transcribe_audio_full_success(tmp_path, monkeypatch):
    monkeypatch.setattr("src.core.editor.get_app_data_path", lambda: str(tmp_path / "appdata"))
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {
        "segments": [{"start": 0.0, "end": 5.0, "text": "GG WP"}]
    }
    monkeypatch.setattr("src.core.editor.whisper.load_model", lambda *args, **kwargs: mock_model)
    monkeypatch.setattr("src.core.editor.extract_audio_hidden", lambda f: np.zeros(16000, dtype=np.float32))

    video = tmp_path / "video.mp4"
    video.write_text("dummy")

    logs = []
    config = {"openai": {"whisper_model": "base", "whisper_language": "Auto-Detect"}, "settings": {"audio_peak_detection": True, "combat_detection": True}}
    result = editor._transcribe_audio_to_segments(str(video), config, logger=logs.append, is_cancelled=None)
    assert result is not None
    assert len(result) == 1


# ==============================================================================
# WINDOWED LLM PIPELINE, RE-RANKING, SELECTION
# ==============================================================================
def _openai_reply(payload: str):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = payload
    return resp


def _long_segments(minutes: int = 40):
    return [{"start": float(i * 10), "end": float(i * 10 + 10), "text": f"line {i}"} for i in range(minutes * 6)]


def _window_index(user_prompt: str) -> int:
    return int(user_prompt.split("part ")[1].split(" ")[0])


def test_parse_json_clips_keeps_peak_time():
    raw = '{"clips": [{"start_time": 10, "end_time": 30, "peak_time": 22.5, "virality_score": 8, "reasoning": "x"}]}'
    clips = editor._parse_json_clips(raw)
    assert clips[0]["peak_time"] == 22.5
    assert "peak_time" not in editor._parse_json_clips('{"clips": [{"start_time": 1, "end_time": 9, "virality_score": 7}]}')[0]


def test_parse_json_scores():
    assert editor._parse_json_scores('{"scores": [{"id": 0, "score": 12}, {"id": 1, "score": 4}, {"bad": 1}]}') == {0: 10, 1: 4}
    assert editor._parse_json_scores('[{"id": 2, "score": 7}]') == {2: 7}
    assert editor._parse_json_scores("nope") is None
    assert editor._parse_json_scores('{"scores": []}') is None


def test_long_video_is_split_into_windows_and_reranked(monkeypatch):
    prompts = []

    def fake_create(**kwargs):
        user = kwargs["messages"][1]["content"]
        prompts.append(user)
        if user.startswith("Below are"):
            n = user.count("\nid=") + 1
            return _openai_reply(json.dumps({"scores": [{"id": i, "score": 9} for i in range(n)]}))
        start = _window_index(user) * 600.0
        return _openai_reply(json.dumps({"clips": [
            {"start_time": start, "end_time": start + 30, "peak_time": start + 20, "virality_score": 7, "reasoning": "x"}]}))

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = fake_create
    monkeypatch.setattr("src.core.editor.OpenAI", lambda **kwargs: mock_client)

    config = {"openai": {"api_key": "k", "base_url": ""}, "settings": {"clips_per_hour": 12}}
    logs = []
    clips = editor._generate_clips_with_llm(_long_segments(40), config, "gpt-5.5", "SYSTEM", logs.append)

    window_prompts = [p for p in prompts if not p.startswith("Below are")]
    assert len(window_prompts) >= 4
    assert "overlap" in window_prompts[0]
    assert any(p.startswith("Below are") for p in prompts)
    assert len(clips) >= 4 and all(c["virality_score"] == 9 for c in clips)
    assert any("Re-ranking" in line for line in logs)


def test_rerank_failure_keeps_original_scores(monkeypatch):
    def fake_create(**kwargs):
        user = kwargs["messages"][1]["content"]
        if user.startswith("Below are"):
            return _openai_reply("garbage")
        start = _window_index(user) * 600.0
        return _openai_reply(json.dumps({"clips": [{"start_time": start, "end_time": start + 30, "virality_score": 8, "reasoning": "x"}]}))

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = fake_create
    monkeypatch.setattr("src.core.editor.OpenAI", lambda **kwargs: mock_client)
    monkeypatch.setattr("time.sleep", lambda s: None)

    logs = []
    clips = editor._generate_clips_with_llm(_long_segments(40), {"openai": {"api_key": "k"}}, "gpt-5.5", "SYS", logs.append)
    assert len(clips) >= 4 and all(c["virality_score"] == 8 for c in clips)
    assert any("Re-ranking failed" in line for line in logs)


def test_window_failure_does_not_abort_other_windows(monkeypatch):
    def fake_create(**kwargs):
        user = kwargs["messages"][1]["content"]
        if user.startswith("Below are"):
            return _openai_reply("garbage")
        if _window_index(user) == 1:
            raise RuntimeError("boom")
        start = _window_index(user) * 600.0
        return _openai_reply(json.dumps({"clips": [{"start_time": start, "end_time": start + 30, "virality_score": 8, "reasoning": "ok"}]}))

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = fake_create
    monkeypatch.setattr("src.core.editor.OpenAI", lambda **kwargs: mock_client)
    monkeypatch.setattr("time.sleep", lambda s: None)

    logs = []
    clips = editor._generate_clips_with_llm(_long_segments(40), {"openai": {"api_key": "k"}}, "gpt-5.5", "SYS", logs.append)
    assert len(clips) >= 3
    assert any("API Error (window 1/" in line for line in logs)


def test_cancel_stops_window_loop(monkeypatch):
    mock_client = MagicMock()
    monkeypatch.setattr("src.core.editor.OpenAI", lambda **kwargs: mock_client)
    clips = editor._generate_clips_with_llm(_long_segments(40), {"openai": {"api_key": "k"}}, "gpt-5.5", "SYS", None, lambda: True)
    assert clips == []
    mock_client.chat.completions.create.assert_not_called()


def test_anthropic_output_budget_stays_non_streaming(monkeypatch):
    mock_client = MagicMock()
    block = MagicMock()
    block.text = '{"clips": [{"start_time": 2.0, "end_time": 20.0, "virality_score": 8, "reasoning": "f"}]}'
    mock_client.messages.create.return_value = MagicMock(content=[block])
    monkeypatch.setattr("anthropic.Anthropic", lambda **kwargs: mock_client)

    editor._generate_clips_with_llm([{"start": 0.0, "end": 30.0, "text": "speech"}], {"anthropic": {"api_key": "k"}}, "claude-sonnet-5", "P", None)
    kwargs = mock_client.messages.create.call_args[1]
    assert kwargs["max_tokens"] == editor.ANTHROPIC_MAX_OUTPUT_TOKENS
    # the SDK raises for non-streaming requests whose max_tokens implies > 10 minutes (3600 * max_tokens / 128000)
    assert 3600 * kwargs["max_tokens"] / 128_000 <= 600


def test_deepseek_extra_body_dropped_after_rejection(monkeypatch):
    seen = []

    def fake_create(**kwargs):
        seen.append("extra_body" in kwargs)
        if len(seen) == 1:
            raise RuntimeError("unknown field: thinking")
        return _openai_reply('{"clips": [{"start_time": 1.0, "end_time": 15.0, "virality_score": 7, "reasoning": "x"}]}')

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = fake_create
    monkeypatch.setattr("src.core.editor.OpenAI", lambda **kwargs: mock_client)
    monkeypatch.setattr("time.sleep", lambda s: None)

    config = {"openai": {"api_key": "k", "base_url": "https://proxy.example/deepseek"}}
    clips = editor._generate_clips_with_llm([{"start": 0.0, "end": 30.0, "text": "speech"}], config, "deepseek-v4-flash", "P", None)
    assert seen == [True, False] and len(clips) == 1


def test_route_detection():
    r = editor._LLMRoute({"openai": {"api_key": "a"}, "deepseek": {"api_key": "d"}, "active_ai_provider": "deepseek"}, "deepseek-v4-flash")
    assert r.is_deepseek and r.openai_key == "d" and r.openai_base_url == "https://api.deepseek.com"
    assert editor._LLMRoute({}, "gemini-3.5-flash").engine_name == "Gemini"
    assert editor._LLMRoute({}, "claude-sonnet-5").engine_name == "Claude"
    assert editor._LLMRoute({"openai": {"base_url": "https://openrouter.ai/api/v1"}}, "x/y").engine_name == "Custom Base URL"
    grok = editor._LLMRoute({"xai": {"api_key": "x"}}, "grok-4.3")
    assert grok.is_grok and grok.openai_base_url == "https://api.x.ai/v1" and grok.openai_key == "x"


def test_process_video_selects_by_density_and_refines(tmp_path, monkeypatch):
    video = tmp_path / "gameplay.mp4"
    video.write_text("dummy")
    fake_config = {
        "active_ai_provider": "openai", "openai_model": "gpt-4o",
        "settings": {"clips_dir": str(tmp_path / "clips"), "clips_per_hour": 2, "min_clip_score": 6},
        "prompts": {"profiles": {"Default": "Prompt"}},
    }
    monkeypatch.setattr("src.core.editor.config_manager.load_config", lambda *a, **k: fake_config)
    monkeypatch.setattr("src.core.editor._validate_api_keys", lambda *a, **k: True)
    segments = [{"start": float(i * 100), "end": float(i * 100 + 90), "text": "talk"} for i in range(36)]  # 1 hour
    monkeypatch.setattr("src.core.editor._transcribe_audio_to_segments", lambda *a, **k: segments)
    monkeypatch.setattr("src.core.editor._generate_clips_with_llm", lambda *a, **k: [
        {"start_time": 100.0 + i * 500, "end_time": 130.0 + i * 500, "virality_score": 6 + i % 4, "reasoning": "r"} for i in range(7)
    ])
    captured = {}

    def fake_extract(input_file, clips_data, *args, **kwargs):
        captured["clips"] = clips_data["clips"]
        return ["a.mp4"] * len(clips_data["clips"])

    monkeypatch.setattr("src.core.editor.extract_clips", fake_extract)
    logs = []
    assert editor.process_video(str(video), logger=logs.append) is True
    assert len(captured["clips"]) == 2
    assert sorted(c["virality_score"] for c in captured["clips"]) == [8, 9]
    assert any("Selected 2 of 7" in line for line in logs)


def test_process_video_cancel_after_llm(tmp_path, monkeypatch):
    video = tmp_path / "gameplay.mp4"
    video.write_text("dummy")
    fake_config = {"active_ai_provider": "openai", "openai_model": "gpt-4o", "settings": {"clips_dir": str(tmp_path)}, "prompts": {"profiles": {"Default": "P"}}}
    monkeypatch.setattr("src.core.editor.config_manager.load_config", lambda *a, **k: fake_config)
    monkeypatch.setattr("src.core.editor._validate_api_keys", lambda *a, **k: True)
    monkeypatch.setattr("src.core.editor._transcribe_audio_to_segments", lambda *a, **k: [{"start": 0.0, "end": 10.0, "text": "x"}])
    monkeypatch.setattr("src.core.editor._generate_clips_with_llm", lambda *a, **k: [])
    state = {"n": 0}

    def cancelled():
        state["n"] += 1
        return state["n"] >= 2

    assert editor.process_video(str(video), is_cancelled=cancelled) is False


# ==============================================================================
# TRANSCRIPTION: EVENTS, WORD TIMESTAMPS, CACHE VERSION
# ==============================================================================
def test_transcribe_adds_events_words_and_loudness(tmp_path, monkeypatch):
    monkeypatch.setattr("src.core.editor.get_app_data_path", lambda: str(tmp_path / "appdata"))
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {"segments": [{
        "id": 0, "seek": 0, "tokens": [1, 2, 3], "start": 0.0, "end": 5.0, "text": " GG WP",
        "words": [{"word": " GG", "start": 0.1, "end": 0.6, "probability": 0.9}],
    }]}
    monkeypatch.setattr("src.core.editor.whisper.load_model", lambda *a, **k: mock_model)
    monkeypatch.setattr("src.core.editor.extract_audio_hidden", lambda f: np.zeros(16000 * 10, dtype=np.float32))
    monkeypatch.setattr("src.core.editor.audio_events.detect_audio_events", lambda audio: [
        {"type": "LAUGHTER", "start": 1.0, "end": 4.0, "strength": 80}])

    video = tmp_path / "video.mp4"
    video.write_text("dummy")
    config = {"openai": {"whisper_model": "base", "whisper_language": "Auto-Detect"}, "settings": {"audio_peak_detection": True, "combat_detection": False}}
    result = editor._transcribe_audio_to_segments(str(video), config, logger=None, is_cancelled=None)

    assert mock_model.transcribe.call_args[1]["word_timestamps"] is True
    assert sorted(bool(e.get("event")) for e in result) == [False, True]
    speech = next(e for e in result if not e.get("event"))
    assert "tokens" not in speech and speech["words"] == [{"word": " GG", "start": 0.1, "end": 0.6}]
    assert "loudness" in speech


def test_event_detection_failure_is_not_fatal(monkeypatch):
    def boom(audio):
        raise ValueError("bad audio")

    monkeypatch.setattr("src.core.editor.audio_events.detect_audio_events", boom)
    logs = []
    assert editor._detect_events_safe(np.zeros(10, dtype=np.float32), logs.append) == []
    assert any("skipped" in line for line in logs)


def test_transcript_cache_key_is_versioned(tmp_path, monkeypatch):
    monkeypatch.setattr("src.core.editor.get_app_data_path", lambda: str(tmp_path))
    video = tmp_path / "v.mp4"
    video.write_text("x")
    before = editor._get_transcription_cache_path(str(video), "base", "en", True, True)
    monkeypatch.setattr("src.core.editor.TRANSCRIPT_CACHE_VERSION", editor.TRANSCRIPT_CACHE_VERSION + 1)
    assert editor._get_transcription_cache_path(str(video), "base", "en", True, True) != before


def test_rerank_uses_dedicated_system_prompt_not_the_clip_extraction_prompt(monkeypatch):
    systems = {}

    def fake_create(**kwargs):
        system, user = kwargs["messages"][0]["content"], kwargs["messages"][1]["content"]
        if user.startswith("Below are"):
            systems["rerank"] = system
            n = user.count("\nid=") + 1
            return _openai_reply(json.dumps({"scores": [{"id": i, "score": 8} for i in range(n)]}))
        systems["window"] = system
        start = _window_index(user) * 600.0
        return _openai_reply(json.dumps({"clips": [{"start_time": start, "end_time": start + 30, "virality_score": 7, "reasoning": "x"}]}))

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = fake_create
    monkeypatch.setattr("src.core.editor.OpenAI", lambda **kwargs: mock_client)
    editor._generate_clips_with_llm(_long_segments(40), {"openai": {"api_key": "k"}}, "gpt-5.5", "CLIP-EXTRACTION-PROMPT", None)
    assert systems["window"] == "CLIP-EXTRACTION-PROMPT"
    assert systems["rerank"] == editor.RERANK_SYSTEM


def test_generation_stops_after_consecutive_window_failures(monkeypatch):
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RuntimeError("401 invalid api key")
    monkeypatch.setattr("src.core.editor.OpenAI", lambda **kwargs: mock_client)
    monkeypatch.setattr("time.sleep", lambda s: None)

    logs = []
    clips = editor._generate_clips_with_llm(_long_segments(40), {"openai": {"api_key": "bad"}}, "gpt-5.5", "P", logs.append)
    assert clips == []
    assert mock_client.chat.completions.create.call_count == editor.MAX_CONSECUTIVE_WINDOW_FAILURES * editor.LLM_ATTEMPTS
    assert any("windows failed in a row" in line for line in logs)


# ==============================================================================
# LEGACY (pre-v2) TRANSCRIPT CACHE MIGRATION
# ==============================================================================
def test_migrate_legacy_segments_strips_baked_in_tags():
    legacy = [
        {"id": 0, "start": 0.0, "end": 2.0, "text": "[LOUDNESS: 42%] [ACTION: COMBAT]  Привет", "tokens": [1, 2], "rms": 0.1},
        {"start": 2.0, "end": 4.0, "text": " no tags here"},
        {"start": 4.0, "text": "broken"},
        "junk",
    ]
    migrated = editor._migrate_legacy_segments(legacy)
    assert [s["text"].strip() for s in migrated] == ["Привет", "no tags here"]
    assert all(set(s) == {"start", "end", "text"} for s in migrated)


def test_legacy_cache_is_upgraded_without_running_whisper(tmp_path, monkeypatch):
    monkeypatch.setattr("src.core.editor.get_app_data_path", lambda: str(tmp_path / "appdata"))
    video = tmp_path / "video.mp4"
    video.write_text("dummy")

    legacy_file = editor._get_transcription_cache_path(str(video), "base", "ru", True, True, legacy=True)
    new_file = editor._get_transcription_cache_path(str(video), "base", "ru", True, True)
    assert legacy_file != new_file
    editor._save_cached_segments(legacy_file, [{"start": 0.0, "end": 5.0, "text": "[LOUDNESS: 80%] GG WP"}])

    load_model = MagicMock()
    monkeypatch.setattr("src.core.editor.whisper.load_model", load_model)
    monkeypatch.setattr("src.core.editor.extract_audio_hidden", lambda f: np.full(16000 * 10, 0.05, dtype=np.float32))
    monkeypatch.setattr("src.core.editor.audio_events.detect_audio_events", lambda audio: [
        {"type": "LAUGHTER", "start": 5.0, "end": 8.0, "strength": 60}])

    logs = []
    config = {"openai": {"whisper_model": "base", "whisper_language": "Russian"}, "settings": {"audio_peak_detection": True, "combat_detection": True}}
    result = editor._transcribe_audio_to_segments(str(video), config, logs.append, None)

    load_model.assert_not_called()
    assert [bool(e.get("event")) for e in result] == [False, True]
    assert result[0]["text"].strip() == "GG WP" and "loudness" in result[0]
    assert os.path.exists(new_file)
    assert any("older cached transcript" in line for line in logs)

    again = editor._transcribe_audio_to_segments(str(video), config, logs.append, None)  # now served from the v2 cache
    assert again == result


def test_legacy_migration_falls_back_to_whisper_when_audio_fails(tmp_path, monkeypatch):
    monkeypatch.setattr("src.core.editor.get_app_data_path", lambda: str(tmp_path / "appdata"))
    video = tmp_path / "video.mp4"
    video.write_text("dummy")
    editor._save_cached_segments(editor._get_transcription_cache_path(str(video), "base", "ru", True, True, legacy=True),
                                 [{"start": 0.0, "end": 5.0, "text": "x"}])
    monkeypatch.setattr("src.core.editor.extract_audio_hidden", MagicMock(side_effect=RuntimeError("ffmpeg died")))
    logs = []
    assert editor._migrate_legacy_cache(str(video), "unused.json", "base", "ru", True, True, logs.append, None) is None
    assert any("transcribing from scratch" in line for line in logs)


def test_legacy_migration_noop_without_legacy_file_and_cancel(tmp_path, monkeypatch):
    monkeypatch.setattr("src.core.editor.get_app_data_path", lambda: str(tmp_path / "appdata"))
    video = tmp_path / "video.mp4"
    video.write_text("dummy")
    assert editor._migrate_legacy_cache(str(video), "unused.json", "base", "ru", True, True, None, None) is None
    editor._save_cached_segments(editor._get_transcription_cache_path(str(video), "base", "ru", True, True, legacy=True),
                                 [{"start": 0.0, "end": 5.0, "text": "x"}])
    assert editor._migrate_legacy_cache(str(video), "unused.json", "base", "ru", True, True, None, lambda: True) == []
