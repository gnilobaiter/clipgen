import json
import os
import subprocess
import pytest
from unittest.mock import MagicMock
import numpy as np

from src.core import editor

REAL_REVIEW = editor._review_clip_boundaries  # kept so the review tests can call the real function


@pytest.fixture(autouse=True)
def _no_llm_review_in_pipeline_tests(monkeypatch):
    """process_video tests must never reach an LLM; the review pass has its own tests below."""
    monkeypatch.setattr(editor, "_review_clip_boundaries", lambda route, clips, *args, **kwargs: clips)


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
        {"start": 56.0, "end": 58.0, "text": "screaming explosion"}
    ]
    # 60 seconds of 16kHz audio: quiet background, then a short loud burst with a sharp spike
    audio = np.full(960000, 0.02, dtype=np.float32)
    audio[896000:928000] = 0.8
    audio[902000:902200] = 1.0

    enhanced = editor.analyze_audio_peaks(
        audio, segments, sample_rate=16000, peak_detection=True, combat_detection=True
    )
    assert enhanced[0]["text"] == "calm dialogue"  # text is never polluted with tags
    assert enhanced[0]["loudness"] < 10
    assert enhanced[1]["loudness"] == 100


def test_combat_flag_needs_a_sharp_transient_not_plain_speech_dynamics():
    rng = np.random.default_rng(2)
    audio = (rng.standard_normal(16000 * 4) * 0.05).astype(np.float32)  # steady, speech-like level
    segments = [{"start": 0.0, "end": 2.0, "text": "steady"}, {"start": 2.0, "end": 4.0, "text": "bang"}]
    audio[16000 * 3:16000 * 3 + 300] = 1.0  # gunshot-like spike
    enhanced = editor.analyze_audio_peaks(audio, segments, peak_detection=False, combat_detection=True)
    assert [s["is_combat"] for s in enhanced] == [False, True]


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
    assert call_kwargs.get("extra_body") == {"thinking": {"type": "enabled"}}  # reasoning is on by default


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
    monkeypatch.setattr("src.core.editor.audio_events.detect_audio_events", lambda audio, **kw: [
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
    def boom(audio, **kw):
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
    monkeypatch.setattr("src.core.editor.audio_events.detect_audio_events", lambda audio, **kw: [
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


def test_get_whisper_models_lists_everything_the_library_can_load(monkeypatch):
    models = editor.get_whisper_models()
    assert {"tiny", "base", "small", "medium", "large-v3", "turbo", "large-v3-turbo", "large-v2", "tiny.en"} <= set(models)
    assert len(models) == len(set(models))
    assert editor.get_whisper_models("my-local.pt")[-1] == "my-local.pt"
    assert editor.get_whisper_models("large-v3").count("large-v3") == 1

    monkeypatch.setattr("src.core.editor.whisper.available_models", MagicMock(side_effect=RuntimeError("broken")))
    assert editor.get_whisper_models("base") == editor.FALLBACK_WHISPER_MODELS


# ==============================================================================
# CUT POINTS: RANGED AUDIO EXTRACTION, EXPORTER PADDING, SNAPPING IN THE PIPELINE
# ==============================================================================
def test_extract_audio_hidden_can_read_only_a_range(monkeypatch):
    recorded = {}
    process = MagicMock(returncode=0)
    process.communicate.return_value = (np.zeros(4, dtype=np.int16).tobytes(), b"")

    def popen(cmd, **kwargs):
        recorded["cmd"] = cmd
        return process

    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(editor, "_count_audio_streams", MagicMock(side_effect=AssertionError("must not probe again")))
    result = editor.extract_audio_hidden("v.mp4", start=12.5, duration=3.0, num_audio=1)
    cmd = recorded["cmd"]
    assert len(result) == 4
    assert cmd[cmd.index("-ss") + 1] == "12.500" and cmd[cmd.index("-t") + 1] == "3.000"
    assert cmd.index("-ss") < cmd.index("-i")  # input seeking: fast even deep into a long recording


def _capture_ffmpeg(monkeypatch):
    commands = []

    def run(cmd, **kwargs):
        commands.append(cmd)
        return MagicMock(returncode=0, stderr=b"")

    monkeypatch.setattr(subprocess, "run", run)
    return commands


def test_exporter_uses_per_clip_padding_and_keeps_legacy_defaults(tmp_path, monkeypatch):
    commands = _capture_ffmpeg(monkeypatch)
    monkeypatch.setattr(os.path, "isfile", lambda p: True)
    clips = {"clips": [
        {"start_time": 100.0, "end_time": 120.0, "virality_score": 8, "reasoning": "snapped", "pre_roll": 0.0, "post_roll": 0.0},
        {"start_time": 200.0, "end_time": 220.0, "virality_score": 7, "reasoning": "legacy"},
    ]}
    editor.extract_clips("source.mp4", clips, str(tmp_path), logger=lambda m: None)
    cut_commands = [c for c in commands if "-ss" in c and "-t" in c and "source.mp4" in c]
    first, second = cut_commands
    assert float(first[first.index("-ss") + 1]) == 100.0 and float(first[first.index("-t") + 1]) == 20.0
    assert float(second[second.index("-ss") + 1]) == 199.25 and float(second[second.index("-t") + 1]) == 22.0


def test_snap_helper_places_boundaries_and_reports(monkeypatch):
    monkeypatch.setattr(editor, "_count_audio_streams", lambda f: 1)
    sr = 16000
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(sr * 60) * 0.001).astype(np.float32)
    t = np.arange(len(audio)) / sr
    for a, b in [(9.0, 9.4), (9.6, 10.0), (11.0, 11.4), (11.6, 12.0), (12.2, 12.6)]:
        m = (t >= a) & (t < b)
        audio[m] += (0.3 * np.sin(2 * np.pi * 220 * t[m])).astype(np.float32)

    monkeypatch.setattr(editor, "extract_audio_hidden", lambda f, s, length, n: audio[int(s * sr):int((s + length) * sr)])
    logs = []
    snapped = editor._snap_clips_to_pauses("v.mp4", [{"start_time": 11.1, "end_time": 12.3, "virality_score": 8}], 60.0, logs.append)
    assert 10.0 < snapped[0]["start_time"] < 11.0 and snapped[0]["pre_roll"] == 0.0
    assert any("boundaries inside speech pauses" in line for line in logs)

    def broken(f, s, length, n):
        raise RuntimeError("no audio")

    monkeypatch.setattr(editor, "extract_audio_hidden", broken)
    clip = {"start_time": 11.1, "end_time": 12.3}
    result = editor._snap_clips_to_pauses("v.mp4", [clip], 60.0, logs.append)
    assert result[0]["start_time"] == 11.1 and result[0]["pre_roll"] == 0.2  # unsnapped side keeps a tiny margin only

    monkeypatch.setattr(editor, "_count_audio_streams", MagicMock(side_effect=RuntimeError("ffprobe missing")))
    logs.clear()
    assert editor._snap_clips_to_pauses("v.mp4", [clip], 60.0, logs.append) == [clip]
    assert any("Could not snap" in line for line in logs)
    assert editor._snap_clips_to_pauses("v.mp4", [], 60.0, None) == []


def test_process_video_snaps_boundaries_and_logs_lengths(tmp_path, monkeypatch):
    video = tmp_path / "gameplay.mp4"
    video.write_text("dummy")
    fake_config = {"active_ai_provider": "openai", "openai_model": "gpt-4o", "settings": {"clips_dir": str(tmp_path / "clips")},
                   "prompts": {"profiles": {"Default": "P"}}}
    monkeypatch.setattr("src.core.editor.config_manager.load_config", lambda *a, **k: fake_config)
    monkeypatch.setattr("src.core.editor._validate_api_keys", lambda *a, **k: True)
    monkeypatch.setattr("src.core.editor._transcribe_audio_to_segments", lambda *a, **k: [{"start": 0.0, "end": 600.0, "text": "talk"}])
    monkeypatch.setattr("src.core.editor._generate_clips_with_llm", lambda *a, **k: [
        {"start_time": 100.0, "end_time": 130.0, "virality_score": 8, "reasoning": "r"}])
    seen = {}
    monkeypatch.setattr("src.core.editor._snap_clips_to_pauses",
                        lambda f, clips, _d, logger: [dict(c, start_time=99.5, end_time=131.0, pre_roll=0.0, post_roll=0.0) for c in clips])

    def fake_extract(f, data, *a, **k):
        seen["clips"] = data["clips"]
        return ["x.mp4"]

    monkeypatch.setattr("src.core.editor.extract_clips", fake_extract)
    logs = []
    assert editor.process_video(str(video), logger=logs.append) is True
    assert (seen["clips"][0]["start_time"], seen["clips"][0]["end_time"]) == (99.5, 131.0)
    assert any("Clip lengths" in line for line in logs)


def test_only_overlapping_nominations_merge_back_to_back_moments_stay_separate():
    overlapping = [{"start_time": 0, "end_time": 100, "virality_score": 7, "reasoning": "a"},
                   {"start_time": 90, "end_time": 140, "virality_score": 8, "reasoning": "b"}]
    merged = editor._clean_and_merge_clips(overlapping)
    assert len(merged) == 1 and (merged[0]["start_time"], merged[0]["end_time"], merged[0]["virality_score"]) == (0, 140, 8)
    # two different moments that merely follow each other (a topic change) used to be glued into one clip
    back_to_back = [{"start_time": 0, "end_time": 100, "virality_score": 7, "reasoning": "joke about the gestures"},
                    {"start_time": 101, "end_time": 140, "virality_score": 8, "reasoning": "explaining the new panel"}]
    assert len(editor._clean_and_merge_clips(back_to_back)) == 2
    too_long = [{"start_time": 0, "end_time": 100, "virality_score": 7, "reasoning": "a"},
                {"start_time": 90, "end_time": 260, "virality_score": 8, "reasoning": "b"}]
    assert len(editor._clean_and_merge_clips(too_long)) == 2


def test_whisper_triton_fallback_warning_is_silenced():
    import importlib
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        importlib.reload(editor)  # re-runs the module-level filter registration inside this warnings context
        warnings.warn("Failed to launch Triton kernels, likely due to missing CUDA toolkit; falling back to a slower DTW implementation...", UserWarning)
        warnings.warn("some other warning", UserWarning)
    assert [str(w.message) for w in caught] == ["some other warning"]


# ==============================================================================
# LOGGING: THE USER MUST BE ABLE TO SEE WHETHER THE SOUND MODEL IS USED
# ==============================================================================
def test_cache_hit_reports_stored_audio_events(tmp_path, monkeypatch):
    monkeypatch.setattr("src.core.editor.get_app_data_path", lambda: str(tmp_path / "appdata"))
    video = tmp_path / "video.mp4"
    video.write_text("dummy")
    cache_file = editor._get_transcription_cache_path(str(video), "base", "en", True, True)
    editor._save_cached_segments(cache_file, [
        {"start": 0.0, "end": 5.0, "text": "hi"},
        {"start": 1.0, "end": 3.0, "text": "", "event": "LAUGHTER", "strength": 60},
        {"start": 2.0, "end": 3.0, "text": "", "event": "LAUGHTER", "strength": 40},
        {"start": 4.0, "end": 4.5, "text": "", "event": "LOUD", "strength": 70},
    ])
    logs = []
    config = {"openai": {"whisper_model": "base", "whisper_language": "English"}, "settings": {}}
    editor._transcribe_audio_to_segments(str(video), config, logs.append, None)
    assert any("Audio events stored in the cache: LAUGHTER: 2, LOUD: 1" in line and "not re-run" in line for line in logs)

    editor._save_cached_segments(cache_file, [{"start": 0.0, "end": 5.0, "text": "hi"}])
    logs.clear()
    editor._transcribe_audio_to_segments(str(video), config, logs.append, None)
    assert any("Audio events stored in the cache: none" in line for line in logs)


def test_llm_stage_logs_the_audio_tags_it_sends(monkeypatch):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _openai_reply('{"clips": []}')
    monkeypatch.setattr("src.core.editor.OpenAI", lambda **kwargs: mock_client)
    segments = [{"start": 0.0, "end": 30.0, "text": "speech"},
                {"start": 10.0, "end": 12.0, "text": "", "event": "LAUGHTER", "strength": 55},
                {"start": 20.0, "end": 21.0, "text": "", "event": "SCREAM", "strength": 40}]
    logs = []
    editor._generate_clips_with_llm(segments, {"openai": {"api_key": "k"}}, "gpt-5.5", "P", logs.append)
    assert any("Audio event tags included in the AI prompts: LAUGHTER: 1, SCREAM: 1" in line for line in logs)
    prompt = mock_client.chat.completions.create.call_args[1]["messages"][1]["content"]
    assert "[LAUGHTER 55%]" in prompt and "[SCREAM 40%]" in prompt  # and they really are in the text the model reads


# ==============================================================================
# CANCELLATION
# ==============================================================================
def test_cancel_interrupts_a_running_whisper_transcription(tmp_path, monkeypatch):
    monkeypatch.setattr("src.core.editor.get_app_data_path", lambda: str(tmp_path / "appdata"))
    monkeypatch.setattr("src.core.editor.extract_audio_hidden", lambda f: np.zeros(16000 * 10, dtype=np.float32))
    printed = {"segments": 0}
    state = {"cancel": False}

    def fake_transcribe(audio, **kwargs):
        for i in range(1000):  # what whisper does with verbose=True: print every segment as it is decoded
            print(f"[00:{i % 60:02d}.000 --> 00:{i % 60 + 1:02d}.000] segment {i}")
            printed["segments"] += 1
            if printed["segments"] == 3:
                state["cancel"] = True  # the user presses Cancel
        return {"segments": []}

    mock_model = MagicMock()
    mock_model.transcribe.side_effect = fake_transcribe
    monkeypatch.setattr("src.core.editor.whisper.load_model", lambda *a, **k: mock_model)
    video = tmp_path / "video.mp4"
    video.write_text("dummy")
    logs = []
    config = {"openai": {"whisper_model": "base", "whisper_language": "English"}, "settings": {}}
    result = editor._transcribe_audio_to_segments(str(video), config, logs.append, lambda: state["cancel"])
    assert result is None
    assert printed["segments"] == 3  # stopped right away instead of decoding all 1000
    assert any("Transcription cancelled" in line for line in logs)
    assert not any("Transcription error" in line for line in logs)
    assert not any(cache_file.endswith(".json") for cache_file in os.listdir(tmp_path / "appdata" / "transcripts")) if (tmp_path / "appdata" / "transcripts").exists() else True


def test_process_video_does_not_export_after_cancel_during_boundary_snapping(tmp_path, monkeypatch):
    video = tmp_path / "gameplay.mp4"
    video.write_text("dummy")
    fake_config = {"active_ai_provider": "openai", "openai_model": "gpt-4o", "settings": {"clips_dir": str(tmp_path / "clips")},
                   "prompts": {"profiles": {"Default": "P"}}}
    monkeypatch.setattr("src.core.editor.config_manager.load_config", lambda *a, **k: fake_config)
    monkeypatch.setattr("src.core.editor._validate_api_keys", lambda *a, **k: True)
    monkeypatch.setattr("src.core.editor._transcribe_audio_to_segments", lambda *a, **k: [{"start": 0.0, "end": 600.0, "text": "talk"}])
    monkeypatch.setattr("src.core.editor._generate_clips_with_llm", lambda *a, **k: [
        {"start_time": 100.0, "end_time": 130.0, "virality_score": 8, "reasoning": "r"}])
    state = {"cancel": False}

    def snap(f, clips, _d, logger):
        state["cancel"] = True
        return clips

    monkeypatch.setattr("src.core.editor._snap_clips_to_pauses", snap)
    export = MagicMock()
    monkeypatch.setattr("src.core.editor.extract_clips", export)
    assert editor.process_video(str(video), is_cancelled=lambda: state["cancel"]) is False
    export.assert_not_called()


# ==============================================================================
# BOUNDARY REVIEW (PASS 3)
# ==============================================================================
def _entries():
    return [{"start": float(i * 5), "end": float(i * 5 + 4.5), "text": f"line {i}"} for i in range(60)]


def _review_route(monkeypatch, replies, prompts=None):
    calls = iter(replies)

    def fake_create(**kwargs):
        if prompts is not None:
            prompts.append(kwargs["messages"][1]["content"])
        reply = next(calls)
        if isinstance(reply, Exception):
            raise reply
        return _openai_reply(reply)

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = fake_create
    monkeypatch.setattr("src.core.editor.OpenAI", lambda **kwargs: mock_client)
    monkeypatch.setattr("time.sleep", lambda s: None)
    return editor._LLMRoute({"openai": {"api_key": "k"}}, "gpt-5.5")


def _scenes_reply(scenes, first, last, reason="because"):
    return json.dumps({"scenes": [{"start_time": a, "end_time": b, "topic": t, "humor": 5} for a, b, t in scenes],
                       "clip_first": first, "clip_last": last, "reason": reason})


def test_review_extends_a_clip_that_lacked_its_setup_and_shows_marked_context(monkeypatch):
    prompts = []
    reply = _scenes_reply([(100.0, 115.0, "the question"), (115.0, 160.0, "the joke")], 0, 1, "needs the question before the joke")
    route = _review_route(monkeypatch, [reply], prompts)
    clip = {"start_time": 115.0, "end_time": 160.0, "virality_score": 8, "reasoning": "r", "peak_time": 140.0}
    logs = []
    result = REAL_REVIEW(route, [clip], _entries(), 300.0, logs.append, None, [True])
    assert (result[0]["start_time"], result[0]["end_time"]) == (100.0, 160.0)
    assert clip["start_time"] == 115.0  # input untouched
    assert any("Clip 1: 115-160s -> 100-160s" in line and "needs the question" in line for line in logs)
    assert any("1 of 1 clip(s) adjusted" in line for line in logs)
    prompt = prompts[0]
    assert "from 115.0s to 160.0s" in prompt and "STEP 1" in prompt and "STEP 2" in prompt
    marked = [line for line in prompt.splitlines() if line.startswith("> ")]
    assert marked and all(115 - 5 < float(line.split("[")[1].split("s")[0]) < 160 for line in marked)
    assert any(line.startswith("  [") for line in prompt.splitlines())  # context lines are unmarked
    assert "[65.0s" not in prompt and "[210.0s" not in prompt  # only +-45 s of context around 115-160


def test_review_drops_a_scene_about_another_topic(monkeypatch):
    """The real case: a joke followed directly by an explanation of a different puzzle was one clip."""
    scenes = [(90.0, 100.0, "counting"), (100.0, 135.0, "the prayer-gesture joke"), (135.0, 175.0, "describing a new panel")]
    route = _review_route(monkeypatch, [_scenes_reply(scenes, 1, 1, "the panel is a different topic")])
    clip = {"start_time": 100.0, "end_time": 175.0, "virality_score": 8, "reasoning": "r", "peak_time": 120.0}
    result = REAL_REVIEW(route, [clip], _entries(), 300.0, None, None, [True])
    assert (result[0]["start_time"], result[0]["end_time"]) == (100.0, 135.0)


def test_review_rejects_unsafe_answers_and_keeps_the_original(monkeypatch):
    clip = {"start_time": 100.0, "end_time": 140.0, "virality_score": 8, "reasoning": "r", "peak_time": 130.0}
    ok_scene = (100.0, 140.0, "x")
    bad_answers = [
        _scenes_reply([(100.0, 102.0, "too short")], 0, 0),
        _scenes_reply([(200.0, 260.0, "a different moment entirely")], 0, 0),
        _scenes_reply([(150.0, 190.0, "barely overlaps")], 0, 0),
        _scenes_reply([ok_scene], 0, 3),  # index out of range
        _scenes_reply([ok_scene, ok_scene], 1, 0),  # first > last
        _scenes_reply([], 0, 0),
        json.dumps({"scenes": [{"start_time": "abc", "end_time": 1}], "clip_first": 0, "clip_last": 0}),
        json.dumps({"clip_first": 0, "clip_last": 0}),
        "not json at all",
    ]
    for answer in bad_answers:
        route = _review_route(monkeypatch, [answer, answer, answer])
        result = REAL_REVIEW(route, [clip], _entries(), 300.0, None, None, [True])
        assert (result[0]["start_time"], result[0]["end_time"]) == (100.0, 140.0), answer


def test_review_can_only_add_a_limited_amount_of_setup_and_tail(monkeypatch):
    """The unbounded review added 25+ s of "setup" to most clips (median AI length 24 s -> final 53 s)."""
    clip = {"start_time": 100.0, "end_time": 140.0, "virality_score": 8, "reasoning": "r", "peak_time": 130.0}
    route = _review_route(monkeypatch, [_scenes_reply([(40.0, 100.0, "long setup"), (100.0, 400.0, "joke and much more")], 0, 1)])
    result = REAL_REVIEW(route, [clip], _entries(), 800.0, None, None, [True])
    assert (result[0]["start_time"], result[0]["end_time"]) == (100.0 - editor.MAX_REVIEW_HEAD_GROWTH, 140.0 + editor.MAX_REVIEW_TAIL_GROWTH)
    changed = editor._limit_review_growth(clip, {"start_time": 120.0, "end_time": 130.0, "reason": ""})  # shrinking is not limited here
    assert (changed["start_time"], changed["end_time"]) == (120.0, 130.0)


def test_review_keeps_clips_that_are_already_right_and_survives_api_failures(monkeypatch):
    clips = [{"start_time": 50.0 + 60 * i, "end_time": 90.0 + 60 * i, "virality_score": 7, "reasoning": "r"} for i in range(4)]
    replies = [_scenes_reply([(50.0, 90.0, "fine")], 0, 0)] + [RuntimeError("401")] * 6
    prompts = []
    route = _review_route(monkeypatch, replies, prompts)
    logs = []
    result = REAL_REVIEW(route, clips, _entries(), 400.0, logs.append, None, [True])
    assert [(c["start_time"], c["end_time"]) for c in result] == [(c["start_time"], c["end_time"]) for c in clips]
    assert any("0 of 4 clip(s) adjusted" in line for line in logs)
    # after two failed clips in a row the remaining clips are not sent (no point hammering a broken API)
    assert len(prompts) == 1 + editor.MAX_REVIEW_FAILURES * editor.LLM_ATTEMPTS


def test_review_stops_when_cancelled(monkeypatch):
    route = _review_route(monkeypatch, [])
    clips = [{"start_time": 50.0, "end_time": 90.0, "virality_score": 7, "reasoning": "r"}]
    assert REAL_REVIEW(route, clips, _entries(), 400.0, None, lambda: True, [True]) == clips


def test_process_video_runs_the_review_between_selection_and_snapping(tmp_path, monkeypatch):
    video = tmp_path / "gameplay.mp4"
    video.write_text("dummy")
    fake_config = {"active_ai_provider": "openai", "openai_model": "gpt-4o", "openai": {"api_key": "k"},
                   "settings": {"clips_dir": str(tmp_path / "clips"), "review_boundaries": True}, "prompts": {"profiles": {"Default": "P"}}}
    monkeypatch.setattr("src.core.editor.config_manager.load_config", lambda *a, **k: fake_config)
    monkeypatch.setattr("src.core.editor._validate_api_keys", lambda *a, **k: True)
    monkeypatch.setattr("src.core.editor._transcribe_audio_to_segments", lambda *a, **k: [{"start": 0.0, "end": 600.0, "text": "talk"}])
    monkeypatch.setattr("src.core.editor._generate_clips_with_llm", lambda *a, **k: [
        {"start_time": 100.0, "end_time": 130.0, "virality_score": 8, "reasoning": "r"}])
    order = []

    def review(route, clips, entries, duration, logger, is_cancelled, extra):
        order.append("review")
        return [dict(c, start_time=90.0) for c in clips]

    def snap(f, clips, _d, logger):
        order.append(("snap", clips[0]["start_time"]))
        return clips

    monkeypatch.setattr(editor, "_review_clip_boundaries", review)
    monkeypatch.setattr(editor, "_snap_clips_to_pauses", snap)
    monkeypatch.setattr(editor, "extract_clips", lambda f, data, *a, **k: ["x.mp4"])
    assert editor.process_video(str(video)) is True
    assert order == ["review", ("snap", 90.0)]

    order.clear()
    fake_config["settings"]["review_boundaries"] = False
    editor.process_video(str(video))
    assert "review" not in order


def test_process_video_never_exports_a_clip_longer_than_the_absolute_maximum(tmp_path, monkeypatch):
    video = tmp_path / "gameplay.mp4"
    video.write_text("dummy")
    fake_config = {"active_ai_provider": "openai", "openai_model": "gpt-4o", "settings": {"clips_dir": str(tmp_path / "clips")},
                   "prompts": {"profiles": {"Default": "P"}}}
    monkeypatch.setattr("src.core.editor.config_manager.load_config", lambda *a, **k: fake_config)
    monkeypatch.setattr("src.core.editor._validate_api_keys", lambda *a, **k: True)
    monkeypatch.setattr("src.core.editor._transcribe_audio_to_segments", lambda *a, **k: [{"start": 0.0, "end": 900.0, "text": "talk"}])
    monkeypatch.setattr("src.core.editor._generate_clips_with_llm", lambda *a, **k: [
        {"start_time": 100.0, "end_time": 360.0, "peak_time": 200.0, "virality_score": 6, "reasoning": "a 260 s clip nobody asked for"}])
    monkeypatch.setattr("src.core.editor._snap_clips_to_pauses", lambda f, clips, _d, logger: clips)
    seen = {}

    def fake_extract(f, data, *a, **k):
        seen["clips"] = data["clips"]
        return ["x.mp4"]

    monkeypatch.setattr("src.core.editor.extract_clips", fake_extract)
    logs = []
    assert editor.process_video(str(video), logger=logs.append) is True
    clip = seen["clips"][0]
    assert clip["end_time"] - clip["start_time"] <= 240.0
    assert clip["start_time"] <= 200.0 <= clip["end_time"]  # the punchline survives
    assert any("Dropped:" in line or "Selected 1 of 1" in line for line in logs)


def test_selection_log_explains_the_numbers(tmp_path, monkeypatch):
    video = tmp_path / "gameplay.mp4"
    video.write_text("dummy")
    fake_config = {"active_ai_provider": "openai", "openai_model": "gpt-4o",
                   "settings": {"clips_dir": str(tmp_path / "clips"), "clips_per_hour": 12, "min_clip_score": 6},
                   "prompts": {"profiles": {"Default": "P"}}}
    monkeypatch.setattr("src.core.editor.config_manager.load_config", lambda *a, **k: fake_config)
    monkeypatch.setattr("src.core.editor._validate_api_keys", lambda *a, **k: True)
    monkeypatch.setattr("src.core.editor._transcribe_audio_to_segments", lambda *a, **k: [{"start": 0.0, "end": 3600.0, "text": "talk"}])
    monkeypatch.setattr("src.core.editor._generate_clips_with_llm", lambda *a, **k: [
        {"start_time": 100.0, "end_time": 130.0, "virality_score": 8, "reasoning": "a"},
        {"start_time": 400.0, "end_time": 430.0, "virality_score": 5, "reasoning": "weak"},
        {"start_time": 120.0, "end_time": 150.0, "virality_score": 7, "reasoning": "overlaps a better clip"}])
    monkeypatch.setattr("src.core.editor._snap_clips_to_pauses", lambda f, clips, _d, logger: clips)
    monkeypatch.setattr("src.core.editor.extract_clips", lambda f, data, *a, **k: ["x.mp4"])
    logs = []
    editor.process_video(str(video), logger=logs.append)
    line = next(line for line in logs if "Selected" in line)
    assert "Selected 1 of 3 candidate(s) (up to 12 at 12/hour)" in line
    assert "1 below score 6" in line and "1 overlapping a better clip" in line


def test_review_prompt_holds_long_clips_to_a_stricter_standard():
    assert "longer than 120 seconds must be entertaining throughout" in editor.REVIEW_PROMPT


def test_each_window_asks_for_a_full_quota_to_rank_not_up_to_n(monkeypatch):
    prompts = []

    def fake_create(**kwargs):
        prompts.append(kwargs["messages"][1]["content"])
        return _openai_reply('{"clips": []}')

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = fake_create
    monkeypatch.setattr("src.core.editor.OpenAI", lambda **kwargs: mock_client)
    config = {"openai": {"api_key": "k"}, "settings": {"clips_per_hour": 12}}
    editor._generate_clips_with_llm(_long_segments(40), config, "gpt-5.5", "SYS", None)
    window_prompts = [p for p in prompts if not p.startswith("Below are")]
    assert window_prompts
    import re

    quotas = [int(re.search(r"Return (\d+) candidate highlights", p).group(1)) for p in window_prompts]
    assert quotas[0] == 5 and all(q >= 2 for q in quotas) and quotas[-1] <= quotas[0]  # the shorter last window asks for fewer
    for prompt in window_prompts:
        assert "RANK, not to gatekeep" in prompt
        assert "fewer if" not in prompt and "only return candidates scoring 5 or higher" not in prompt
        assert "never inflate a score" in prompt


def test_default_prompt_asks_for_ranked_candidates_with_honest_low_scores():
    from src.core.config import get_default_config

    prompt = get_default_config()["prompts"]["profiles"]["Default"]
    assert "HOW MANY CANDIDATES" in prompt and "RANK, not to gatekeep" in prompt
    assert "'virality_score': integer 1-10" in prompt
    assert "do NOT return" not in prompt and "Return no more candidates than requested" not in prompt
    assert "never raise a score to fill a slot" in prompt


def test_low_scored_candidates_are_parsed_and_left_for_selection_to_drop():
    clips = editor._parse_json_clips('{"clips": [{"start_time": 10, "end_time": 40, "virality_score": 3, "reasoning": "meh"}]}')
    assert clips[0]["virality_score"] == 3
    from src.core import highlights

    assert highlights.select_clips(clips, duration=3600, min_score=6) == []


# ==============================================================================
# DEEPSEEK REASONING MODE, PARALLEL BATCHES
# ==============================================================================
def test_reasoning_flag_defaults_on_for_deepseek_only():
    assert editor._LLMRoute({"active_ai_provider": "deepseek"}, "deepseek-v4-flash").thinking is True
    assert editor._LLMRoute({"active_ai_provider": "deepseek", "settings": {"deepseek_thinking": False}}, "deepseek-v4-flash").thinking is False
    assert editor._LLMRoute({"settings": {"deepseek_thinking": True}}, "gpt-5.5").thinking is False  # nothing to switch elsewhere
    assert editor._LLMRoute({"settings": {"deepseek_thinking": True}}, "claude-sonnet-5").thinking is False


def test_deepseek_request_carries_the_configured_reasoning_mode(monkeypatch):
    for setting, expected in ((True, "enabled"), (False, "disabled")):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _openai_reply('{"clips": []}')
        monkeypatch.setattr("src.core.editor.OpenAI", lambda **kwargs: mock_client)
        config = {"openai": {"api_key": "k"}, "deepseek": {"api_key": "d"}, "active_ai_provider": "deepseek", "settings": {"deepseek_thinking": setting}}
        editor._generate_clips_with_llm([{"start": 0.0, "end": 30.0, "text": "speech"}], config, "deepseek-v4-flash", "P", None)
        assert mock_client.chat.completions.create.call_args[1]["extra_body"] == {"thinking": {"type": expected}}


def test_log_says_whether_deepseek_runs_with_reasoning(monkeypatch):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _openai_reply('{"clips": []}')
    monkeypatch.setattr("src.core.editor.OpenAI", lambda **kwargs: mock_client)
    segments = [{"start": 0.0, "end": 30.0, "text": "speech"}]
    on, off, other = [], [], []
    base = {"openai": {"api_key": "k"}, "deepseek": {"api_key": "d"}, "active_ai_provider": "deepseek"}
    editor._generate_clips_with_llm(segments, dict(base, settings={"deepseek_thinking": True}), "deepseek-v4-flash", "P", on.append)
    editor._generate_clips_with_llm(segments, dict(base, settings={"deepseek_thinking": False}), "deepseek-v4-flash", "P", off.append)
    editor._generate_clips_with_llm(segments, {"openai": {"api_key": "k"}}, "gpt-5.5", "P", other.append)
    assert any("DeepSeek reasoning mode: ON" in line and f"{editor.LLM_WORKERS} requests in parallel" in line for line in on)
    assert any("DeepSeek reasoning mode: OFF" in line for line in off) and not any("in parallel" in line for line in off)
    assert not any("reasoning mode" in line for line in other)


def test_run_parallel_keeps_workers_busy_and_yields_results_as_they_finish():
    import threading
    import time

    lock = threading.Lock()
    state = {"now": 0, "max": 0}
    starts = {}
    t0 = time.monotonic()

    def worker(x):
        with lock:
            state["now"] += 1
            state["max"] = max(state["max"], state["now"])
            starts[x] = time.monotonic() - t0
        time.sleep(0.5 if x == 1 else 0.05)  # item 1 is slow
        with lock:
            state["now"] -= 1
        return x * 10

    out = list(editor._run_parallel([1, 2, 3, 4, 5], worker, 2, None))
    assert sorted(out) == [(1, 10), (2, 20), (3, 30), (4, 40), (5, 50)]
    assert [item for item, _r in out][-1] == 1  # the slow one arrives last, the others were not held up behind it
    assert state["max"] == 2
    assert starts[3] < 0.3 and starts[5] < 0.4  # sliding: a new request starts as soon as a slot frees up, not after the slow one

    state["max"] = 0
    assert [r for _i, r in editor._run_parallel([2, 3, 4], worker, 1, None)] == [20, 30, 40] and state["max"] == 1  # one at a time, in order
    assert list(editor._run_parallel([], worker, 4, None)) == []


def test_run_parallel_starts_nothing_more_once_the_consumer_stops_or_cancel_is_set():
    import time

    calls = []

    def counting(x):
        calls.append(x)
        return x

    for _item, _result in editor._run_parallel([1, 2, 3, 4], counting, 1, None):
        break  # e.g. two windows failed in a row: no further request may be sent
    time.sleep(0.2)
    assert calls == [1]

    calls.clear()
    assert list(editor._run_parallel([1, 2, 3], counting, 2, lambda: True)) == [] and calls == []  # cancelled before the start


def test_heartbeat_fires_while_requests_are_in_flight(monkeypatch):
    import time

    monkeypatch.setattr(editor, "HEARTBEAT_SECONDS", 0.15)
    beats = []

    def slow(x):
        time.sleep(0.6)
        return x

    results = list(editor._run_parallel([1, 2, 3], slow, 3, None, lambda seconds, done, total: beats.append((round(seconds, 1), done, total))))
    assert sorted(r for _i, r in results) == [1, 2, 3]
    assert len(beats) >= 2 and all(total == 3 and done == 0 for _s, done, total in beats)
    assert beats == sorted(beats)  # elapsed time only grows


def test_cancel_returns_immediately_even_though_requests_are_still_in_flight():
    import time

    started = time.monotonic()

    def slow(x):
        time.sleep(3.0)  # an AI request that would take 3 s to answer
        return x

    def cancel():
        return time.monotonic() - started > 0.3

    out = list(editor._run_parallel([1, 2, 3, 4], slow, 4, cancel))
    assert out == [] and time.monotonic() - started < 1.5  # returned long before the requests finished


def test_reasoning_mode_runs_windows_and_reviews_in_parallel(monkeypatch):
    import threading
    import time

    lock = threading.Lock()
    state = {"now": 0, "max": 0}

    def fake_create(**kwargs):
        user = kwargs["messages"][1]["content"]
        with lock:
            state["now"] += 1
            state["max"] = max(state["max"], state["now"])
        time.sleep(0.05)
        with lock:
            state["now"] -= 1
        if user.startswith("Below are"):
            n = user.count("\nid=") + 1
            return _openai_reply(json.dumps({"scores": [{"id": i, "score": 8} for i in range(n)]}))
        start = _window_index(user) * 600.0
        return _openai_reply(json.dumps({"clips": [{"start_time": start, "end_time": start + 30, "virality_score": 7, "reasoning": "x"}]}))

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = fake_create
    monkeypatch.setattr("src.core.editor.OpenAI", lambda **kwargs: mock_client)
    config = {"openai": {"api_key": "k"}, "deepseek": {"api_key": "d"}, "active_ai_provider": "deepseek", "settings": {"deepseek_thinking": True}}
    clips = editor._generate_clips_with_llm(_long_segments(40), config, "deepseek-v4-flash", "P", None)
    assert state["max"] > 1  # windows really ran side by side
    starts = sorted(c["start_time"] for c in clips)
    assert len(clips) >= 4 and starts == sorted(starts)

    state["max"] = 0
    off_config = dict(config, settings={"deepseek_thinking": False})
    editor._generate_clips_with_llm(_long_segments(40), off_config, "deepseek-v4-flash", "P", None)
    assert state["max"] == 1  # without reasoning the requests stay sequential


def test_parallel_review_keeps_clip_order_and_stops_after_repeated_failures(monkeypatch):
    import threading
    import time as real_time

    good_answered = threading.Event()

    def fake_create(**kwargs):
        user = kwargs["messages"][1]["content"]
        if "from 50.0s to 90.0s" in user:  # clip 1
            reply = _openai_reply(_scenes_reply([(20.0, 60.0, "setup"), (60.0, 90.0, "joke")], 0, 1))
            good_answered.set()
            return reply
        good_answered.wait(2.0)  # the other clips fail only after the good answer exists (results arrive in completion order)
        real_time.sleep(0.15)
        raise RuntimeError("boom")

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = fake_create
    monkeypatch.setattr("src.core.editor.OpenAI", lambda **kwargs: mock_client)
    monkeypatch.setattr("time.sleep", lambda s: None)
    route = editor._LLMRoute({"openai": {"api_key": "k"}, "deepseek": {"api_key": "d"}, "active_ai_provider": "deepseek"}, "deepseek-v4-flash")
    assert route.thinking
    clips = [{"start_time": 50.0 + 100 * i, "end_time": 90.0 + 100 * i, "virality_score": 7, "reasoning": "r"} for i in range(6)]
    logs = []
    result = REAL_REVIEW(route, clips, _entries(), 800.0, logs.append, None, [True])
    assert [c["end_time"] for c in result] == [c["end_time"] for c in clips]  # order kept
    assert result[0]["start_time"] == 30.0  # the one good answer was applied to the right clip (setup growth is capped at 20 s)
    assert all(r["start_time"] == c["start_time"] for r, c in zip(result[1:], clips[1:]))
    assert any(f"reasoning ON, {editor.LLM_WORKERS} in parallel" in line for line in logs)
    assert any("keeps failing" in line for line in logs)


# ==============================================================================
# PROGRESS MESSAGES AND IMMEDIATE CANCEL WHILE THE AI IS THINKING
# ==============================================================================
def test_heartbeat_waits_for_real_silence_answers_count_as_progress(monkeypatch):
    import time

    monkeypatch.setattr(editor, "HEARTBEAT_SECONDS", 0.3)
    beats = []
    durations = {1: 0.1, 2: 0.2, 3: 0.3, 4: 1.0}  # answers keep arriving until 0.3 s, then 0.7 s of silence

    def worker(x):
        time.sleep(durations[x])
        return x

    list(editor._run_parallel([1, 2, 3, 4], worker, 4, None, lambda seconds, done, total: beats.append((round(seconds, 2), done))))
    assert beats, "the long silence at the end must be reported"
    assert beats[0][0] >= 0.55 and beats[0][1] == 3  # only after the last answer + a full quiet period, with 3 already done


def test_review_logs_only_changes_and_cancel_stops_waiting(monkeypatch):
    import time

    def fake_create(**kwargs):
        time.sleep(0.2)
        user = kwargs["messages"][1]["content"]
        if "from 50.0s to 90.0s" in user:
            return _openai_reply(_scenes_reply([(20.0, 60.0, "setup"), (60.0, 90.0, "joke")], 0, 1, "needs the setup"))
        return _openai_reply(_scenes_reply([(150.0, 190.0, "same")], 0, 0))

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = fake_create
    monkeypatch.setattr("src.core.editor.OpenAI", lambda **kwargs: mock_client)
    route = editor._LLMRoute({"openai": {"api_key": "k"}, "deepseek": {"api_key": "d"}, "active_ai_provider": "deepseek"}, "deepseek-v4-flash")
    clips = [{"start_time": 50.0, "end_time": 90.0, "virality_score": 7, "reasoning": "r"},
             {"start_time": 150.0, "end_time": 190.0, "virality_score": 7, "reasoning": "r"}]
    logs = []
    REAL_REVIEW(route, clips, _entries(), 800.0, logs.append, None, [True])
    assert not any(line.startswith("📨") for line in logs)
    assert any("Clip 1: 50-90s -> 30-90s" in line and "needs the setup" in line for line in logs)  # 30 s of setup asked, 20 s allowed
    assert not any("kept" in line and "Clip 2" in line for line in logs)  # clips that stay as they are are not announced one by one
    assert any("1 of 2 clip(s) adjusted" in line for line in logs)

    slow_started = time.monotonic()

    def very_slow(**kwargs):
        time.sleep(3.0)
        return _openai_reply("{}")

    mock_client.chat.completions.create.side_effect = very_slow
    logs.clear()
    result = REAL_REVIEW(route, clips, _entries(), 800.0, logs.append, lambda: time.monotonic() - slow_started > 0.3, [True])
    assert result == clips and time.monotonic() - slow_started < 2.0
    assert any("Cancelled" in line for line in logs)


# ==============================================================================
# STRUCTURED LOG: STAGE HEADERS AND TOTAL TIME
# ==============================================================================
def _stage_run(tmp_path, monkeypatch, review, with_clips=True):
    video = tmp_path / "gameplay.mp4"
    video.write_text("dummy")
    fake_config = {"active_ai_provider": "openai", "openai_model": "gpt-4o", "settings": {"clips_dir": str(tmp_path / "clips"), "review_boundaries": review},
                   "prompts": {"profiles": {"Default": "P"}}}
    monkeypatch.setattr("src.core.editor.config_manager.load_config", lambda *a, **k: fake_config)
    monkeypatch.setattr("src.core.editor._validate_api_keys", lambda *a, **k: True)
    monkeypatch.setattr("src.core.editor._transcribe_audio_to_segments", lambda *a, **k: [{"start": 0.0, "end": 600.0, "text": "talk"}])
    monkeypatch.setattr("src.core.editor._generate_clips_with_llm", lambda *a, **k: [
        {"start_time": 100.0, "end_time": 130.0, "virality_score": 8, "reasoning": "r"}] if with_clips else [])
    monkeypatch.setattr("src.core.editor._snap_clips_to_pauses", lambda f, clips, _d, logger: clips)
    monkeypatch.setattr("src.core.editor.extract_clips", lambda f, data, *a, **k: ["x.mp4"])
    logs = []
    editor.process_video(str(video), logger=logs.append)
    return logs


def test_pipeline_log_is_divided_into_numbered_stages(tmp_path, monkeypatch):
    logs = _stage_run(tmp_path, monkeypatch, review=True)
    headers = [line for line in logs if line.startswith("══")]
    assert headers == ["══ 1/4 · 🎙️ Transcript & audio ══", "══ 2/4 · 🤖 AI finds and ranks the best moments ══",
                       "══ 3/4 · 🔍 AI reviews the clip boundaries ══", "══ 4/4 · ✂️ Cutting & exporting ══"]
    order = [logs.index(h) for h in headers]
    assert order == sorted(order)
    assert any("Selected 1 of 1" in line for line in logs[order[1]:order[2]])  # the selection line sits under the AI stage
    assert any("Clip lengths" in line for line in logs[order[3]:])
    assert any(line.startswith("✨ Successfully exported 1 file(s)") and "(total " in line for line in logs)


def test_stage_count_shrinks_when_the_review_is_off_and_nothing_to_cut(tmp_path, monkeypatch):
    logs = _stage_run(tmp_path, monkeypatch, review=False)
    assert [line for line in logs if line.startswith("══")] == ["══ 1/3 · 🎙️ Transcript & audio ══", "══ 2/3 · 🤖 AI finds and ranks the best moments ══",
                                                              "══ 3/3 · ✂️ Cutting & exporting ══"]


def test_no_clips_means_no_cutting_stage(tmp_path, monkeypatch):
    logs = _stage_run(tmp_path, monkeypatch, review=True, with_clips=False)
    headers = [line for line in logs if line.startswith("══")]
    assert len(headers) == 2 and not any("Cutting" in h for h in headers)


def test_format_duration():
    assert editor._format_duration(42.4) == "42s" and editor._format_duration(60) == "1m 00s" and editor._format_duration(372.6) == "6m 13s"


def test_prompts_explain_the_pause_markers_and_the_closing_remark():
    assert "⏸" in editor.WINDOW_PROMPT and "continuous speech" in editor.WINDOW_PROMPT
    assert "⏸" in editor.REVIEW_PROMPT and "closing remark" in editor.REVIEW_PROMPT
    assert "one line too early feels cut off" in editor.REVIEW_PROMPT


def test_window_transcripts_carry_pause_markers(monkeypatch):
    prompts = []

    def fake_create(**kwargs):
        prompts.append(kwargs["messages"][1]["content"])
        return _openai_reply('{"clips": []}')

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = fake_create
    monkeypatch.setattr("src.core.editor.OpenAI", lambda **kwargs: mock_client)
    segments = [{"start": 0.0, "end": 3.0, "text": "a."}, {"start": 5.0, "end": 8.0, "text": "b."}]
    editor._generate_clips_with_llm(segments, {"openai": {"api_key": "k"}}, "gpt-5.5", "P", None)
    assert "⏸ 2.0s" in prompts[0]


def test_process_video_finishes_a_sentence_the_ai_cut_in_half(tmp_path, monkeypatch):
    video = tmp_path / "gameplay.mp4"
    video.write_text("dummy")
    fake_config = {"active_ai_provider": "openai", "openai_model": "gpt-4o", "settings": {"clips_dir": str(tmp_path / "clips"), "review_boundaries": False},
                   "prompts": {"profiles": {"Default": "P"}}}
    monkeypatch.setattr("src.core.editor.config_manager.load_config", lambda *a, **k: fake_config)
    monkeypatch.setattr("src.core.editor._validate_api_keys", lambda *a, **k: True)
    segments = [{"start": 100.0, "end": 124.0, "text": "долгое начало без точки"}, {"start": 124.0, "end": 129.0, "text": "Нет, смотри, короче, первый, это"},
                {"start": 129.9, "end": 134.0, "text": "первая лампочка."}, {"start": 200.0, "end": 205.0, "text": "Другое."}]
    monkeypatch.setattr("src.core.editor._transcribe_audio_to_segments", lambda *a, **k: segments)
    monkeypatch.setattr("src.core.editor._generate_clips_with_llm", lambda *a, **k: [
        {"start_time": 100.0, "end_time": 129.0, "virality_score": 8, "reasoning": "r"}])
    monkeypatch.setattr("src.core.editor._snap_clips_to_pauses", lambda f, clips, _d, logger: clips)
    seen = {}

    def fake_extract(f, data, *a, **k):
        seen["clips"] = data["clips"]
        return ["x.mp4"]

    monkeypatch.setattr("src.core.editor.extract_clips", fake_extract)
    editor.process_video(str(video))
    assert seen["clips"][0]["end_time"] == 134.0


# ==============================================================================
# EXPORT: SHORT AUDIO FADES SO A CUT NEVER SOUNDS CHOPPED
# ==============================================================================
def _export_command(tmp_path, monkeypatch, tracks, downmix=True):
    commands = []

    def run(cmd, **kwargs):
        commands.append(cmd)
        return MagicMock(returncode=0, stderr=b"")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(os.path, "isfile", lambda p: True)
    monkeypatch.setattr("src.core.editor._count_audio_streams", lambda f: tracks)
    monkeypatch.setattr("src.core.editor.config_manager.load_config", lambda *a, **k: {"settings": {"audio_downmix": downmix, "hardware_encoding": False}})
    editor.extract_clips("source.mp4", {"clips": [{"start_time": 100.0, "end_time": 130.0, "virality_score": 8, "reasoning": "r", "pre_roll": 0.0, "post_roll": 0.0}]},
                         str(tmp_path), logger=lambda m: None)
    return next(c for c in commands if "-t" in c and "source.mp4" in c)


def test_export_fades_the_audio_in_and_out_multi_track(tmp_path, monkeypatch):
    cmd = _export_command(tmp_path, monkeypatch, tracks=3)
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "amix=inputs=3" in graph and "[amixed];[amixed]afade=t=in:st=0:d=0.04,afade=t=out:st=29.850:d=0.15[aout]" in graph


def test_export_fades_the_audio_single_track_and_leaves_copied_audio_alone(tmp_path, monkeypatch):
    cmd = _export_command(tmp_path, monkeypatch, tracks=1)
    assert cmd[cmd.index("-af") + 1] == "afade=t=in:st=0:d=0.04,afade=t=out:st=29.850:d=0.15"
    copied = _export_command(tmp_path, monkeypatch, tracks=1, downmix=False)
    assert "-af" not in copied and "afade" not in " ".join(copied) and copied[copied.index("-c:a") + 1] == "copy"
