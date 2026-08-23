import contextlib
import hashlib
import io
import json
import os
import re
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from openai import OpenAI
import whisper

from src.core import config as config_manager
from src.utils.hardware import get_hardware_status, log_hardware_info
from src.utils.paths import get_app_data_path, get_ffmpeg_path, get_ffprobe_path, inject_bin_to_path

inject_bin_to_path()
FFMPEG_PATH = get_ffmpeg_path()
FFPROBE_PATH = get_ffprobe_path()

try:
    from google import genai 
    from google.genai import types as genai_types 
    HAS_GEMINI = True
except ImportError:
    genai = None
    genai_types = None
    HAS_GEMINI = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

class WhisperProgressStream(io.StringIO):
    def __init__(self, logger: Optional[Callable[[str], None]]):
        super().__init__()
        self.logger = logger
        self.counter = 0

    def write(self, s: str) -> int:
        raw_msg = s.strip()
        if raw_msg and "-->" in raw_msg:
            # Throttle to log only every 10th segment to prevent UI spam lag
            if self.counter % 10 == 0:
                if "]" in raw_msg:
                    t_part, _, clean_msg = raw_msg.partition("]")
                    clean_msg = clean_msg.strip()
                    timestamp = t_part.replace("[", "").partition("-->")[0].strip()
                else:
                    clean_msg = raw_msg
                    timestamp = raw_msg.partition("-->")[0].strip()
                if self.logger:
                    self.logger(f"⏳ Processed up to {timestamp}: {clean_msg[:40]}...")
            self.counter += 1
        return super().write(s)

def analyze_audio_peaks(audio_array: np.ndarray, segments: List[Dict[str, Any]], sample_rate: int = 16000, peak_detection: bool = True, combat_detection: bool = True) -> List[Dict[str, Any]]:
    """Calculates RMS loudness and detects combat transients for each Whisper segment."""
    enhanced_segments = []
    
    for seg in segments:
        start_idx = int(seg['start'] * sample_rate)
        end_idx = int(seg['end'] * sample_rate)
        
        # Ensure indices stay within array bounds
        start_idx = max(0, min(start_idx, len(audio_array) - 1))
        end_idx = max(start_idx + 1, min(end_idx, len(audio_array)))
        
        chunk = audio_array[start_idx:end_idx]
        
        rms = 0.0
        is_combat = False
        
        if len(chunk) > 0:
            if peak_detection:
                rms = float(np.sqrt(np.mean(chunk**2)))
            
            if combat_detection and len(chunk) >= 512:
                # Transient / Combat detection
                hop_size = 256
                window_size = 512
                sub_chunks = [chunk[i:i+window_size] for i in range(0, len(chunk) - window_size, hop_size)]
                if sub_chunks:
                    sub_rms = np.array([np.sqrt(np.mean(sc**2)) for sc in sub_chunks])
                    mean_sub = np.mean(sub_rms)
                    max_sub = np.max(sub_rms)
                    # A sharp spike (gunshot/hit) typically exceeds 3.5x the local segment average
                    if mean_sub > 0.01 and (max_sub / (mean_sub + 1e-5)) > 3.5:
                        is_combat = True

        seg_copy = dict(seg)
        seg_copy['rms'] = rms
        seg_copy['is_combat'] = is_combat
        enhanced_segments.append(seg_copy)
        
    if not peak_detection and not combat_detection:
        return enhanced_segments

    # Normalize RMS to a 0-100 scale across the entire video
    all_rms = [s['rms'] for s in enhanced_segments if 'rms' in s]
    max_overall_rms = max(all_rms) if all_rms else 1.0
    if max_overall_rms == 0: max_overall_rms = 1.0
    
    for seg in enhanced_segments:
        raw_rms = seg.get('rms', 0.0)
        loudness_score = int((raw_rms / max_overall_rms) * 100)
        
        tags = []
        if peak_detection:
            tags.append(f"[LOUDNESS: {loudness_score}%]")
        if combat_detection and seg.get('is_combat', False):
            tags.append("[ACTION: COMBAT]")
            
        tag_str = " ".join(tags)
        if tag_str:
            seg['text'] = f"{tag_str} {seg['text']}"
            
    return enhanced_segments

def sanitize_filename(name: str) -> str:
    """Removes invalid filesystem characters from clip names."""
    return re.sub(r'[\\/*?:"<>|]', "", name).strip()

def _count_audio_streams(video_file: str) -> int:
    """Uses ffprobe to count how many audio streams exist in the container."""
    startupinfo = None
    if os.name == 'nt' and hasattr(subprocess, 'STARTUPINFO'):
        startupinfo = subprocess.STARTUPINFO()
        if hasattr(subprocess, 'STARTF_USESHOWWINDOW'):
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        if hasattr(subprocess, 'SW_HIDE'):
            startupinfo.wShowWindow = subprocess.SW_HIDE

    cmd = [
        FFPROBE_PATH,
        "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=index",
        "-of", "csv=p=0",
        video_file
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, startupinfo=startupinfo)
        if result.returncode == 0 and result.stdout.strip():
            return len(result.stdout.strip().splitlines())
    except Exception:
        pass
    return 1

def extract_audio_hidden(video_file: str) -> np.ndarray:
    """Extracts 16kHz mono audio directly into memory using FFmpeg.

    Automatically detects and merges all audio streams (e.g. multi-track OBS
    recordings) into a single mono track via amix so Whisper and audio-peak
    analysis hear every track, not just the first one.
    """
    startupinfo = None
    if os.name == 'nt' and hasattr(subprocess, 'STARTUPINFO'):
        startupinfo = subprocess.STARTUPINFO()
        if hasattr(subprocess, 'STARTF_USESHOWWINDOW'):
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        if hasattr(subprocess, 'SW_HIDE'):
            startupinfo.wShowWindow = subprocess.SW_HIDE

    num_audio = _count_audio_streams(video_file)

    cmd = [FFMPEG_PATH, "-nostdin", "-threads", "0", "-i", video_file]

    if num_audio > 1:
        # Build amix filter to merge all audio streams into one
        filter_parts = []
        for idx in range(num_audio):
            filter_parts.append(f"[0:a:{idx}]")
        filter_complex = "".join(filter_parts) + f"amix=inputs={num_audio}:duration=longest:normalize=0[aout]"
        cmd.extend(["-filter_complex", filter_complex, "-map", "[aout]"])

    cmd.extend([
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-"
    ])

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        startupinfo=startupinfo
    )

    out, _ = process.communicate()
    if process.returncode != 0:
        raise RuntimeError("FFmpeg audio extraction failed.")

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

def _calculate_crop_filter(_aspect_ratio_str: str, mode: str, custom_x: str, custom_y: str, custom_w: str, custom_h: str) -> str:
    """Computes ffmpeg crop / pad filter for vertical video production."""
    crop_w, crop_h = "ih*(9/16)", "ih"
    crop_x, crop_y = "(iw-ow)/2", "0"

    try:
        if mode == "Standard Center Crop":
            crop_w, crop_h = "ih*(9/16)", "ih"
            crop_x, crop_y = "(iw-ow)/2", "0"
        elif mode == "Left-Third (Facecam)":
            crop_w, crop_h = "ih*(9/16)", "ih"
            crop_x, crop_y = "0", "0"
        elif mode == "Right-Third":
            crop_w, crop_h = "ih*(9/16)", "ih"
            crop_x, crop_y = "iw-ow", "0"
        elif mode == "Blurred Background (Portrait)":
            return "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=20:5[bg];[0:v]scale=1080:1920:force_original_aspect_ratio=decrease[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2"
        elif mode == "Custom Coordinates":
            crop_w = str(int(custom_w)) if custom_w.isdigit() else "400"
            crop_h = str(int(custom_h)) if custom_h.isdigit() else "225"
            crop_x = str(int(custom_x)) if custom_x.isdigit() else "0"
            crop_y = str(int(custom_y)) if custom_y.isdigit() else "0"
    except Exception:
        crop_w, crop_h = "ih*(9/16)", "ih"
        crop_x, crop_y = "(iw-ow)/2", "0"

    return f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y},scale=1080:1920"

def extract_clips(input_file: str, clips_data: Dict[str, Any], output_dir: str, logger: Optional[Callable[[str], None]] = None, is_cancelled: Optional[Callable[[], bool]] = None) -> List[str]:
    """Cuts clips via FFmpeg based on AI timestamps."""
    if not os.path.isfile(input_file):
        if logger: logger(f"❌ Source video file not found: {input_file}")
        return []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(input_file))[0]
    config = config_manager.load_config()
    
    vr_stabilize = config.get("settings", {}).get("vr_stabilization", False)
    hw_encode = config.get("settings", {}).get("hardware_encoding", False)
    downmix_audio = config.get("settings", {}).get("audio_downmix", True)
    vertical_export = config.get("settings", {}).get("vertical_export", False)
    vertical_mode = config.get("settings", {}).get("vertical_mode", "Standard Center Crop")
    
    crop_x = config.get("settings", {}).get("crop_x", "0")
    crop_y = config.get("settings", {}).get("crop_y", "0")
    crop_w = config.get("settings", {}).get("crop_w", "400")
    crop_h = config.get("settings", {}).get("crop_h", "225")

    created_files = []
    video_filters = []
    
    if vr_stabilize:
        video_filters.append("deshake")
        if logger: logger("🌀 VR Deshake filter active.")

    if vertical_export:
        vf_crop = _calculate_crop_filter("9:16", vertical_mode, crop_x, crop_y, crop_w, crop_h)
        video_filters.append(vf_crop)
        if logger: logger(f"📱 Vertical 9:16 Mode active ({vertical_mode}).")

    final_filter_str = ",".join(video_filters) if video_filters else None

    # Detect multi-track audio (OBS, etc.) and build the correct merge strategy
    num_audio = _count_audio_streams(input_file) if downmix_audio else 1
    use_amix = downmix_audio and num_audio > 1

    if use_amix and logger:
        logger(f"🔊 Detected {num_audio} audio tracks — merging all into stereo via amix.")

    startupinfo = None
    if os.name == 'nt' and hasattr(subprocess, 'STARTUPINFO'):
        startupinfo = subprocess.STARTUPINFO()
        if hasattr(subprocess, 'STARTF_USESHOWWINDOW'):
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        if hasattr(subprocess, 'SW_HIDE'):
            startupinfo.wShowWindow = subprocess.SW_HIDE

    clips_list = clips_data.get("clips", [])
    total_clips = len(clips_list)

    if logger and total_clips > 0:
        if hw_encode:
            logger("🚀 FFmpeg Video Processing: GPU Hardware Acceleration active (NVENC/CUDA)")
        else:
            logger("⚠️ FFmpeg Video Processing: CPU Mode active (Enable GPU Hardware Encoding in Settings for faster exports)")

    for i, clip in enumerate(clips_list):
        if is_cancelled and is_cancelled():
            if logger: logger("🛑 Cancellation requested. Halting FFmpeg clip exports.")
            break

        start_raw = float(clip["start_time"])
        end_raw = float(clip["end_time"])

        # Smart padding: 0.75s pre-roll and 1.25s post-roll so speech and laughter never get clipped
        start = max(0.0, start_raw - 0.75)
        end = end_raw + 1.25
        duration = max(0.5, end - start)
        score = clip.get("virality_score", 5)
        reason = clip.get("reasoning", "viral moment")

        clean_reason = sanitize_filename(reason)[:30]
        vert_tag = "_Vertical" if vertical_export else ""
        out_filename = f"{base_name}_clip_{i+1}_score{score}_{clean_reason}{vert_tag}.mp4"
        out_path = os.path.join(output_dir, out_filename)

        meta_path = os.path.splitext(out_path)[0] + ".json"
        meta_data = {
            "source_video": input_file,
            "clip_index": i + 1,
            "start_time": round(start, 2),
            "end_time": round(end, 2),
            "duration": round(duration, 2),
            "virality_score": score,
            "reasoning": reason,
            "is_vertical": vertical_export,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        if logger:
            reason_preview = f' — "{reason}"' if reason and reason != "viral moment" else ""
            logger(f"🎞️ [{i+1}/{total_clips}] Cutting Clip ({start:.1f}s to {end:.1f}s, {duration:.1f}s) -> Score {score}/10{reason_preview}...")

        cmd = [FFMPEG_PATH, "-y"]
        if hw_encode:
            cmd.extend(["-hwaccel", "cuda"])

        cmd.extend([
            "-ss", str(start),
            "-t", str(duration),
            "-i", input_file
        ])

        if final_filter_str:
            cmd.extend(["-vf", final_filter_str])

        # Audio: merge multiple streams via filter_complex, or just copy/encode
        if use_amix:
            filter_parts = []
            for idx in range(num_audio):
                filter_parts.append(f"[0:a:{idx}]")
            amix_filter = "".join(filter_parts) + f"amix=inputs={num_audio}:duration=longest:normalize=0[aout]"
            cmd.extend(["-filter_complex", amix_filter, "-map", "0:v", "-map", "[aout]"])
            cmd.extend(["-c:a", "aac", "-b:a", "192k"])
        elif downmix_audio:
            # Single audio stream — just encode to AAC (no broken pan filter)
            cmd.extend(["-c:a", "aac", "-b:a", "192k"])
        else:
            cmd.extend(["-c:a", "copy"])

        if hw_encode:
            cmd.extend(["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23"])
        else:
            if final_filter_str:
                cmd.extend(["-c:v", "libx264", "-crf", "18", "-preset", "fast"])
            else:
                cmd.extend(["-c:v", "copy"])

        cmd.extend(["-avoid_negative_ts", "make_zero", out_path])

        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, startupinfo=startupinfo)
        
        if res.returncode == 0:
            try:
                with open(meta_path, "w", encoding="utf-8") as mf:
                    json.dump(meta_data, mf, indent=4, ensure_ascii=False)
            except Exception as e:
                if logger: logger(f"⚠️ Failed to write clip metadata JSON: {e}")

            thumb_path = os.path.splitext(out_path)[0] + ".jpg"
            thumb_time = min(1.0, duration / 2)
            thumb_cmd = [
                FFMPEG_PATH, "-y",
                "-ss", str(thumb_time),
                "-i", out_path,
                "-vframes", "1",
                "-q:v", "3",
                thumb_path
            ]
            thumb_res = subprocess.run(thumb_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, startupinfo=startupinfo)
            if thumb_res.returncode != 0 and logger:
                logger(f"⚠️ Failed to generate thumbnail for clip {i+1}.")

            created_files.append(out_path)
            if logger:
                logger(f"✅ [{i+1}/{total_clips}] Rendered: {out_filename}")
        else:
            if logger:
                err_lines = (res.stderr or b"").decode("utf-8", errors="replace").strip().splitlines()
                tail = "\n".join(err_lines[-3:]) if err_lines else "unknown error"
                logger(f"❌ FFmpeg export failed for clip {i+1}:\n{tail}")

    return created_files

def _validate_api_keys(config: Dict[str, Any], chat_model: str, logger: Optional[Callable[[str], None]]) -> bool:
    openai_key = config.get("openai", {}).get("api_key", "").strip()
    openai_base_url = config.get("openai", {}).get("base_url", "").strip()
    deepseek_key = config.get("deepseek", {}).get("api_key", "").strip()
    google_key = config.get("google", {}).get("api_key", "").strip()
    anthropic_key = config.get("anthropic", {}).get("api_key", "").strip()
    xai_key = config.get("xai", {}).get("api_key", "").strip()
    active_provider = config.get("active_ai_provider", "openai")

    is_gemini_model = (chat_model.startswith("gemini") or "gemini" in chat_model) and "openrouter" not in chat_model and not openai_base_url
    is_anthropic_model = chat_model.startswith("claude") and "openrouter" not in chat_model and not openai_base_url
    is_openrouter = "openrouter" in chat_model or (openai_base_url != "" and "deepseek" not in openai_base_url and "x.ai" not in openai_base_url)
    is_grok_model = chat_model.startswith("grok") or active_provider == "xai"
    is_deepseek_model = "deepseek" in chat_model.lower() or active_provider == "deepseek" or "deepseek" in (openai_base_url or "").lower()

    if is_gemini_model and not is_openrouter:
        if not google_key:
            if logger: logger("❌ Error: Google API Key not set for Gemini model.")
            return False
    elif is_anthropic_model:
        if not anthropic_key:
            if logger: logger("❌ Error: Anthropic API Key not set.")
            return False
    elif is_grok_model:
        if not xai_key:
            if logger: logger("❌ Error: Grok/xAI API Key not set.")
            return False
    elif is_deepseek_model:
        if not deepseek_key and not openai_key:
            if logger: logger("❌ Error: DeepSeek API Key not set.")
            return False
    else:
        if not openai_key:
            if logger: logger("❌ Error: OpenAI/Custom API Key not set.")
            return False
    return True

def _clean_and_merge_clips(clips: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sanitizes, validates timestamps, and merges accidental micro-overlaps or contiguous fragments."""
    if not clips:
        return []

    valid_clips = []
    for c in clips:
        if not isinstance(c, dict):
            continue
        try:
            start = float(c.get("start_time", 0))
            end = float(c.get("end_time", 0))
            score = int(c.get("virality_score", 7))
            reasoning = str(c.get("reasoning", "")).strip()
        except (ValueError, TypeError):
            continue

        if start < 0 or end <= start:
            continue
        if (end - start) < 2.0:
            continue

        valid_clips.append({
            "start_time": round(start, 2),
            "end_time": round(end, 2),
            "virality_score": score,
            "reasoning": reasoning
        })

    valid_clips.sort(key=lambda x: x["start_time"])

    merged: List[Dict[str, Any]] = []
    for clip in valid_clips:
        if not merged:
            merged.append(clip)
            continue

        prev = merged[-1]
        if clip["start_time"] <= prev["end_time"] + 2.0:
            combined_duration = max(prev["end_time"], clip["end_time"]) - prev["start_time"]
            if combined_duration <= 85.0:
                prev["end_time"] = max(prev["end_time"], clip["end_time"])
                prev["virality_score"] = max(prev["virality_score"], clip["virality_score"])
                if clip["reasoning"] and clip["reasoning"] not in prev["reasoning"]:
                    prev["reasoning"] = f"{prev['reasoning']} | {clip['reasoning']}"
                continue

        merged.append(clip)

    return merged

def _parse_json_clips(raw_text: Any) -> Optional[List[Dict[str, Any]]]:
    """Robustly parses a JSON string containing clips."""
    if not raw_text or not isinstance(raw_text, str):
        return None
    
    text = raw_text.strip()
    if not text:
        return None

    # Strip markdown code fences if present
    if text.startswith("```json"):
        text = text.partition("```json")[2].partition("```")[0].strip()
    elif text.startswith("```"):
        text = text.partition("```")[2].partition("```")[0].strip()
    elif "```json" in text:
        match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
        if match:
            text = match.group(1).strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict) and "clips" in data:
            return _clean_and_merge_clips(data.get("clips", []))
        elif isinstance(data, list):
            return _clean_and_merge_clips(data)
    except json.JSONDecodeError:
        match = re.search(r"(\{[\s\S]*\})", text)
        if match:
            try:
                data = json.loads(match.group(1))
                if isinstance(data, dict) and "clips" in data:
                    return _clean_and_merge_clips(data.get("clips", []))
            except Exception:
                pass
    return None

def _get_transcription_cache_path(file_path: str, whisper_model: str, target_language: Optional[str], audio_peak: bool, combat: bool) -> str:
    """Computes a deterministic hash and cache file path for audio transcriptions."""
    try:
        mtime = os.path.getmtime(file_path)
        size = os.path.getsize(file_path)
    except Exception:
        mtime, size = 0, 0
    
    cache_key = f"{os.path.abspath(file_path)}_{mtime}_{size}_{whisper_model}_{target_language}_{audio_peak}_{combat}"
    file_hash = hashlib.md5(cache_key.encode('utf-8')).hexdigest()
    
    cache_dir = os.path.join(get_app_data_path(), "transcripts")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{file_hash}.json")

def _load_cached_segments(cache_file: str, logger: Optional[Callable[[str], None]] = None) -> Optional[List[Dict[str, Any]]]:
    """Attempts to load previously generated transcription segments from disk cache."""
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                segments = json.load(f)
                if isinstance(segments, list) and segments:
                    if logger:
                        logger(f"⚡ Loaded {len(segments)} cached audio segments from disk! (Skipping Whisper transcription)")
                    return segments
        except Exception as e:
            if logger:
                logger(f"⚠️ Cache read error: {e}")
    return None

def _save_cached_segments(cache_file: str, segments: List[Dict[str, Any]], logger: Optional[Callable[[str], None]] = None) -> None:
    """Saves transcription segments to disk cache for instant reuse on subsequent runs."""
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
        if logger:
            logger(f"💾 Cached transcription to disk ({len(segments)} segments).")
    except Exception as e:
        if logger:
            logger(f"⚠️ Failed to cache transcription: {e}")

LANGUAGE_MAP = {
    "Auto-Detect": None,
    "English": "en",
    "Russian": "ru",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Japanese": "ja",
    "Korean": "ko",
    "Chinese": "zh",
    "Ukrainian": "uk",
    "Polish": "pl",
    "Turkish": "tr"
}

def _transcribe_audio_to_segments(file_path: str, config: Dict[str, Any], logger: Optional[Callable[[str], None]], is_cancelled: Optional[Callable[[], bool]]) -> Optional[List[Dict[str, Any]]]:
    whisper_model_type = config.get("openai", {}).get("whisper_model", "medium")
    language_setting = config.get("openai", {}).get("whisper_language", "Auto-Detect")

    if not language_setting or language_setting == "Auto-Detect":
        target_language = None
    elif language_setting in LANGUAGE_MAP:
        target_language = LANGUAGE_MAP[language_setting]
    else:
        target_language = language_setting.lower() if len(language_setting) == 2 else language_setting[:2].lower()

    audio_peak_detection = config.get("settings", {}).get("audio_peak_detection", True)
    combat_detection = config.get("settings", {}).get("combat_detection", True)

    cache_file = _get_transcription_cache_path(file_path, whisper_model_type, target_language, audio_peak_detection, combat_detection)
    cached_segments = _load_cached_segments(cache_file, logger)
    if cached_segments is not None:
        return cached_segments

    # Hardware check & detailed status logging
    log_hardware_info(logger)
    hw_status = get_hardware_status()
    device = hw_status["device"]
    fp16_enabled = hw_status["fp16_supported"]

    if logger: logger(f"⏳ Loading Whisper '{whisper_model_type}' model into memory...")
    try:
        model = whisper.load_model(whisper_model_type, device=device)
    except Exception as e:
        if logger: logger(f"❌ Failed to load Whisper: {e}")
        return None

    if is_cancelled and is_cancelled(): return None
    
    if logger: logger("🎙️ Extracting audio track silently...")
    try:
        audio_array = extract_audio_hidden(file_path)
    except Exception as e:
        if logger: logger(f"❌ Audio extraction error: {e}")
        return None

    if is_cancelled and is_cancelled(): return None

    lang_desc = target_language if target_language else "Auto-Detect"
    if logger:
        if audio_peak_detection or combat_detection:
            logger(f"🎙️ Transcribing audio [Lang: {lang_desc}, Model: {whisper_model_type}] and analyzing sound patterns...")
        else:
            logger(f"🎙️ Transcribing audio [Lang: {lang_desc}, Model: {whisper_model_type}]...")
            
    try:
        progress_stream = WhisperProgressStream(logger)
        with contextlib.redirect_stdout(progress_stream):
            transcribe_kwargs: Dict[str, Any] = {
                "condition_on_previous_text": False,
                "beam_size": 5 if device == "cuda" else 2,
                "best_of": 5 if device == "cuda" else 2,
                "temperature": (0.0, 0.2, 0.4),
                "fp16": fp16_enabled,
                "verbose": True,
                "initial_prompt": "Gameplay livestream recording with conversational banter, laughter, discord chat, and gaming terms."
            }
            if target_language is not None:
                transcribe_kwargs["language"] = target_language

            result = model.transcribe(audio_array, **transcribe_kwargs)
            
        raw_segments = result.get("segments", [])
        
        if audio_peak_detection or combat_detection:
            segments = analyze_audio_peaks(audio_array, raw_segments, peak_detection=audio_peak_detection, combat_detection=combat_detection)
            if logger: logger("✅ Transcription complete! Audio patterns analyzed.")
        else:
            segments = raw_segments
            if logger: logger("✅ Transcription complete!")
            
        if segments:
            _save_cached_segments(cache_file, segments, logger)
        return segments

    except Exception as e:
        if logger: logger(f"❌ Transcription error: {e}")
        return None

def _generate_clips_with_llm(segments: List[Dict[str, Any]], config: Dict[str, Any], chat_model: str, prompt_text: str, logger: Optional[Callable[[str], None]]) -> List[Dict[str, Any]]:
    openai_key = config.get("openai", {}).get("api_key", "")
    openai_base_url = config.get("openai", {}).get("base_url", "")
    deepseek_key = config.get("deepseek", {}).get("api_key", "")
    deepseek_base_url = config.get("deepseek", {}).get("base_url", "https://api.deepseek.com")
    google_key = config.get("google", {}).get("api_key", "")
    anthropic_key = config.get("anthropic", {}).get("api_key", "")
    xai_key = config.get("xai", {}).get("api_key", "")
    active_provider = config.get("active_ai_provider", "openai")

    is_gemini_model = (chat_model.startswith("gemini") or "gemini" in chat_model) and "openrouter" not in chat_model and not openai_base_url
    is_anthropic_model = chat_model.startswith("claude") and "openrouter" not in chat_model and not openai_base_url
    is_openrouter = "openrouter" in chat_model or (openai_base_url != "" and "deepseek" not in openai_base_url and "x.ai" not in openai_base_url)
    is_grok_model = chat_model.startswith("grok") or active_provider == "xai"
    is_deepseek_model = "deepseek" in chat_model.lower() or active_provider == "deepseek" or "deepseek" in (openai_base_url or "").lower()

    if is_grok_model:
        openai_key = xai_key
        openai_base_url = "https://api.x.ai/v1"
    elif is_deepseek_model:
        openai_key = deepseek_key or openai_key
        openai_base_url = deepseek_base_url or "https://api.deepseek.com"

    all_clips = []
    
    full_transcript = "".join(
        f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text'].strip()}\n"
        for seg in segments
    )

    word_count = full_transcript.count(' ') + 1 if full_transcript.strip() else 0
    estimated_tokens = int(word_count * 1.3)
    if logger:
        logger(f"📊 Extracted approx {estimated_tokens:,} tokens ({word_count:,} words) for the AI model's context window.")

    if is_gemini_model and not is_openrouter:
        if not HAS_GEMINI:
            if logger: logger("❌ Error: google-genai module missing.")
            return []
            
        if logger: logger(f"🌌 Routing to native Gemini Engine ({chat_model}) with {len(segments)} segments...")

        client = genai.Client(api_key=google_key)
        config_kwargs: Dict[str, Any] = {}
        if genai_types:
            config_kwargs["config"] = genai_types.GenerateContentConfig(
                system_instruction=prompt_text,
                response_mime_type="application/json"
            )
        user_prompt = f"Analyze this entire gaming transcript from start to finish. Scan all timestamps chronologically and extract EVERY memorable highlight (Score >= 7: funny banter, roasts, screams, clutches, team fails, laughing fits) across the entire video. Return strictly valid JSON with a 'clips' array.\n\n{full_transcript}"
        for attempt in range(1, 4):
            try:
                response = client.models.generate_content(
                    model=chat_model,
                    contents=user_prompt,
                    **config_kwargs
                )
                raw_text = getattr(response, "text", "")
                if not raw_text or not raw_text.strip():
                    if logger: logger(f"⚠️ Gemini returned empty response (Attempt {attempt}/3). Retrying...")
                    time.sleep(1.5 * attempt)
                    continue

                found_clips = _parse_json_clips(raw_text)
                if found_clips is not None:
                    if found_clips:
                        if logger: logger(f"🎯 Gemini found {len(found_clips)} clip(s) in the VOD!")
                        all_clips.extend(found_clips)
                    else:
                        if logger: logger("🤷‍♂️ Gemini finished but didn't find any clips.")
                    break
                else:
                    if logger: logger(f"⚠️ Gemini response was not valid JSON (Attempt {attempt}/3). Retrying...")
                    time.sleep(1.5 * attempt)
            except Exception as e:
                if attempt == 3:
                    if logger: logger(f"❌ Gemini API Error: {e}")
                else:
                    if logger: logger(f"⚠️ Gemini API Error (Attempt {attempt}/3): {e}. Retrying...")
                    time.sleep(1.5 * attempt)

    elif is_anthropic_model:
        if not HAS_ANTHROPIC:
            if logger: logger("❌ Error: anthropic python module missing.")
            return []
            
        if logger: logger(f"🌌 Routing to native Anthropic Engine ({chat_model}) with {len(segments)} segments...")

        client = anthropic.Anthropic(api_key=anthropic_key)
        
        user_prompt = f"Analyze this entire gaming transcript from start to finish. Scan all timestamps chronologically and extract EVERY memorable highlight (Score >= 7: funny banter, roasts, screams, clutches, team fails, laughing fits) across the entire video. Return strictly valid JSON with a 'clips' array.\n\n{full_transcript}"
        for attempt in range(1, 4):
            try:
                response = client.messages.create(
                    model=chat_model,
                    max_tokens=4000,
                    system=prompt_text,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ]
                )
                
                raw_content = ""
                if response and response.content and len(response.content) > 0:
                    raw_content = response.content[0].text
                
                if not raw_content or not raw_content.strip():
                    if logger: logger(f"⚠️ Claude returned empty response (Attempt {attempt}/3). Retrying...")
                    time.sleep(1.5 * attempt)
                    continue

                found_clips = _parse_json_clips(raw_content)
                if found_clips is not None:
                    if found_clips:
                        if logger: logger(f"🎯 Claude found {len(found_clips)} clip(s) in the VOD!")
                        all_clips.extend(found_clips)
                    else:
                        if logger: logger("🤷‍♂️ Claude finished but didn't find any clips.")
                    break
                else:
                    if logger: logger(f"⚠️ Claude response was not valid JSON (Attempt {attempt}/3). Retrying...")
                    time.sleep(1.5 * attempt)
            except Exception as e:
                if attempt == 3:
                    if logger: logger(f"❌ Anthropic API Error: {e}")
                else:
                    if logger: logger(f"⚠️ Anthropic API Error (Attempt {attempt}/3): {e}. Retrying...")
                    time.sleep(1.5 * attempt)

    else:
        if logger: 
            if is_openrouter: logger(f"🤖 Routing via Custom Base URL ({chat_model}) with {len(segments)} segments...")
            elif is_deepseek_model: logger(f"🤖 Routing to DeepSeek Engine ({chat_model}) with {len(segments)} segments...")
            else: logger(f"🤖 Routing to OpenAI Engine ({chat_model}) with {len(segments)} segments...")
            
        client_args = {"api_key": openai_key}
        if openai_base_url:
            client_args["base_url"] = openai_base_url
            
        client = OpenAI(**client_args)
        user_prompt = f"Analyze this entire gaming transcript from start to finish. Scan all timestamps chronologically and extract EVERY memorable highlight (Score >= 7: funny banter, roasts, screams, clutches, team fails, laughing fits) across the entire video. Return strictly valid JSON with a 'clips' array.\n\n{full_transcript}"

        create_kwargs: Dict[str, Any] = {
            "model": chat_model,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": prompt_text},
                {"role": "user", "content": user_prompt}
            ]
        }

        # DeepSeek V4 has thinking mode enabled by default; explicitly disable it for structured JSON clip generation
        if is_deepseek_model:
            create_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

        for attempt in range(1, 4):
            try:
                response = client.chat.completions.create(**create_kwargs)
                content = None
                if response and response.choices and len(response.choices) > 0 and response.choices[0].message:
                    content = response.choices[0].message.content
                
                if not content or not content.strip():
                    if logger: logger(f"⚠️ AI returned empty content (Attempt {attempt}/3). Retrying...")
                    time.sleep(1.5 * attempt)
                    continue

                found_clips = _parse_json_clips(content)
                if found_clips is not None:
                    if found_clips:
                        if logger: logger(f"🎯 Found {len(found_clips)} clip(s)!")
                        all_clips.extend(found_clips)
                    else:
                        if logger: logger("🤷‍♂️ No clips found.")
                    break
                else:
                    if logger: logger(f"⚠️ AI response was not valid JSON (Attempt {attempt}/3). Retrying...")
                    time.sleep(1.5 * attempt)
            except Exception as e:
                # If extra_body was rejected by an incompatible proxy
                if "extra_body" in create_kwargs and "thinking" in str(e).lower():
                    del create_kwargs["extra_body"]
                if attempt == 3:
                    if logger: logger(f"❌ OpenAI/Custom API Error: {e}")
                else:
                    if logger: logger(f"⚠️ OpenAI/Custom API Error (Attempt {attempt}/3): {e}. Retrying...")
                    time.sleep(1.5 * attempt)

    return all_clips

def process_video(file_path: str, prompt_profile: str = "Default", logger: Optional[Callable[[str], None]] = None, is_cancelled: Optional[Callable[[], bool]] = None) -> bool:
    """Main orchestration function for analyzing and cutting clips."""
    if not os.path.exists(file_path):
        if logger: logger(f"❌ Error: Source video file not found: {file_path}")
        return False

    config = config_manager.load_config()
    active_prov = config.get("active_ai_provider", "openai")
    if active_prov == "deepseek":
        chat_model = config.get("deepseek_model", "deepseek-v4-flash")
    elif active_prov == "anthropic":
        chat_model = config.get("anthropic_model", "claude-3-5-sonnet-latest")
    elif active_prov == "google":
        chat_model = config.get("google_model", "gemini-2.5-flash")
    elif active_prov == "xai":
        chat_model = config.get("xai_model", "grok-2-latest")
    else:
        chat_model = config.get("openai_model", "gpt-4o")

    if not chat_model:
        chat_model = config.get("openai", {}).get("chat_model", "gpt-4o")

    clips_dir = config.get("settings", {}).get("clips_dir", "")

    if not clips_dir:
        if logger: logger("❌ Error: Generated Clips folder not set in Settings.")
        return False

    if not _validate_api_keys(config, chat_model, logger):
        return False

    segments = _transcribe_audio_to_segments(file_path, config, logger, is_cancelled)
    if not segments:
        return False

    if is_cancelled and is_cancelled(): return False

    prompt_text = config.get("prompts", {}).get("profiles", {}).get(prompt_profile, "Find the best 15-90s moments. Output JSON.")

    all_clips = _generate_clips_with_llm(segments, config, chat_model, prompt_text, logger)

    if all_clips:
        if logger: logger(f"🎬 Sending {len(all_clips)} total timestamp(s) to FFmpeg...")
        final_clips_data = {"clips": all_clips}
        created = extract_clips(file_path, final_clips_data, clips_dir, logger, is_cancelled)
        if logger: logger(f"✨ Successfully exported {len(created)} file(s) to your folder!")
        return True
    else:
        if logger: logger("🤷‍♂️ AI finished scanning the VOD but didn't extract any clips.")
        return True
