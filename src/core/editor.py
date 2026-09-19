import contextlib
import hashlib
import io
import json
import os
import re
import subprocess
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from openai import OpenAI
import whisper

from src.core import audio_events, cuts, highlights
from src.core import config as config_manager
from src.utils.hardware import get_hardware_status, log_hardware_info
from src.utils.paths import get_app_data_path, get_ffmpeg_path, get_ffprobe_path, inject_bin_to_path

# openai-whisper tries Triton GPU kernels for word timestamps (median filter, DTW). Triton does not exist on Windows,
# so whisper falls back to slower but equivalent implementations and warns once per 30 s of audio. Harmless noise.
warnings.filterwarnings("ignore", message="Failed to launch Triton kernels", category=UserWarning)

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

class TranscriptionCancelled(Exception):
    """Raised from inside Whisper's progress printing to abort a running transcription."""


class WhisperProgressStream(io.StringIO):
    def __init__(self, logger: Optional[Callable[[str], None]], is_cancelled: Optional[Callable[[], bool]] = None):
        super().__init__()
        self.logger = logger
        self.is_cancelled = is_cancelled
        self.counter = 0

    def write(self, s: str) -> int:
        # whisper.transcribe() has no cancel hook, but with verbose=True it prints every segment to stdout,
        # which is redirected here: raising from this print is the only way to stop a running transcription.
        if self.is_cancelled and self.is_cancelled():
            raise TranscriptionCancelled()
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

# Peak/mean RMS ratio of a segment that counts as a gunshot/explosion transient. Calibrated on a real 90-min VOD:
# 3.5 flagged 61% of all lines (plain speech plosives), 8.0 flags the sharpest ~7%.
COMBAT_TRANSIENT_RATIO = 8.0


def analyze_audio_peaks(audio_array: np.ndarray, segments: List[Dict[str, Any]], sample_rate: int = 16000, peak_detection: bool = True, combat_detection: bool = True) -> List[Dict[str, Any]]:
    """Annotates each Whisper segment with `loudness` (0-100, relative to the LOCAL background level,
    not the global maximum) and `is_combat` (sharp percussive transients)."""
    _rms, level_db, baseline_db = audio_events.compute_level_profile(audio_array) if peak_detection else (None, [], [])
    enhanced_segments = []

    for seg in segments:
        seg_copy = dict(seg)
        start_idx = max(0, min(int(seg['start'] * sample_rate), len(audio_array) - 1))
        end_idx = max(start_idx + 1, min(int(seg['end'] * sample_rate), len(audio_array)))
        chunk = audio_array[start_idx:end_idx]

        if peak_detection:
            seg_copy['loudness'] = audio_events.segment_loudness(seg['start'], seg['end'], level_db, baseline_db)

        is_combat = False
        if combat_detection and len(chunk) >= 512:
            hop_size = 256
            window_size = 512
            sub_chunks = [chunk[i:i + window_size] for i in range(0, len(chunk) - window_size, hop_size)]
            if sub_chunks:
                sub_rms = np.array([np.sqrt(np.mean(sc ** 2)) for sc in sub_chunks])
                mean_sub = np.mean(sub_rms)
                max_sub = np.max(sub_rms)
                if mean_sub > 0.01 and (max_sub / (mean_sub + 1e-5)) > COMBAT_TRANSIENT_RATIO:
                    is_combat = True
        seg_copy['is_combat'] = is_combat
        enhanced_segments.append(seg_copy)

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

def extract_audio_hidden(video_file: str, start: Optional[float] = None, duration: Optional[float] = None,
                         num_audio: Optional[int] = None) -> np.ndarray:
    """Extracts 16kHz mono audio directly into memory using FFmpeg (the whole file, or only
    [start, start + duration] when a range is given).

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

    if num_audio is None:
        num_audio = _count_audio_streams(video_file)

    cmd = [FFMPEG_PATH, "-nostdin", "-threads", "0"]
    if start is not None:
        cmd.extend(["-ss", f"{start:.3f}"])
        if duration is not None:
            cmd.extend(["-t", f"{duration:.3f}"])
    cmd.extend(["-i", video_file])

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

        # Pre/post-roll: clips whose ends were snapped into speech pauses (cuts.snap_clip) carry 0, because padding
        # blindly lands inside a neighbouring word. Clips from other callers keep the old 0.75 s / 1.25 s.
        start = max(0.0, start_raw - float(clip.get("pre_roll", 0.75)))
        end = end_raw + float(clip.get("post_roll", 1.25))
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
    """Sanitizes, validates timestamps, and merges nominations of the SAME moment (real overlap only).
    Back-to-back nominations stay separate: gluing them used to put two different topics into one clip."""
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
            peak_raw = c.get("peak_time")
            peak = round(float(peak_raw), 2) if peak_raw is not None else None
        except (ValueError, TypeError):
            continue

        if start < 0 or end <= start:
            continue
        if (end - start) < 2.0:
            continue

        clip = {
            "start_time": round(start, 2),
            "end_time": round(end, 2),
            "virality_score": score,
            "reasoning": reasoning
        }
        if peak is not None:
            clip["peak_time"] = peak
        valid_clips.append(clip)

    valid_clips.sort(key=lambda x: x["start_time"])

    merged: List[Dict[str, Any]] = []
    for clip in valid_clips:
        if not merged:
            merged.append(clip)
            continue

        prev = merged[-1]
        if clip["start_time"] < prev["end_time"]:
            combined_duration = max(prev["end_time"], clip["end_time"]) - prev["start_time"]
            if combined_duration <= MAX_MERGED_CLIP_SECONDS:
                prev["end_time"] = max(prev["end_time"], clip["end_time"])
                if clip["virality_score"] > prev["virality_score"] and "peak_time" in clip:
                    prev["peak_time"] = clip["peak_time"]
                prev["virality_score"] = max(prev["virality_score"], clip["virality_score"])
                if clip["reasoning"] and clip["reasoning"] not in prev["reasoning"]:
                    prev["reasoning"] = f"{prev['reasoning']} | {clip['reasoning']}"
                continue

        merged.append(clip)

    return merged

def _load_json_payload(raw_text: Any) -> Optional[Any]:
    """Decodes a JSON object/list from a raw LLM answer (tolerates code fences and surrounding prose)."""
    if not raw_text or not isinstance(raw_text, str):
        return None
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).partition("```")[0].strip()
    elif "```" in text:
        match = re.search(r"```(?:json)?\s*([\[{][\s\S]*?[\]}])\s*```", text)
        if match:
            text = match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"(\{[\s\S]*\})", text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                return None
    return None

def _parse_json_clips(raw_text: Any) -> Optional[List[Dict[str, Any]]]:
    """Robustly parses a JSON string containing clips."""
    data = _load_json_payload(raw_text)
    if isinstance(data, dict) and "clips" in data:
        return _clean_and_merge_clips(data.get("clips", []))
    if isinstance(data, list):
        return _clean_and_merge_clips(data)
    return None

# Bump when the cached segment layout or detector calibration changes (v2: word timestamps, local-baseline loudness, audio events; v3: calibrated event thresholds; v4: YAMNet laughter/scream classifier replaces the modulation heuristic; v5: calibrated combat threshold)
TRANSCRIPT_CACHE_VERSION = 5


def _get_transcription_cache_path(file_path: str, whisper_model: str, target_language: Optional[str], audio_peak: bool, combat: bool, legacy: bool = False) -> str:
    """Computes a deterministic hash and cache file path for audio transcriptions.
    `legacy=True` gives the path of the pre-v2 cache entry (no version prefix in the key)."""
    try:
        mtime = os.path.getmtime(file_path)
        size = os.path.getsize(file_path)
    except Exception:
        mtime, size = 0, 0
    
    version_prefix = "" if legacy else f"v{TRANSCRIPT_CACHE_VERSION}_"
    cache_key = f"{version_prefix}{os.path.abspath(file_path)}_{mtime}_{size}_{whisper_model}_{target_language}_{audio_peak}_{combat}"
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

def _slim_segments(raw_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keeps only what the pipeline needs (Whisper's token arrays would bloat the disk cache)."""
    slim = []
    for seg in raw_segments:
        entry: Dict[str, Any] = {"start": float(seg["start"]), "end": float(seg["end"]), "text": str(seg.get("text", ""))}
        words = [{"word": w.get("word", ""), "start": float(w["start"]), "end": float(w["end"])}
                 for w in seg.get("words", []) or [] if "start" in w and "end" in w]
        if words:
            entry["words"] = words
        slim.append(entry)
    return slim


_LEGACY_TAGS = re.compile(r"^\s*((?:\[LOUDNESS: \d+%\]|\[ACTION: COMBAT\])\s*)+")


def _migrate_legacy_segments(legacy: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Turns a pre-v2 cache entry (audio tags baked into `text`, global-max loudness) back into clean Whisper
    segments so the expensive transcription is not repeated. Loudness/events are recomputed from the audio."""
    clean = []
    for seg in legacy:
        if not isinstance(seg, dict) or "start" not in seg or "end" not in seg:
            continue
        text = str(seg.get("text", ""))
        match = _LEGACY_TAGS.match(text)
        if match:
            text = text[match.end():]
        clean.append({"start": seg["start"], "end": seg["end"], "text": text})
    return _slim_segments(clean)


def _annotate_with_audio(audio_array: np.ndarray, raw_segments: List[Dict[str, Any]], audio_peak_detection: bool,
                         combat_detection: bool, logger: Optional[Callable[[str], None]]) -> List[Dict[str, Any]]:
    """Adds local-baseline loudness / combat flags to segments and merges detected audio events into the timeline."""
    segments = raw_segments
    if audio_peak_detection or combat_detection:
        segments = analyze_audio_peaks(audio_array, raw_segments, peak_detection=audio_peak_detection, combat_detection=combat_detection)
    if audio_peak_detection and raw_segments:
        events = _detect_events_safe(audio_array, logger)
        segments = sorted(segments + events, key=lambda e: (e["start"], 0 if e.get("event") else 1))
    return segments


def _format_event_counts(segments: List[Dict[str, Any]]) -> str:
    counts: Dict[str, int] = {}
    for seg in segments:
        if seg.get("event"):
            counts[seg["event"]] = counts.get(seg["event"], 0) + 1
    return ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())) or "none"


def _detect_events_safe(audio_array: np.ndarray, logger: Optional[Callable[[str], None]]) -> List[Dict[str, Any]]:
    """Laughter / scream / loud-burst detection as timeline entries; never fails the whole transcription."""
    try:
        detected = audio_events.detect_audio_events(audio_array, logger=logger)
    except Exception as e:
        if logger: logger(f"⚠️ Audio event detection skipped: {e}")
        return []
    if logger:
        counts: Dict[str, int] = {}
        for ev in detected:
            counts[ev["type"]] = counts.get(ev["type"], 0) + 1
        summary = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())) or "none"
        logger(f"🔊 Audio events detected: {summary}")
    return [{"start": ev["start"], "end": ev["end"], "text": "", "event": ev["type"], "strength": ev["strength"]} for ev in detected]


FALLBACK_WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3", "turbo"]


def get_whisper_models(current: Optional[str] = None) -> List[str]:
    """Every model the installed openai-whisper can load (tiny.en ... large-v3-turbo), so the UI never drifts
    from the library. A saved custom value (e.g. a local .pt path) is kept selectable."""
    try:
        models = list(whisper.available_models())
    except Exception:
        models = list(FALLBACK_WHISPER_MODELS)
    if current and current not in models:
        models.append(current)
    return models


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

def _migrate_legacy_cache(file_path: str, cache_file: str, whisper_model: str, target_language: Optional[str], audio_peak: bool, combat: bool,
                          logger: Optional[Callable[[str], None]], is_cancelled: Optional[Callable[[], bool]]) -> Optional[List[Dict[str, Any]]]:
    """Upgrades a pre-v2 transcript cache entry (audio extraction only, no Whisper). None = nothing to migrate / fall back to Whisper; [] = cancelled."""
    legacy_file = _get_transcription_cache_path(file_path, whisper_model, target_language, audio_peak, combat, legacy=True)
    legacy = _load_cached_segments(legacy_file, None)
    if legacy is None:
        return None
    raw_segments = _migrate_legacy_segments(legacy)
    if not raw_segments:
        return None
    if logger: logger(f"♻️ Found an older cached transcript ({len(raw_segments)} segments) - reusing the Whisper text, re-analysing the audio only...")
    if is_cancelled and is_cancelled(): return []
    try:
        audio_array = extract_audio_hidden(file_path)
        segments = _annotate_with_audio(audio_array, raw_segments, audio_peak, combat, logger)
    except Exception as e:
        if logger: logger(f"⚠️ Could not upgrade the older cache ({e}); transcribing from scratch.")
        return None
    _save_cached_segments(cache_file, segments, logger)
    return segments


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
        if logger: logger(f"🔊 Audio events stored in the cache: {_format_event_counts(cached_segments)} (sound classifier not re-run).")
        return cached_segments

    migrated = _migrate_legacy_cache(file_path, cache_file, whisper_model_type, target_language, audio_peak_detection, combat_detection, logger, is_cancelled)
    if migrated is not None:
        return migrated

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
        progress_stream = WhisperProgressStream(logger, is_cancelled)
        with contextlib.redirect_stdout(progress_stream):
            transcribe_kwargs: Dict[str, Any] = {
                "condition_on_previous_text": False,
                "beam_size": 5 if device == "cuda" else 2,
                "best_of": 5 if device == "cuda" else 2,
                "temperature": (0.0, 0.2, 0.4),
                "fp16": fp16_enabled,
                "verbose": True,
                "word_timestamps": True,
                "hallucination_silence_threshold": 2.0,
                "initial_prompt": "Gameplay livestream recording with conversational banter, laughter, discord chat, and gaming terms."
            }
            if target_language is not None:
                transcribe_kwargs["language"] = target_language

            result = model.transcribe(audio_array, **transcribe_kwargs)
            
        raw_segments = _slim_segments(result.get("segments", []))

        segments = _annotate_with_audio(audio_array, raw_segments, audio_peak_detection, combat_detection, logger)
        if logger:
            logger("✅ Transcription complete! Audio patterns analyzed." if (audio_peak_detection or combat_detection) else "✅ Transcription complete!")

        if segments:
            _save_cached_segments(cache_file, segments, logger)
        return segments

    except TranscriptionCancelled:
        if logger: logger("🛑 Transcription cancelled.")
        return None
    except Exception as e:
        if logger: logger(f"❌ Transcription error: {e}")
        return None

MAX_MERGED_CLIP_SECONDS = 150.0  # adjacent nominations of one moment are merged up to this length
ANTHROPIC_MAX_OUTPUT_TOKENS = 8192  # per window (a handful of clips); stays under the SDK's non-streaming limit
LLM_ATTEMPTS = 3
MAX_CONSECUTIVE_WINDOW_FAILURES = 2

WINDOW_PROMPT = (
    "This is part {index} of {total} of a {minutes:.0f}-minute gaming stream transcript, covering {w_start:.0f}s to {w_end:.0f}s. "
    "All timestamps are absolute seconds from the start of the video.\n"
    "{overlap_note}"
    "Line format: [start - end] optional audio tags, then speech. [LOUDNESS: X%] = how far the line rises above the local background level "
    "(shown only when notable); [ACTION: COMBAT] = sharp gunshot/explosion transients; standalone lines such as [LAUGHTER 80%], [SCREAM 90%] or [LOUD 60%] "
    "were detected in the raw audio (LAUGHTER/SCREAM by a pretrained sound classifier, the percentage is its confidence; LOUD by a loudness detector). A laugh is usually the payoff of the line(s) right BEFORE it, so include both.\n\n"
    "Return {want} candidate highlights: the {want} best distinct moments of this section, best first. Your job here is to RANK, not to gatekeep - "
    "a later step compares the candidates of all sections on one scale and drops the weak ones, but it can only choose from what you nominate. "
    "If the section has fewer real highlights, fill the remaining slots with the best of the rest and give them the low score they deserve; "
    "never merge separate moments and never inflate a score to fill a slot. "
    "Rate each one on an ABSOLUTE 1-10 scale relative to the whole stream. "
    "Clip length must come from the content, never from a target: many of the best clips are short, and a long clip needs to be great all the way through. "
    "Return strictly valid JSON with a 'clips' array.\n\n"
    "{transcript}"
)

RERANK_PROMPT = (
    "Below are {count} candidate highlights nominated independently from different sections of the same {minutes:.0f}-minute gaming stream. "
    "Because each section was scored in isolation, the scores are not comparable. Re-score every candidate on ONE absolute 1-10 scale "
    "(10 = the best moment of the whole stream, 5 = skippable); be decisive and spread the scores out. "
    "Return strictly valid JSON: {{\"scores\": [{{\"id\": <int>, \"score\": <int 1-10>}}, ...]}} covering every id.\n\n{items}"
)


RERANK_SYSTEM = (
    "You are a meticulous editor calibrating highlight scores for a gaming stream. "
    "You never nominate new clips and never change timestamps; you only re-score the numbered candidates you are given. "
    "Answer with strictly valid JSON exactly in the format requested by the user, and nothing else."
)


class _LLMRoute:
    """Which SDK / endpoint a chat model is routed to (resolved once per video)."""

    def __init__(self, config: Dict[str, Any], chat_model: str):
        openai_cfg = config.get("openai", {})
        self.chat_model = chat_model
        self.openai_key = openai_cfg.get("api_key", "")
        self.openai_base_url = openai_cfg.get("base_url", "") or ""
        self.google_key = config.get("google", {}).get("api_key", "")
        self.anthropic_key = config.get("anthropic", {}).get("api_key", "")
        active_provider = config.get("active_ai_provider", "openai")

        has_custom_url = bool(self.openai_base_url)
        self.is_openrouter = "openrouter" in chat_model or (has_custom_url and "deepseek" not in self.openai_base_url and "x.ai" not in self.openai_base_url)
        self.is_gemini = "gemini" in chat_model and "openrouter" not in chat_model and not has_custom_url
        self.is_anthropic = chat_model.startswith("claude") and "openrouter" not in chat_model and not has_custom_url
        self.is_grok = chat_model.startswith("grok") or active_provider == "xai"
        self.is_deepseek = "deepseek" in chat_model.lower() or active_provider == "deepseek" or "deepseek" in self.openai_base_url.lower()

        # DeepSeek V4 has an optional reasoning ("thinking") mode. It is slower but decides noticeably better where a moment
        # begins and ends (measured on the boundary review), so it is on by default and can be turned off in Settings.
        self.thinking = self.is_deepseek and bool(config.get("settings", {}).get("deepseek_thinking", True))

        if self.is_grok:
            self.openai_key = config.get("xai", {}).get("api_key", "")
            self.openai_base_url = "https://api.x.ai/v1"
        elif self.is_deepseek:
            self.openai_key = config.get("deepseek", {}).get("api_key", "") or self.openai_key
            self.openai_base_url = config.get("deepseek", {}).get("base_url", "") or "https://api.deepseek.com"

    @property
    def engine_name(self) -> str:
        if self.is_gemini and not self.is_openrouter: return "Gemini"
        if self.is_anthropic: return "Claude"
        if self.is_openrouter: return "Custom Base URL"
        if self.is_deepseek: return "DeepSeek"
        if self.is_grok: return "Grok"
        return "OpenAI"


def _complete_text(route: _LLMRoute, system_prompt: str, user_prompt: str, extra_body_enabled: List[bool]) -> Optional[str]:
    """One raw chat completion. Raises on API errors; returns None for an empty answer."""
    if route.is_gemini and not route.is_openrouter:
        if not HAS_GEMINI:
            raise RuntimeError("google-genai module missing.")
        client = genai.Client(api_key=route.google_key)
        kwargs: Dict[str, Any] = {}
        if genai_types:
            kwargs["config"] = genai_types.GenerateContentConfig(system_instruction=system_prompt, response_mime_type="application/json")
        response = client.models.generate_content(model=route.chat_model, contents=user_prompt, **kwargs)
        return getattr(response, "text", "") or None

    if route.is_anthropic:
        if not HAS_ANTHROPIC:
            raise RuntimeError("anthropic python module missing.")
        client = anthropic.Anthropic(api_key=route.anthropic_key)
        response = client.messages.create(
            model=route.chat_model,
            max_tokens=ANTHROPIC_MAX_OUTPUT_TOKENS,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        if response and response.content:
            return "".join(getattr(block, "text", "") for block in response.content) or None
        return None

    client_args = {"api_key": route.openai_key}
    if route.openai_base_url:
        client_args["base_url"] = route.openai_base_url
    client = OpenAI(**client_args)
    create_kwargs: Dict[str, Any] = {
        "model": route.chat_model,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
    }
    # DeepSeek V4: switch the reasoning mode explicitly (the API default is "enabled"), the same way for every request
    if route.is_deepseek and extra_body_enabled[0]:
        create_kwargs["extra_body"] = {"thinking": {"type": "enabled" if route.thinking else "disabled"}}
    try:
        response = client.chat.completions.create(**create_kwargs)
    except Exception as e:
        # extra_body rejected by an incompatible proxy: drop it for all following requests
        if "extra_body" in create_kwargs and "thinking" in str(e).lower():
            extra_body_enabled[0] = False
        raise
    if response and response.choices and response.choices[0].message:
        return response.choices[0].message.content or None
    return None


def _request_json(route: _LLMRoute, system_prompt: str, user_prompt: str, parser: Callable[[Any], Optional[Any]],
                  logger: Optional[Callable[[str], None]], label: str, extra_body_enabled: List[bool]) -> Optional[Any]:
    """Runs a completion with retries until `parser` accepts the answer. Returns the parsed value or None."""
    for attempt in range(1, LLM_ATTEMPTS + 1):
        try:
            raw = _complete_text(route, system_prompt, user_prompt, extra_body_enabled)
            if not raw or not raw.strip():
                if logger: logger(f"⚠️ {route.engine_name} returned an empty response for {label} (Attempt {attempt}/{LLM_ATTEMPTS}). Retrying...")
            else:
                parsed = parser(raw)
                if parsed is not None:
                    return parsed
                if logger: logger(f"⚠️ {route.engine_name} response for {label} was not valid JSON (Attempt {attempt}/{LLM_ATTEMPTS}). Retrying...")
        except Exception as e:
            if attempt == LLM_ATTEMPTS:
                if logger: logger(f"❌ {route.engine_name} API Error ({label}): {e}")
                return None
            if logger: logger(f"⚠️ {route.engine_name} API Error ({label}, Attempt {attempt}/{LLM_ATTEMPTS}): {e}. Retrying...")
        if attempt < LLM_ATTEMPTS:
            time.sleep(1.5 * attempt)
    return None


def _parse_json_scores(raw_text: Any) -> Optional[Dict[int, int]]:
    data = _load_json_payload(raw_text)
    items = data.get("scores") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return None
    scores: Dict[int, int] = {}
    for item in items:
        try:
            scores[int(item["id"])] = max(1, min(10, int(item["score"])))
        except (KeyError, TypeError, ValueError):
            continue
    return scores or None


def _excerpt(segments: List[Dict[str, Any]], start: float, end: float, limit: int = 320) -> str:
    text = " ".join(str(s.get("text", "")).strip() for s in segments if not s.get("event") and s["end"] > start and s["start"] < end)
    return text if len(text) <= limit else text[:limit] + "…"


def _rerank_candidates(route: _LLMRoute, candidates: List[Dict[str, Any]], segments: List[Dict[str, Any]], duration: float,
                       logger: Optional[Callable[[str], None]], extra_body_enabled: List[bool]) -> List[Dict[str, Any]]:
    """Pass 2: one short request that puts scores from independently-scored windows on a common scale."""
    items = "\n".join(
        f"id={i} [{c['start_time']:.0f}s-{c['end_time']:.0f}s] section_score={c['virality_score']} why: {c['reasoning']} | transcript: {_excerpt(segments, c['start_time'], c['end_time'])}"
        for i, c in enumerate(candidates)
    )
    prompt = RERANK_PROMPT.format(count=len(candidates), minutes=duration / 60.0, items=items)
    scores = _request_json(route, RERANK_SYSTEM, prompt, _parse_json_scores, logger, "re-ranking", extra_body_enabled)
    if not scores:
        if logger: logger("⚠️ Re-ranking failed - keeping per-section scores.")
        return candidates
    reranked = []
    for i, cand in enumerate(candidates):
        updated = dict(cand)
        updated["virality_score"] = scores.get(i, cand["virality_score"])
        reranked.append(updated)
    return reranked


def _generate_clips_with_llm(segments: List[Dict[str, Any]], config: Dict[str, Any], chat_model: str, prompt_text: str, logger: Optional[Callable[[str], None]], is_cancelled: Optional[Callable[[], bool]] = None) -> List[Dict[str, Any]]:
    """Pass 1: nominate candidates per overlapping transcript window. Pass 2: dedupe + global re-score.
    Returns candidates only - density selection and boundary refinement happen in process_video."""
    route = _LLMRoute(config, chat_model)
    extra_body_enabled = [True]
    clips_per_hour = float(config.get("settings", {}).get("clips_per_hour", highlights.DEFAULT_CLIPS_PER_HOUR))

    entries = highlights.build_utterances(segments)  # lines that follow real pauses, not Whisper's arbitrary chunking
    windows = highlights.build_windows(entries)
    if not windows:
        return []
    duration = max(w["end"] for w in windows)

    word_count = sum(len(str(s.get("text", "")).split()) for s in segments)
    if logger:
        logger(f"📊 Extracted approx {int(word_count * 1.3):,} tokens ({word_count:,} words) in {len(windows)} window(s) for the AI model.")
        logger(f"🤖 Routing to {route.engine_name} Engine ({chat_model}) with {len(segments)} segments...")
        logger(f"🏷️ Audio event tags included in the AI prompts: {_format_event_counts(segments)}.")

    workers = LLM_WORKERS if route.thinking else 1
    if logger and route.is_deepseek:
        logger(f"🧠 DeepSeek reasoning mode: {'ON (slower, better decisions)' if route.thinking else 'OFF (fast)'}"
               + (f", {workers} requests in parallel." if workers > 1 else "."))

    def nominate(item: Any) -> Optional[List[Dict[str, Any]]]:
        idx, window = item
        transcript = "".join(highlights.format_entry(e) for e in window["lines"])
        overlap_note = ""
        if len(windows) > 1:
            overlap_note = (f"Only nominate moments whose peak lies inside this section; the first and last {highlights.WINDOW_OVERLAP:.0f}s overlap the "
                            "neighbouring sections and are given for context. Do not cut a moment short just because the section ends - use its real end.\n")
        user_prompt = WINDOW_PROMPT.format(
            index=idx, total=len(windows), minutes=duration / 60.0, w_start=window["start"], w_end=window["end"],
            overlap_note=overlap_note, want=highlights.requested_candidates(window["end"] - window["start"], clips_per_hour),
            transcript=transcript,
        )
        return _request_json(route, prompt_text, user_prompt, _parse_json_clips, logger, f"window {idx}/{len(windows)}", extra_body_enabled)

    candidates: List[Dict[str, Any]] = []
    consecutive_failures = 0
    for (idx, _window), found in _run_batches(list(enumerate(windows, start=1)), nominate, workers, is_cancelled):
        consecutive_failures = consecutive_failures + 1 if found is None else 0
        if found:
            if logger: logger(f"🎯 Window {idx}/{len(windows)}: {len(found)} candidate(s).")
            candidates.extend(found)
        elif found is not None and logger:
            logger(f"🤷‍♂️ Window {idx}/{len(windows)}: no candidates.")
        if consecutive_failures >= MAX_CONSECUTIVE_WINDOW_FAILURES:
            if logger: logger(f"❌ {consecutive_failures} windows failed in a row - stopping (check your API key, model name and connection).")
            break
    if is_cancelled and is_cancelled():
        return []

    candidates = highlights.dedupe_candidates(candidates)
    if len(windows) > 1 and len(candidates) > 3 and not (is_cancelled and is_cancelled()):
        if logger: logger(f"⚖️ Re-ranking {len(candidates)} candidates on a common scale...")
        candidates = _rerank_candidates(route, candidates, segments, duration, logger, extra_body_enabled)
    return candidates


LLM_WORKERS = 4  # parallel requests when the reasoning mode makes each one slow


def _run_batches(items: List[Any], worker: Callable[[Any], Any], workers: int, is_cancelled: Optional[Callable[[], bool]]):
    """Yields (item, result) in order, running `worker` on `workers` items at a time in threads. Stops quietly (before
    starting the next batch) when cancelled; the consumer may also stop early by breaking out of the loop."""
    for i in range(0, len(items), max(1, workers)):
        if is_cancelled and is_cancelled():
            return
        batch = items[i:i + max(1, workers)]
        if len(batch) == 1:
            results = [worker(batch[0])]
        else:
            with ThreadPoolExecutor(max_workers=len(batch)) as pool:
                results = list(pool.map(worker, batch))
        yield from zip(batch, results)


REVIEW_CONTEXT = 45.0
MAX_REVIEW_FAILURES = 2

REVIEW_SYSTEM = (
    "You are a precise video editor. You split a stream excerpt into scenes and pick the one self-contained clip range in it. "
    "Answer with strictly valid JSON exactly in the requested format and nothing else."
)

REVIEW_PROMPT = (
    "Below is a transcript excerpt of a gaming stream ({lo:.0f}s - {hi:.0f}s). A highlight clip was proposed from {start:.1f}s to {end:.1f}s "
    "(lines inside it are marked with '>'). All times are absolute seconds. Line format: [start - end] tags and speech; standalone "
    "[LAUGHTER n%] / [SCREAM n%] / [LOUD n%] lines come from an audio classifier.\n\n"
    "STEP 1 - split the WHOLE excerpt into consecutive scenes. A new scene starts whenever the subject of the conversation changes "
    "(a different joke, a different puzzle or game object, a different question, moving from banter to giving instructions, and so on). "
    "For every scene give start_time and end_time (only values printed on the lines), a short topic, and 'humor' 0-10 = how funny or entertaining it is "
    "for someone who has never seen the stream (10 = laugh out loud, 3 = people merely explaining or coordinating).\n"
    "STEP 2 - choose the clip: a range of CONSECUTIVE scenes (clip_first .. clip_last) that forms ONE self-contained moment for someone who has never "
    "seen the stream. It is the scene with the payoff, plus the directly preceding scene(s) the payoff needs in order to be understood (the setup, the "
    "question, the running gag), plus the immediate reaction ONLY when it is about the same subject. A scene that starts a new subject (noticing a new device, "
    "panel or task, moving on to the next step) is never part of the clip, even when it follows the payoff directly. Never include filler, or scenes that "
    "merely coordinate the game. A punchline without its setup is not a clip: the range must let a stranger understand who is talking to whom and what "
    "event, question or line is being reacted to (an insult, a scream or a 'wow' needs the thing that caused it, usually the 10-30 seconds before it). "
    "Only after that is satisfied, keep the range as tight as possible. A clip longer than 120 seconds must be entertaining throughout: if it contains "
    "slow or purely coordinating stretches, keep only the best continuous stretch.{short_note}\n\n"
    "Return strictly valid JSON: {{\"scenes\": [{{\"start_time\": <float>, \"end_time\": <float>, \"topic\": \"...\", \"humor\": <int>}}, ...], "
    "\"clip_first\": <int>, \"clip_last\": <int>, \"reason\": \"<one short sentence>\"}}\n\n{excerpt}"
)


def _parse_review(raw_text: Any) -> Optional[Dict[str, Any]]:
    """Turns the scene split + chosen scene range into {start_time, end_time, reason, scenes}."""
    data = _load_json_payload(raw_text)
    if not isinstance(data, dict) or not isinstance(data.get("scenes"), list):
        return None
    try:
        scenes = [{"start_time": float(sc["start_time"]), "end_time": float(sc["end_time"]), "topic": str(sc.get("topic", ""))} for sc in data["scenes"]]
        first, last = int(data["clip_first"]), int(data["clip_last"])
    except (KeyError, TypeError, ValueError):
        return None
    if not scenes or not 0 <= first <= last < len(scenes):
        return None
    return {"start_time": scenes[first]["start_time"], "end_time": scenes[last]["end_time"],
            "reason": str(data.get("reason", "")).strip(), "scenes": len(scenes)}


def _accept_review(clip: Dict[str, Any], review: Dict[str, Any], duration: float) -> bool:
    """A reviewed range is only trusted if it is sane and clearly the same moment (the model may legitimately drop
    a scene about another topic, so it may not lose the whole original range)."""
    start, end = review["start_time"], review["end_time"]
    if start < 0 or end > duration + 1.0 or not highlights.MIN_CLIP_SECONDS <= end - start <= highlights.MAX_CLIP_SECONDS:
        return False
    if abs(start - clip["start_time"]) > REVIEW_CONTEXT or abs(end - clip["end_time"]) > REVIEW_CONTEXT:
        return False
    original = clip["end_time"] - clip["start_time"]
    if end - start < min(highlights.MIN_CONTEXT_SECONDS, 0.6 * original):  # shrinking to a bare punchline is how context got lost
        return False
    overlap = min(end, clip["end_time"]) - max(start, clip["start_time"])
    return overlap >= 0.3 * min(end - start, clip["end_time"] - clip["start_time"])


def _review_clip_boundaries(route: _LLMRoute, clips: List[Dict[str, Any]], entries: List[Dict[str, Any]], duration: float,
                            logger: Optional[Callable[[str], None]], is_cancelled: Optional[Callable[[], bool]],
                            extra_body_enabled: List[bool]) -> List[Dict[str, Any]]:
    """Pass 3: every selected clip is looked at on its own with 45 s of surrounding transcript. The model first splits the
    excerpt into scenes (a new one whenever the subject changes) and then picks the range of scenes that forms one
    self-contained moment. Splitting first matters: asked to merely "judge" a clip, a fast model called a clip that
    glued two topics together "one continuous exchange". This fixes clips that lack their setup, run into the next
    topic or drag on; pass 1 looks at 12 minutes at once and is much less precise about where a moment really is."""
    workers = LLM_WORKERS if route.thinking else 1
    if logger:
        mode = f" (reasoning {'ON' if route.thinking else 'OFF'}, {workers} in parallel)" if route.is_deepseek else ""
        logger(f"🔍 Reviewing the boundaries of {len(clips)} clip(s){mode}...")

    def review_one(item: Any) -> Optional[Dict[str, Any]]:
        i, clip = item
        lo, hi = clip["start_time"] - REVIEW_CONTEXT, clip["end_time"] + REVIEW_CONTEXT
        excerpt = "".join(("> " if e["end"] > clip["start_time"] and e["start"] < clip["end_time"] else "  ") + highlights.format_entry(e)
                          for e in entries if e["end"] > lo and e["start"] < hi)
        length = clip["end_time"] - clip["start_time"]
        short_note = (f" NOTE: this clip is only {length:.0f} seconds long. Clips that short are almost always missing the situation that makes "
                      "the punchline understandable: unless it is a complete joke on its own, start EARLIER and include the scene(s) that set it up."
                      if length < 2 * highlights.MIN_CONTEXT_SECONDS else "")
        prompt = REVIEW_PROMPT.format(lo=lo, hi=hi, start=clip["start_time"], end=clip["end_time"], excerpt=excerpt, short_note=short_note)
        return _request_json(route, REVIEW_SYSTEM, prompt, _parse_review, logger, f"boundary review {i}/{len(clips)}", extra_body_enabled)

    reviewed: List[Dict[str, Any]] = []
    failures = changed = 0
    for (i, clip), review in _run_batches(list(enumerate(clips, start=1)), review_one, workers, is_cancelled):
        if review is None:
            failures += 1
        else:
            failures = 0
            if _accept_review(clip, review, duration) and (abs(review["start_time"] - clip["start_time"]) > 0.5 or abs(review["end_time"] - clip["end_time"]) > 0.5):
                changed += 1
                if logger: logger(f"🔍 Clip {i}: {clip['start_time']:.0f}-{clip['end_time']:.0f}s -> {review['start_time']:.0f}-{review['end_time']:.0f}s ({review['reason'][:110]})")
                clip = dict(clip, start_time=round(review["start_time"], 2), end_time=round(review["end_time"], 2))
        reviewed.append(clip)
        if failures >= MAX_REVIEW_FAILURES:
            if logger: logger("⚠️ The boundary review keeps failing - keeping the remaining clips as they are.")
            break
    if is_cancelled and is_cancelled():
        return clips
    reviewed.extend(clips[len(reviewed):])
    if logger: logger(f"🔍 Boundary review: {changed} of {len(clips)} clip(s) adjusted, the rest were already right.")
    return reviewed


def _snap_clips_to_pauses(file_path: str, clips: List[Dict[str, Any]], duration: float,
                          logger: Optional[Callable[[str], None]]) -> List[Dict[str, Any]]:
    """Moves every clip start/end into a nearby pause in the audio so no word is cut in half."""
    if not clips:
        return clips
    try:
        num_audio = _count_audio_streams(file_path)
        snapped = [cuts.snap_clip(c, lambda s, length: extract_audio_hidden(file_path, s, length, num_audio), duration) for c in clips]
    except Exception as e:
        if logger: logger(f"⚠️ Could not snap clip boundaries to speech pauses ({e}); using the AI timestamps as they are.")
        return clips
    done = sum(1 for c in snapped for k in ("pre_roll", "post_roll") if c.get(k) == 0.0)
    if logger: logger(f"✂️ Placed {done} of {2 * len(snapped)} clip boundaries inside speech pauses (no cut words).")
    return snapped


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
        chat_model = config.get("anthropic_model", "claude-sonnet-5")
    elif active_prov == "google":
        chat_model = config.get("google_model", "gemini-3.5-flash")
    elif active_prov == "xai":
        chat_model = config.get("xai_model", "grok-4.3")
    else:
        chat_model = config.get("openai_model", "gpt-5.5")

    if not chat_model:
        chat_model = config.get("openai", {}).get("chat_model", "gpt-5.5")

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

    candidates = _generate_clips_with_llm(segments, config, chat_model, prompt_text, logger, is_cancelled)
    if is_cancelled and is_cancelled(): return False

    settings_cfg = config.get("settings", {})
    clips_per_hour = float(settings_cfg.get("clips_per_hour", highlights.DEFAULT_CLIPS_PER_HOUR))
    duration = max((seg["end"] for seg in segments), default=0.0)
    min_score = int(settings_cfg.get("min_clip_score", highlights.DEFAULT_MIN_SCORE))
    stats: Dict[str, int] = {}
    all_clips = highlights.select_clips(candidates, duration, clips_per_hour=clips_per_hour, min_score=min_score, stats=stats)
    if logger and candidates:
        dropped = [f"{stats[key]} {label}" for key, label in (("low_score", f"below score {min_score}"), ("too_short", "too short"),
                                                              ("too_close", "overlapping a better clip"), ("over_limit", "over the density limit")) if stats.get(key)]
        limit_note = f"up to {stats['limit']} at {clips_per_hour:g}/hour" if stats.get("limit", -1) >= 0 else "no limit"
        logger(f"🎚️ Selected {len(all_clips)} of {len(candidates)} candidate(s) ({limit_note}). Dropped: {', '.join(dropped) or 'none'}.")
    raw_lengths = [c["end_time"] - c["start_time"] for c in all_clips]
    entries = highlights.build_utterances(segments)
    if all_clips and settings_cfg.get("review_boundaries", True):
        all_clips = _review_clip_boundaries(_LLMRoute(config, chat_model), all_clips, entries, duration, logger, is_cancelled, [True])
        if is_cancelled and is_cancelled(): return False
    all_clips = [highlights.extend_short_clip(clip, entries) for clip in all_clips]
    all_clips = [highlights.limit_length(highlights.refine_clip(clip, entries, duration)) for clip in all_clips]
    all_clips = highlights.resolve_overlaps(_snap_clips_to_pauses(file_path, all_clips, duration, logger))
    if is_cancelled and is_cancelled(): return False
    if logger and all_clips:
        final_lengths = sorted(c["end_time"] - c["start_time"] for c in all_clips)
        logger(f"📏 Clip lengths: from the AI median {sorted(raw_lengths)[len(raw_lengths) // 2]:.0f}s -> final median {final_lengths[len(final_lengths) // 2]:.0f}s "
               f"(min {final_lengths[0]:.0f}s, max {final_lengths[-1]:.0f}s).")

    if all_clips:
        if logger: logger(f"🎬 Sending {len(all_clips)} total timestamp(s) to FFmpeg...")
        final_clips_data = {"clips": all_clips}
        created = extract_clips(file_path, final_clips_data, clips_dir, logger, is_cancelled)
        if logger: logger(f"✨ Successfully exported {len(created)} file(s) to your folder!")
        return True
    else:
        if logger: logger("🤷‍♂️ AI finished scanning the VOD but didn't extract any clips.")
        return True
