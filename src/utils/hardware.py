import logging
from typing import Any, Callable, Dict, Optional

logger_obj = logging.getLogger("hardware_detector")

def get_hardware_status() -> Dict[str, Any]:
    """Detects available computing hardware (CUDA GPU vs CPU) and returns hardware metrics."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        device = "cuda" if cuda_available else "cpu"
        device_count = torch.cuda.device_count() if cuda_available else 0
        gpu_name = torch.cuda.get_device_name(0) if cuda_available and device_count > 0 else None
        
        vram_gb = None
        if cuda_available and device_count > 0:
            props = torch.cuda.get_device_properties(0)
            vram_gb = round(props.total_memory / (1024 ** 3), 1)

        return {
            "torch_available": True,
            "cuda_available": cuda_available,
            "device": device,
            "gpu_name": gpu_name,
            "device_count": device_count,
            "vram_gb": vram_gb,
            "fp16_supported": cuda_available
        }
    except Exception as e:
        return {
            "torch_available": False,
            "cuda_available": False,
            "device": "cpu",
            "gpu_name": None,
            "device_count": 0,
            "vram_gb": None,
            "fp16_supported": False,
            "error": str(e)
        }

def log_hardware_info(logger: Optional[Callable[[str], None]] = None) -> str:
    """Logs the current hardware status with descriptive icons and returns the primary status message."""
    status = get_hardware_status()
    
    if status["cuda_available"]:
        gpu_info = f"{status['gpu_name']} ({status['vram_gb']} GB VRAM)" if status.get("vram_gb") else status["gpu_name"]
        primary_msg = f"🧠 Hardware Check: NVIDIA GPU Detected ({gpu_info})"
        speed_msg = "🚀 Processing Mode: GPU (CUDA fp16 acceleration active - maximum speed)"
        
        if logger:
            logger(primary_msg)
            logger(speed_msg)
        else:
            logger_obj.info(primary_msg)
            logger_obj.info(speed_msg)
            
        return f"GPU: {status['gpu_name']} (CUDA fp16 active)"
    else:
        primary_msg = "⚠️ Hardware Check: No NVIDIA CUDA GPU detected."
        mode_msg = "⏳ Processing Mode: CPU (Whisper transcription will run on CPU, processing will take longer)"
        
        if logger:
            logger(primary_msg)
            logger(mode_msg)
        else:
            logger_obj.warning(primary_msg)
            logger_obj.warning(mode_msg)
            
        return "CPU Mode (CUDA not available)"
