"""Puts the GPU build of onnxruntime back on top after `uv pip install`.

faster-whisper depends on the CPU package `onnxruntime`, which installs the same `onnxruntime` module as
`onnxruntime-gpu` and can overwrite it, taking the CUDA provider away from the sound classifier. Run by `make install`
with the project's Python; does nothing when the CUDA provider is already there. (A script instead of a shell one-liner,
because `make` on Windows has no shell for `||` / quoting.)
"""

import subprocess
import sys
import time

PIN = "onnxruntime-gpu==1.30.0"


def has_cuda_provider() -> bool:
    try:
        import onnxruntime
        return "CUDAExecutionProvider" in onnxruntime.get_available_providers()
    except Exception:
        return False


def main() -> int:
    if has_cuda_provider():
        print("onnxruntime: CUDA provider available, nothing to do.")
        return 0
    print(f"onnxruntime lost its CUDA provider: reinstalling {PIN}")
    cmd = ["uv", "pip", "install", "--python", sys.executable, "--reinstall", "--no-deps", PIN]
    code = subprocess.call(cmd)
    if code != 0:  # Windows may lock a DLL for a moment (IDE / antivirus scanning the venv): try once more
        print("reinstall failed (a file in .venv is locked?), retrying once...")
        time.sleep(3)
        code = subprocess.call(cmd)
    if code != 0:
        print("Could not reinstall onnxruntime-gpu: close programs using .venv (IDE, running app) and run `make install` again.")
    return code


if __name__ == "__main__":
    sys.exit(main())
