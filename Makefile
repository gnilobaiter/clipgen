.PHONY: run test install lint

ifeq ($(OS),Windows_NT)
VENV_PYTHON := .venv/Scripts/python.exe
VENV_DIR := .venv
else
VENV_PYTHON := .venv/bin/python
VENV_DIR := .venv
endif

# faster-whisper pulls in the CPU 'onnxruntime', which shares its module with onnxruntime-gpu: after installing,
# scripts/ensure_onnxruntime_gpu.py puts the GPU build back when it lost the CUDA provider.
install: $(VENV_PYTHON)
	uv pip install -p $(VENV_DIR) --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu129 -r requirements.txt -r requirements-dev.txt
	$(VENV_PYTHON) scripts/ensure_onnxruntime_gpu.py

$(VENV_PYTHON):
	uv venv --clear $(VENV_DIR)

run: install
	$(VENV_PYTHON) main.py

lint: install
	$(VENV_PYTHON) -m ruff check main.py src tests
	$(VENV_PYTHON) -m vulture main.py src tests --min-confidence 80 --ignore-names event,icon,cls

test: install
	$(VENV_PYTHON) -m pytest -v --cov=src --cov=main --cov-report=term-missing tests
	$(VENV_PYTHON) -m ruff check main.py src tests
	$(VENV_PYTHON) -m vulture main.py src tests --min-confidence 80 --ignore-names event,icon,cls
