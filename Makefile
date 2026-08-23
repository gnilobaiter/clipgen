.PHONY: run test install lint

ifeq ($(OS),Windows_NT)
VENV_PYTHON := .venv/Scripts/python.exe
VENV_DIR := .venv
else
VENV_PYTHON := .venv/bin/python
VENV_DIR := .venv
endif

install: $(VENV_PYTHON)
	uv pip install -p $(VENV_DIR) --extra-index-url https://download.pytorch.org/whl/cu126 -r requirements.txt -r requirements-dev.txt

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
