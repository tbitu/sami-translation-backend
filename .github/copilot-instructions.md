## Project snapshot: Sami Translation Backend

This file contains concise, actionable guidance for AI coding agents working on this repository.

Key files
- `main.py` - FastAPI server and startup hooks. Models are loaded on startup via `TranslationService`.
- `translation_service.py` - Core model-loading and translation logic. See `TEST_MODE` handling and env overrides (`SENTENCEPIECE_MODEL`, `FIXED_DICTIONARY`).
- `start.sh` - Example startup script and venv instructions used by developers.
- `README.md` - Installation notes: Python 3.11 required, PyTorch must be installed with CUDA first.
- `tests/` - Integration tests that start the server in `TEST_MODE` and exercise `/translation/v2` endpoints.

Big-picture architecture
- A single-process FastAPI app exposes a small, TartuNLP-compatible translation API under `/translation/*`.
- `TranslationService` is responsible for downloading (via `huggingface_hub.snapshot_download`), locating files (sentencepiece models and dictionaries), and instantiating a Fairseq `TransformerModel`.
- The app expects relatively large model files and is GPU-accelerated. The server is intended to run with a single Uvicorn worker so models are not duplicated in memory.

Developer workflows and commands (explicit)
- There is a project `venv` used by developers. Always activate the virtual environment before running any Python command, tests, or the server. Example (Linux/macOS):

```bash
source venv/bin/activate
```

- Install PyTorch with CUDA support first, then install other requirements. Example (match CUDA to your system):

```bash
# Install PyTorch (example for CUDA 12.9)
pip install torch --index-url https://download.pytorch.org/whl/cu129
# Then install other deps
pip install -r requirements.txt
```

- Run server for development (in `venv`):

```bash
python main.py
```

- `start.sh` is a convenience script that checks `venv` and dependencies before starting the server.

- Tests: run `pytest` from inside the activated `venv`. Tests in `tests/` start a subprocess server with `TEST_MODE=1` (see `tests/test_api.py`).

- IMPORTANT WARNING: NEVER start the server in `TEST_MODE` when attempting to reproduce real runtime or model-related issues. Setting `TEST_MODE=1` enables deterministic mocks and skips real model downloads and inference, so it will not reproduce production behavior or runtime errors. Use `TEST_MODE` only for the fast unit/integration tests under `tests/`.

- Important: the integrated tests use `TEST_MODE` and deterministic mocks — they do not reproduce full runtime behavior (model downloads, true GPU OOMs, or real inference). To reproduce real runtime errors, run the server (with real models) and exercise the API directly (curl or HTTP client) — see the curl example below.

Project-specific conventions & patterns
- TEST_MODE is used to avoid heavy downloads and dependencies during tests. Set `TEST_MODE=1` in the environment for fast, deterministic behavior (see `translation_service.py` and `tests/test_api.py`).
- Language codes: the service accepts `sme` and `nor` as top-level codes (mapped to internal tokens `sme_Latn` / `nob_Latn` in `translation_service.py`).
- Model snapshot discovery: the service searches the HuggingFace snapshot for `*.model` (SentencePiece) and `*dict*.txt` (fixed dictionary); maintainers sometimes override via `SENTENCEPIECE_MODEL` and `FIXED_DICTIONARY` env vars.
- Device handling: prefer `cuda:0` explicitly when CUDA is available. Code attempts multiple ways to move Fairseq wrappers to device (`.to()`, `.cuda()`, underlying `.model`).

Integration points & external dependencies
- HuggingFace (`huggingface_hub.snapshot_download`) for model artifacts.
- Fairseq `TransformerModel` for model loading and generation.
- PyTorch for tensors and device management (must match CUDA version installed).
- FastAPI / Uvicorn for serving the HTTP API.

Examples useful for edits or PRs
- To add a new language mapping, update `lang_map` in `translation_service.py` and the validation in `main.py`.
- To mock additional phrases for tests, extend `mock_map` in `TranslationService` (used when `TEST_MODE` is enabled).

Edge cases agents should watch for
- Missing model files in snapshot: `translation_service.py` raises an OSError listing available files — when editing snapshot logic, preserve helpful diagnostics.
- Moving Fairseq wrappers to device may fail silently; any change touching device logic should preserve multiple fallbacks.

When creating PRs
- Run `pytest` locally; tests start a subprocess server with `TEST_MODE=1` and will fail if `main.py`'s startup changes behavior.
- Avoid importing heavy libraries at module import time. `translation_service.py` intentionally imports Fairseq and huggingface_hub lazily inside the constructor to keep tests lightweight.

If you need clarification
- If a scaffolded change requires runtime access to real model files (non-TEST_MODE), document how to set `SENTENCEPIECE_MODEL` and `FIXED_DICTIONARY` so CI can be configured or maintainers can reproduce locally.

Quick manual reproduction (real errors)

1. Activate venv and start the server (do not set `TEST_MODE`):

```bash
source venv/bin/activate
python main.py
```

2. Reproduce an API call with curl (example Sami -> Norwegian):

```bash
curl -X POST http://localhost:8000/translation/v2 \
  -H "Content-Type: application/json" \
  -d '{"text":"Bures!","src":"sme","tgt":"nor"}'
```

Use the server logs when debugging real issues (model download issues, device placement, OOMs). Tests alone won't capture those runtime-specific failures.

Please review these notes and tell me if you'd like me to expand any section (examples, tests, or CI guidance).
## Project snapshot: Sami Translation Backend

This file contains concise, actionable guidance for AI coding agents working on this repository.

Key files
- `main.py` - FastAPI server and startup hooks. Models are loaded on startup via `TranslationService`.
- `translation_service.py` - Core model-loading and translation logic. See `TEST_MODE` handling and env overrides (`SENTENCEPIECE_MODEL`, `FIXED_DICTIONARY`).
- `start.sh` - Example startup script and venv instructions used by developers.
- `README.md` - Installation notes: Python 3.11 required, PyTorch must be installed with CUDA first.
- `tests/` - Integration tests that start the server in `TEST_MODE` and exercise `/translation/v2` endpoints.

Big-picture architecture
- A single-process FastAPI app exposes a small, TartuNLP-compatible translation API under `/translation/*`.
- `TranslationService` is responsible for downloading (via `huggingface_hub.snapshot_download`), locating files (sentencepiece models and dictionaries), and instantiating a Fairseq `TransformerModel`.
- The app expects relatively large model files and is GPU-accelerated. The server is intended to run with a single Uvicorn worker so models are not duplicated in memory.

Developer workflows and commands (explicit)
- Create and activate a Python 3.11 venv, then install PyTorch with CUDA support before `pip install -r requirements.txt` (see `README.md`).
  - PyTorch install example: `pip install torch --index-url https://download.pytorch.org/whl/cu129` (match CUDA version to your system).
- Run server for development: `python main.py` (starts FastAPI on 0.0.0.0:8000). `start.sh` demonstrates a full startup flow and checks.
- Tests: `pytest` (tests live under `tests/` and expect `TEST_MODE=1` when the server subprocess is started by the test fixture).

Project-specific conventions & patterns
- TEST_MODE is used to avoid heavy downloads and dependencies during tests. Set `TEST_MODE=1` in the environment for fast, deterministic behavior (see `translation_service.py` and `tests/test_api.py`).
- Language codes: the service accepts `sme` and `nor` as top-level codes (mapped to internal tokens `sme_Latn` / `nob_Latn` in `translation_service.py`).
- Model snapshot discovery: the service searches the HuggingFace snapshot for `*.model` (SentencePiece) and `*dict*.txt` (fixed dictionary); maintainers sometimes override via `SENTENCEPIECE_MODEL` and `FIXED_DICTIONARY` env vars.
- Device handling: prefer `cuda:0` explicitly when CUDA is available. Code attempts multiple ways to move Fairseq wrappers to device (`.to()`, `.cuda()`, underlying `.model`).

Integration points & external dependencies
- HuggingFace (`huggingface_hub.snapshot_download`) for model artifacts.
- Fairseq `TransformerModel` for model loading and generation.
- PyTorch for tensors and device management (must match CUDA version installed).
- FastAPI / Uvicorn for serving the HTTP API.

Examples useful for edits or PRs
- To add a new language mapping, update `lang_map` in `translation_service.py` and the validation in `main.py`.
- To mock additional phrases for tests, extend `mock_map` in `TranslationService` (used when `TEST_MODE` is enabled).

Edge cases agents should watch for
- Missing model files in snapshot: `translation_service.py` raises an OSError listing available files — when editing snapshot logic, preserve helpful diagnostics.
- Moving Fairseq wrappers to device may fail silently; any change touching device logic should preserve multiple fallbacks.

When creating PRs
- Run `pytest` locally; tests start a subprocess server with `TEST_MODE=1` and will fail if `main.py`'s startup changes behavior.
- Avoid importing heavy libraries at module import time. `translation_service.py` intentionally imports Fairseq and huggingface_hub lazily inside the constructor to keep tests lightweight.

If you need clarification
- If a scaffolded change requires runtime access to real model files (non-TEST_MODE), document how to set `SENTENCEPIECE_MODEL` and `FIXED_DICTIONARY` so CI can be configured or maintainers can reproduce locally.

Please review these notes and tell me if you'd like me to expand any section (examples, tests, or CI guidance).
