## Sami Translation Backend (agent notes)

### Big picture
- [main.py](main.py) runs a single-process FastAPI app exposing a TartuNLP-compatible API:
  - `GET /translation/v2` returns capabilities (language pairs)
  - `POST /translation/v2` translates (`text` can be a string or list)
- [translation_service.py](translation_service.py) owns *all* model lifecycle: Hugging Face snapshot download/cache, locating `*.model` + `*dict*.txt`, and loading a Fairseq `TransformerModel`.
- Deploy with **one Uvicorn worker** so the multi‑GB model isn’t duplicated in memory (see [Dockerfile](Dockerfile) CMD and README).

### Reverse proxy / prefixes
- The app serves docs/OpenAPI under `/translation/*` and honors `X-Forwarded-Prefix` via middleware so `/translation/openapi.json` and `/translation/docs` work behind a path prefix (see [main.py](main.py)).

### Dev workflows (local)
- Create/activate venv, install CUDA-enabled PyTorch **first**, then `pip install -r requirements.txt` (see [README.md](README.md) and [requirements.txt](requirements.txt)).
- Run the server: `python main.py` (or `uvicorn main:app --workers 1`).
- Use [start.sh](start.sh) as a reference startup script; it also exports `MODEL_DTYPE=fp32` / `USE_FP32=1` for higher numerical stability (more VRAM).

### Model loading + configuration
- HF caching: set `HF_CACHE_DIR` (or legacy `MODEL_CACHE_DIR`) to persist downloads; containers default to `/data/hf` (see [translation_service.py](translation_service.py), [Dockerfile](Dockerfile), [docker-compose.yml](docker-compose.yml)).
- Offline best-effort: `HF_HUB_OFFLINE=1` / `HF_LOCAL_FILES_ONLY=1` / `TRANSFORMERS_OFFLINE=1` uses an existing cached snapshot when `HF_CACHE_DIR` is set.
- File overrides (relative to snapshot allowed): `SENTENCEPIECE_MODEL`, `FIXED_DICTIONARY`.
- Language codes are **API-facing** and mapped to internal Fairseq identifiers via `SUPPORTED_LANGUAGE_ALIASES`.

### TEST_MODE
- Setting `TEST_MODE=1` makes `TranslationService` a lightweight mock (no HF download, no Fairseq import, CPU-only) and returns deterministic canned outputs for a few phrases.
- Don’t use `TEST_MODE` to debug real model/GPU issues.

### Docker/CI gotchas (important)
- [Dockerfile](Dockerfile) intentionally uses an NVIDIA NGC PyTorch base and prevents pip from replacing CUDA torch; Fairseq is installed with `--no-deps` and then patched for Python 3.11 + torch version parsing.
- GitHub Actions workflow [docker-build.yml](.github/workflows/docker-build.yml) can skip builds if `nvcr.io/nvidia/pytorch:*` isn’t accessible from runners.

### Repo conventions
- Avoid importing heavy deps at module import time; Fairseq + HF Hub are imported lazily inside `TranslationService.__init__` (keep this pattern when refactoring).
