## Sami Translation Backend (agent notes)

### Big picture
- [main.py](main.py) runs a single-process FastAPI app exposing a TartuNLP-compatible API:
  - `GET /translation/v2` returns capabilities (language pairs)
  - `POST /translation/v2` translates (`text` can be a string or list)
- [translation_service.py](translation_service.py) owns *all* model lifecycle: Hugging Face snapshot download/cache, loading transformers `AutoModelForCausalLM` and `AutoTokenizer`, and translation via Tower-style prompting.
- Deploy with **one Uvicorn worker** so the 9GB model isn't duplicated in memory (see [Dockerfile](Dockerfile) CMD and README).

### Model Architecture
- Uses **tartuNLP/Tahetorn_9B** (based on Unbabel/Tower-Plus-9B, Gemma2 9B)
- **Translation approach**: Chat-based prompting, NOT encoder-decoder NMT
- **Prompt format**: `"Translate the following {src_lang} source text to {tgt_lang}:\n{src_lang}: {text}\n{tgt_lang}: "`
- Model generates translation autoregressively using transformers `pipeline("text-generation")`
- Language codes (API: `sme`, `nor`, `fin`) mapped to human-readable names in prompts (`"Northern Sami"`, `"Norwegian (Bokmål)"`, `"Finnish"`)

### Reverse proxy / prefixes
- The app serves docs/OpenAPI under `/translation/*` and honors `X-Forwarded-Prefix` via middleware so `/translation/openapi.json` and `/translation/docs` work behind a path prefix (see [main.py](main.py)).

### Dev workflows (local)
- Create/activate venv, install CUDA-enabled PyTorch **first**, then `pip install -r requirements.txt` (see [README.md](README.md) and [requirements.txt](requirements.txt)).
- Run the server: `python main.py` (or `uvicorn main:app --workers 1`).
- Use [start.sh](start.sh) as a reference startup script; it exports `MODEL_DTYPE=fp32` / `USE_FP32=1` for higher numerical stability (more VRAM).

### Model loading + configuration
- HF caching: set `HF_CACHE_DIR` (or legacy `MODEL_CACHE_DIR`) to persist downloads (~9GB model); containers default to `/data/hf` (see [translation_service.py](translation_service.py), [Dockerfile](Dockerfile), [docker-compose.yml](docker-compose.yml)).
- Offline best-effort: `HF_HUB_OFFLINE=1` / `HF_LOCAL_FILES_ONLY=1` / `TRANSFORMERS_OFFLINE=1` uses an existing cached snapshot when `HF_CACHE_DIR` is set.
- Quantization: `MODEL_QUANTIZATION=8bit` or `=4bit` for reduced VRAM (useful for resource-constrained deployments).
- Language codes are **API-facing** (3-letter codes) and mapped to human-readable display names for prompts via `LANGUAGE_DISPLAY_NAMES`.

### TEST_MODE
- Setting `TEST_MODE=1` makes `TranslationService` a lightweight mock (no HF download, no Transformers import, CPU-only) and returns deterministic canned outputs for a few phrases.
- Don't use `TEST_MODE` to debug real model/GPU issues.

### Docker/CI gotchas (important)
- [Dockerfile](Dockerfile) uses an NVIDIA NGC PyTorch base for consistent CUDA support across platforms (including Arm SBSA/DX Spark).
- Unlike the previous Fairseq-based setup, no Python patches are needed - transformers installs cleanly.
- GitHub Actions workflow [docker-build.yml](.github/workflows/docker-build.yml) can skip builds if `nvcr.io/nvidia/pytorch:*` isn't accessible from runners.

### Repo conventions
- Avoid importing heavy deps at module import time; Transformers + HF Hub are imported lazily inside `TranslationService.__init__` (keep this pattern when refactoring).
- The model is large (9B parameters, ~9GB in fp16/bf16) - expect longer download times and higher VRAM usage than previous version.

### Version history
- **v1.0.0+**: Current version uses Tahetorn_9B via Transformers
- **v0.2.0**: Last release using the previous model (tartuNLP/smugri3_14-finno-ugric-nmt)
