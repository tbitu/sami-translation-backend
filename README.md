# Sami Translation Backend

FastAPI service that exposes a TartuNLP-compatible translation API for Sami ↔ Norwegian/Finnish using the Hugging Face model `tartuNLP/smugri3_14-finno-ugric-nmt` via Fairseq.

This repo is optimized for GPU inference (CUDA), but can also run on CPU.

## What’s in here

- **HTTP API**: `GET /translation/v2` (capabilities) and `POST /translation/v2` (translate)
- **Docs/OpenAPI**: `/translation/openapi.json`, `/translation/docs`, `/translation/redoc`
- **Model loader**: downloads a Hugging Face snapshot on startup and locates SentencePiece + dictionary files automatically
- **Container support**: Dockerfile uses an NVIDIA NGC PyTorch base image and patches Fairseq for Python 3.11 compatibility

## Requirements

### Runtime

- **Python 3.11** (Fairseq 0.12.x is not compatible with Python 3.12+)
- Linux recommended
- For GPU: NVIDIA driver + CUDA-capable GPU (and a CUDA-enabled PyTorch install)

### Disk

- Expect a multi-GB Hugging Face cache on first startup (model + metadata).

## Quickstart (local, recommended)

1) Create and activate a virtualenv:

```bash
python3.11 -m venv venv
source venv/bin/activate
```

2) Install **PyTorch first** (pick the CUDA build that matches your system):

```bash
# Example for CUDA 12.9
pip install torch --index-url https://download.pytorch.org/whl/cu129
```

3) Install the remaining dependencies:

```bash
pip install -r requirements.txt
```

4) Start the server:

```bash
python main.py
```

Server listens on `http://localhost:8000`.

## Running in production

Use a **single worker** so the model isn’t loaded multiple times in memory:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

If you’re deploying behind Kubernetes / a container orchestrator, prefer **one Uvicorn process per container** and scale via replicas.

## API

### Health

```bash
curl http://localhost:8000/
```

### OpenAPI + docs

- OpenAPI: `GET /translation/openapi.json`
- Swagger UI: `GET /translation/docs`
- ReDoc: `GET /translation/redoc`

### Capabilities (TartuNLP-compatible)

```bash
curl http://localhost:8000/translation/v2
```

Returns a minimal domain config with the supported language pairs.

### Translate

```bash
curl -X POST http://localhost:8000/translation/v2 \
  -H "Content-Type: application/json" \
  -d '{"text":"Bures!","src":"sme","tgt":"nor"}'
```

Response shape:

```json
{"result":"..."}
```

The `text` field can also be an array of strings.

### Language codes

This service exposes **API-facing** codes and maps them to the internal model identifiers:

- `sme` (Northern Sami)
- `smj` (Lule Sami)
- `sma` (South Sami)
- `smn` (Inari Sami)
- `sms` (Skolt Sami)
- `sjd` (Kildin Sami)
- `sje` (Pite Sami)
- `sju` (Ume Sami)
- `nor` (Norwegian Bokmål)
- `fin` (Finnish)

The final set of enabled languages is discovered at startup; check `GET /translation/v2` (or logs) to see what is currently available.

## Reverse proxy / path prefixes

If you serve this app under a URL prefix (e.g. `/api`), the docs/OpenAPI routes need to know the prefix.

This app supports the `X-Forwarded-Prefix` header (e.g. set by NGINX/Traefik). When present, it is used to set the ASGI `root_path`, and OpenAPI’s `servers[0].url` is generated accordingly.

## Configuration

### Model + cache

The model snapshot is downloaded via `huggingface_hub.snapshot_download()` and cached on disk.

Environment variables:

- `HF_CACHE_DIR`: cache directory passed to `snapshot_download()` (recommended override)
- `MODEL_CACHE_DIR`: legacy alias for `HF_CACHE_DIR`
- `HF_TOKEN`: used by Hugging Face Hub for private repos (if applicable)

Offline / no-network modes (best effort):

- `HF_HUB_OFFLINE=1` or `HF_LOCAL_FILES_ONLY=1` or `TRANSFORMERS_OFFLINE=1`

Download tuning:

- `HF_MAX_WORKERS` (default `8`)
- `HF_ETAG_TIMEOUT` seconds (default `30`)

### Forcing specific model files

If the auto-discovery logic can’t find the right files inside the snapshot, you can override:

- `SENTENCEPIECE_MODEL`: absolute path, or a path relative to the snapshot directory
- `FIXED_DICTIONARY`: absolute path, or a path relative to the snapshot directory

### Precision

- `MODEL_DTYPE=fp32` or `USE_FP32=1`: forces model parameters to float32 (higher numerical stability, more VRAM)

## TEST_MODE (fast, deterministic, not “real” inference)

Setting `TEST_MODE=1` starts a lightweight mock translator:

```bash
TEST_MODE=1 python main.py
```

Important:

- No Hugging Face downloads
- No Fairseq import/model loading
- CPU-only
- Returns deterministic canned outputs for a small set of phrases (and a simple fallback for everything else)

Use this for quick API wiring checks—not for performance or model debugging.

## Docker

The Dockerfile defaults to an **NVIDIA NGC PyTorch** base image (`nvcr.io/nvidia/pytorch:*`) so CUDA-enabled PyTorch is available in the container.

### Build

```bash
docker build -t sami-translation-backend:latest .
```

NGC images may require registry login:

```bash
docker login nvcr.io
# Username: $oauthtoken
# Password: <your NGC API key>
```

### Run (GPU + persistent Hugging Face cache)

```bash
docker run --rm -p 8000:8000 --gpus all \
  -e HF_CACHE_DIR=/data/hf \
  -v "$HOME/.cache/huggingface":/data/hf \
  sami-translation-backend:latest
```

### Docker Compose

```bash
export HF_CACHE_HOST_DIR="$HOME/.cache/huggingface"
docker compose up --build
```

For GPU, some setups require:

```bash
docker compose run --gpus all --service-ports sami-translation-backend
```

## Troubleshooting

### CUDA is available: False

- Verify your PyTorch install is CUDA-enabled: `python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"`
- Ensure your NVIDIA driver is installed and compatible with your CUDA runtime

### Slow first startup

First run downloads the model snapshot; subsequent startups reuse the cache.

### OOM (GPU out of memory)

- Close other GPU processes
- Consider running with CPU (no special flag; install CPU-only PyTorch)
- Avoid `MODEL_DTYPE=fp32` if you’re tight on VRAM
