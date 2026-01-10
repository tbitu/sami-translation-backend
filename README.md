# Sami Translation Backend

FastAPI service exposing a TartuNLP-compatible translation API (Sami ↔ Finnish/Norwegian) using the Hugging Face model `tartuNLP/smugri3_14-finno-ugric-nmt` via Fairseq.

Primary usage is via **prebuilt Docker images in GHCR**.

## Quickstart (GHCR + Docker, recommended)

Pull the image:

```bash
docker pull ghcr.io/tbitu/sami-translation-backend:latest
```

Create a persistent cache volume (recommended):

```bash
docker volume create sami-hf-cache
```

Run it (GPU, with a persistent Hugging Face cache):

```bash
docker run --rm -p 8000:8000 --gpus all \
  -e HF_CACHE_DIR=/data/hf \
  -v sami-hf-cache:/data/hf \
  ghcr.io/tbitu/sami-translation-backend:latest
```

Then open:

- Health: `GET http://localhost:8000/`
- Swagger UI: `GET http://localhost:8000/translation/docs`
- OpenAPI: `GET http://localhost:8000/translation/openapi.json`

Note: if the GHCR package is private, you’ll need `docker login ghcr.io`.

## Docker Compose (recommended for services)

This repo includes a compose file with a persistent HF cache volume.

```bash
docker compose up
```

If you prefer a host bind-mount cache directory instead of a Docker volume, edit `docker-compose.yml` to switch the `volumes:` entry.

If your Docker setup requires an explicit GPU flag:

```bash
docker compose run --gpus all --service-ports sami-translation-backend
```

## API

Endpoints (TartuNLP-compatible):

- `GET /translation/v2` capabilities (language pairs)
- `POST /translation/v2` translate (`text` can be a string or list)

Example translate:

```bash
curl -X POST http://localhost:8000/translation/v2 \
  -H "Content-Type: application/json" \
  -d '{"text":"Bures!","src":"sme","tgt":"nor"}'
```

## Reverse proxy / path prefixes

Docs/OpenAPI are served under `/translation/*` and the app honors `X-Forwarded-Prefix` so it works behind a path prefix (e.g. `/api`).

## Configuration (runtime)

Model download + caching (recommended):

- `HF_CACHE_DIR` (preferred) or `MODEL_CACHE_DIR` (legacy): persist HF downloads across restarts
- Offline best-effort: `HF_HUB_OFFLINE=1` / `HF_LOCAL_FILES_ONLY=1` / `TRANSFORMERS_OFFLINE=1` (requires a populated cache)
- Download tuning: `HF_MAX_WORKERS` (default 8), `HF_ETAG_TIMEOUT` seconds (default 30)

Model file overrides (relative to snapshot allowed):

- `SENTENCEPIECE_MODEL`
- `FIXED_DICTIONARY`

Precision:

- `MODEL_DTYPE=fp32` or `USE_FP32=1` (more stable, more VRAM)

`TEST_MODE=1` runs a lightweight mock translator (no HF/Fairseq, CPU-only) for quick wiring checks.

## Notes for contributors (optional)

- Use **one Uvicorn worker** in production so the multi‑GB model isn’t duplicated in memory.
- The Docker image is based on an NVIDIA NGC PyTorch base and includes in-image Fairseq patches for Python 3.11 + NGC torch version strings; building from source may require access to `nvcr.io/nvidia/pytorch`.
