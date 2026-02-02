# Sami Translation Backend

FastAPI service exposing a TartuNLP-compatible translation API (Sami ↔ Finnish/Norwegian) using the Hugging Face model `tartuNLP/Tahetorn_9B` via Transformers.

**Model**: TartuNLP Tahetorn_9B (based on Unbabel/Tower-Plus-9B, Gemma2 9B translation-specialized model)

Primary usage is via **prebuilt Docker images in GHCR**.

> **Note**: v0.2.0 was the last release using the previous model ([tartuNLP/smugri3_14-finno-ugric-nmt](https://huggingface.co/tartuNLP/smugri3_14-finno-ugric-nmt)). Current version uses Tahetorn_9B.

## Quickstart (GHCR + Docker, recommended)

Pull the image:

```bash
docker pull ghcr.io/tbitu/sami-translation-backend:latest
```

Create a persistent cache volume (recommended):

```bash
docker volume create hf-cache
```

Run it (GPU, with a persistent Hugging Face cache):

```bash
docker run --rm -p 8000:8000 --gpus all \
  -e HF_CACHE_DIR=/data/hf \
  -v hf-cache:/data/hf \
  ghcr.io/tbitu/sami-translation-backend:latest
```

Then open:

- Health: `GET http://localhost:8000/health`
- Swagger UI: `GET http://localhost:8000/api/translation/docs`
- OpenAPI: `GET http://localhost:8000/api/translation/openapi.json`

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

- `GET /api/translation` capabilities (language pairs)
- `POST /api/translation` translate (`text` can be a string or list)

Example translate:

```bash
curl -X POST http://localhost:8000/api/translation \
  -H "Content-Type: application/json" \
  -d '{"text":"Bures!","src":"sme","tgt":"nor"}'
```

## Reverse proxy / path prefixes

The app is configured with root path `/api/translation` and honors `X-Forwarded-Prefix` for dynamic proxy path detection. Configure your reverse proxy to pass requests to the app root:

```apache
<Location /api/translation>
    ProxyPass http://localhost:8000
    ProxyPassReverse http://localhost:8000
    RequestHeader set X-Forwarded-Prefix "/api/translation"
</Location>
```

## Configuration (runtime)

Model download + caching (recommended):

- `HF_CACHE_DIR` (preferred) or `MODEL_CACHE_DIR` (legacy): persist HF downloads across restarts (model is ~9GB)
- Offline best-effort: `HF_HUB_OFFLINE=1` / `HF_LOCAL_FILES_ONLY=1` / `TRANSFORMERS_OFFLINE=1` (requires a populated cache)
- Download tuning: `HF_MAX_WORKERS` (default 8), `HF_ETAG_TIMEOUT` seconds (default 30)

Quantization (for resource-constrained deployments):

- `MODEL_QUANTIZATION=8bit` for 8-bit quantization (~5GB VRAM)
- `MODEL_QUANTIZATION=4bit` for 4-bit quantization (~3GB VRAM, slower inference)

Memory optimization:

- **Flash Attention 2** is included for memory-efficient attention computation (requires CUDA compute capability ≥8.0, automatic detection)
- Reduces VRAM usage and improves inference speed on supported GPUs (Ampere/Ada/Hopper architectures)

`TEST_MODE=1` runs a lightweight mock translator (no HF/Transformers, CPU-only) for quick wiring checks.

## Notes for contributors (optional)

- Use **one Uvicorn worker** in production so the multi‑GB model (9GB) isn't duplicated in memory.
- The Docker image is based on an NVIDIA NGC PyTorch base for consistent CUDA support across platforms (including Arm SBSA).
- The model uses Tower-style prompting for translation: prompts are formatted as `"Translate the following {src_lang} source text to {tgt_lang}:\n{src_lang}: {text}\n{tgt_lang}: "` and generation is done via the transformers pipeline.
- Translation quality is optimized for Sami languages based on TartuNLP fine-tuning of Tower-Plus-9B.
## License

This software is licensed under the [MIT License](LICENSE).

**⚠️ IMPORTANT - NON-COMMERCIAL USE ONLY ⚠️**

The neural machine translation model used by this software (`tartuNLP/Tahetorn_9B`, based on Unbabel/Tower-Plus-9B and Google Gemma 2 9B) is developed by [TartuNLP](https://tartunlp.ai/) at the University of Tartu and is licensed under [CC-BY-NC-SA-4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

**The CC-BY-NC-SA-4.0 license PROHIBITS COMMERCIAL USE.** While this software's code is MIT-licensed, the combined work (software + model) cannot be used for commercial purposes.

When using this software, you must:
- Give appropriate attribution to TartuNLP and the University of Tartu
- Comply with all terms of the CC-BY-NC-SA-4.0 license
- NOT use it for commercial purposes

The base model (Unbabel/Tower-Plus-9B) is licensed under CC-BY-NC-4.0, and the base architecture (Google Gemma 2 9B) has its own terms of use.