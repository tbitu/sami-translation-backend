# Translation Backend Server

This backend server provides local Northern Sami ↔ Norwegian translation using TartuNLP models from HuggingFace, running on NVIDIA GPU for fast inference.

## Requirements

### Hardware
- **NVIDIA GPU** with CUDA support (recommended: 4GB+ VRAM)
- At least 8GB RAM
- 5GB free disk space for models

### Software
- **Python 3.11** (Required - fairseq is not compatible with Python 3.12)
- CUDA Toolkit 12.9+ (for GPU support)
- pip (Python package manager)

> **⚠️ Important:** This project requires Python 3.11. Fairseq (the translation library) has compatibility issues with Python 3.12 due to dataclass changes.

## Installation

### 1. Install CUDA (if not already installed)

Check if CUDA is installed:
```bash
nvidia-smi
```

If not installed, download from: https://developer.nvidia.com/cuda-downloads

### 2. Create Python Virtual Environment

> **Quick Setup:** If you're on Linux and need to install Python 3.11, you can use the provided setup script:
> ```bash
> ./setup_python311.sh
> ```
> This will install Python 3.11 (if needed), create a new virtual environment, and install all dependencies.

**Manual Setup:**

```bash
# Install Python 3.11 if needed (Ubuntu/Debian)
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev

# Create virtual environment with Python 3.11
python3.11 -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows
```

### 3. Install Dependencies

**Important:** Install PyTorch with CUDA support FIRST before other dependencies.

**Step 1: Install PyTorch with CUDA 12.9 support:**

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu129
```

**Step 2: Install remaining dependencies:**

```bash
pip install -r requirements.txt
```

This will install:
- **PyTorch with CUDA 12.9**: Deep learning framework with GPU support (installed first)
- **FastAPI**: Web framework for the API
- **Uvicorn**: ASGI server
- **Fairseq**: Library for TartuNLP's Finno-Ugric NMT models
- **SentencePiece**: Tokenizer for the translation models
- **Accelerate**: For optimized model loading

### 4. Download Models (Automatic on First Run)

The TartuNLP smugri3_14 model will be automatically downloaded from HuggingFace on first startup:
- **tartuNLP/smugri3_14-finno-ugric-nmt**: Multilingual Finno-Ugric NMT model supporting 27 languages including Northern Sami and Norwegian Bokmål (bidirectional, ~2GB)

This is a specialized model trained on Finno-Ugric and Uralic language families, providing excellent quality for Northern Sami.

Models are cached in `~/.cache/huggingface/` for future use.

## Running the Server

### Development Mode

```bash
cd backend
source venv/bin/activate  # Activate virtual environment
python main.py
```

The server will start on `http://localhost:8000`

### Production Mode

```bash
cd backend
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

**Note**: Use only 1 worker to avoid loading models multiple times in memory.

## Verifying GPU Usage

Check the startup logs for:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 3080
GPU Memory: 10.00 GB
```

Monitor GPU usage while translating:
```bash
watch -n 1 nvidia-smi
```

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/
```

Response:
```json
{
  "status": "ok",
  "service": "Sami Translation API",
  "cuda_available": true,
  "device": "cuda:0"
}
```

### Translation
```bash
curl -X POST http://localhost:8000/translation/v2 \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Bures!",
    "src": "sme",
    "tgt": "nor"
  }'
```

Response:
```json
{
  "result": "Hei!"
}
```

## Language Codes

- `sme`: Northern Sami (Davvisámegiella)
- `smj`: Lule Sami
- `sma`: South Sami
- `smn`: Inari Sami
- `sms`: Skolt Sami
- `sjd`: Kildin Sami
- `sje`: Pite Sami
- `sju`: Ume Sami
- `fin`: Finnish
- `nor`: Norwegian (Bokmål)

## Performance

### First Translation
- Initial model loading: ~10-30 seconds (one-time on startup)
- First inference: ~2-5 seconds (JIT compilation)

### Subsequent Translations
- Short texts (<50 words): ~100-500ms
- Long texts (>200 words): ~1-3 seconds

GPU acceleration provides 5-10x speedup compared to CPU inference.

## Troubleshooting

### GPU Not Detected

**Problem**: Server starts but shows `CUDA available: False`

**Solutions**:
1. Verify CUDA installation: `nvcc --version` (should show 12.x)
2. Check PyTorch CUDA support:
   ```python
   import torch
   print(torch.cuda.is_available())
   print(torch.version.cuda)  # Should show 12.9
   ```
3. Reinstall PyTorch with CUDA 12.9:
   ```bash
   pip uninstall torch
   pip install torch --index-url https://download.pytorch.org/whl/cu129
   ```

### Out of Memory (OOM)

**Problem**: `CUDA out of memory` error

**Solutions**:
1. Close other GPU applications
2. Use smaller batch sizes (models already use batch_size=1)
3. Use CPU mode if GPU has <4GB VRAM:
   ```python
   # In translation_service.py, change:
   self.device = torch.device("cpu")
   ```

### Slow First Request

**Problem**: First translation takes 30+ seconds

**Cause**: Models need to be downloaded from HuggingFace on first run.

**Solution**: Wait for initial download. Subsequent runs will use cached models.

### Port Already in Use

**Problem**: `Address already in use` error

**Solution**: Change port or kill existing process:
```bash
# Find process using port 8000
lsof -i :8000
# Kill it
kill -9 <PID>
# Or use a different port
python main.py --port 8001
```

## Updating Frontend Configuration

The frontend is already configured to use `http://localhost:8000` by default.

To use a different host/port, create a `.env` file in the project root:
```bash
VITE_TRANSLATION_API_URL=http://your-server:8000/translation/v2
```

Then rebuild the frontend:
```bash
npm run build
```

## Production Deployment

### Option 1: Systemd Service (Linux)

Create `/etc/systemd/system/sami-translation.service`:
```ini
[Unit]
Description=Sami Translation API
After=network.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/home/youruser/repos/sami-chat-spa/backend
Environment="PATH=/home/youruser/repos/sami-chat-spa/backend/venv/bin"
ExecStart=/home/youruser/repos/sami-chat-spa/backend/venv/bin/python main.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable sami-translation
sudo systemctl start sami-translation
sudo systemctl status sami-translation
```

### Option 2: Docker (Coming Soon)

Docker support is included. The key design goal is that large HuggingFace model
downloads are cached *outside* the container filesystem so restarts/upgrades do
not re-download multi-GB artifacts.

#### How caching works (recommended)

HuggingFace Hub caches downloads under `HF_HOME` / `HF_HUB_CACHE`.
This repo’s container setup pins those to `/data/hf`.

If you want Docker and non-Docker processes on the host to share the same cache
(recommended), bind-mount the host HuggingFace cache directory into the container:
`$HOME/.cache/huggingface` → `/data/hf`.

If you run Docker commands from inside a devcontainer (where `$HOME` may be `/root`),
do **not** use `$HOME/.cache/...` in `-v` arguments. Instead, set an explicit host
path via `HF_CACHE_HOST_DIR`.

The app also supports `HF_CACHE_DIR` (or `MODEL_CACHE_DIR`) which is passed to
`huggingface_hub.snapshot_download()`.

#### Docker (GPU) quickstart

Prereqs:
- Docker Engine
- NVIDIA drivers + NVIDIA Container Toolkit (so `--gpus all` works)

Build:
```bash
docker build -t sami-translation-backend:latest .
```

This project is GPU-first. For `linux/arm64` GPU builds (like DGX Spark / Arm SBSA),
installing CUDA-enabled PyTorch via `pip` is often not viable because official
CUDA wheels may not exist for that platform.

So the Dockerfile defaults to an NVIDIA NGC PyTorch base image (`nvcr.io/nvidia/pytorch:*`).
The newest NGC tags ship Python 3.12, so the Dockerfile bootstraps a Python 3.11 runtime
inside the container (via micromamba) to keep `fairseq==0.12.2` working.
which is published with multi-arch support and includes PyTorch + CUDA already.

With your host driver `580.95.05`, you can run containers based on CUDA 12.8
(NGC 25.01 requires driver 570+) and CUDA 12.6 (NGC 24.10 requires driver 560+).
The default base tag is `24.10-py3` to avoid Python 3.12 compatibility risk.

If you want to pin a different NGC tag, override the build arg:
```bash
docker build \
  --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:25.12-py3 \
  -t sami-translation-backend:latest .
```

NGC images require a registry login:
```bash
docker login nvcr.io
# Username: $oauthtoken
# Password: <your NGC API key>
```

Run with a persistent HF cache:
```bash
export HF_CACHE_HOST_DIR="$HOME/.cache/huggingface"

docker run --rm -p 8000:8000 --gpus all \
  -e MODEL_DTYPE=fp32 \
  -e HF_CACHE_DIR=/data/hf \
  -v "$HF_CACHE_HOST_DIR":/data/hf \
  sami-translation-backend:latest
```

#### Rootless Docker + shared cache

Yes — HuggingFace caching works fine in rootless Docker because it’s just files
on disk. The only requirement is that the mounted cache directory is writable
by the user your rootless Docker daemon runs as.

Two common approaches:

1) **Named volume (easy, per-user)**
   - Good when all containers run under the same rootless Docker user.
   - Volumes live under `~/.local/share/docker/volumes` for that user.

2) **Bind mount a shared host directory (most shareable)**
   - Good when you want *many* different containers/images/stacks to share one cache path.
   - Example using the host user’s default HF cache location:
     ```bash
     export HF_CACHE_HOST_DIR="$HOME/.cache/huggingface"

     docker run --rm -p 8000:8000 --gpus all \
       -e HF_CACHE_DIR=/data/hf \
       -v "$HF_CACHE_HOST_DIR":/data/hf \
       sami-translation-backend:latest
     ```
   - Or use a dedicated directory like `/srv/hf-cache` (ensure permissions allow writes).

#### Docker Compose

Starts the service with the host HuggingFace cache bind-mounted at `/data/hf`:
```bash
export HF_CACHE_HOST_DIR="$HOME/.cache/huggingface"
docker compose up --build
```

If you prefer a named Docker volume instead (not shared with non-Docker runs), use the
commented `hf-cache` volume option in `docker-compose.yml`.

For GPU, many setups require:
```bash
docker compose run --gpus all --service-ports sami-translation-backend
```

#### CI (GHCR builds)

The GitHub Actions workflow builds multi-arch images. Because the Dockerfile pulls
an NGC base image by default, you need to add an NGC API key as the repository
secret `NGC_API_KEY` (used to `docker login nvcr.io`).

#### Kubernetes note

Mount a PVC at `/data/hf` (or set `HF_CACHE_DIR` to your mounted path). This is
the most reliable way to keep the HuggingFace cache across pod restarts.

## Model Information

The translation models are provided by TartuNLP (University of Tartu) as part of their Uralic language machine translation project:
- **Models**: TartuNLP/opus-mt-urj-mul and TartuNLP/opus-mt-mul-urj
- **Paper**: "Neural Machine Translation for Low-Resource Uralic Languages"
- **License**: Apache 2.0
- **Training Data**: OPUS corpus with enhanced Uralic language data
- **Model Type**: Transformer-based sequence-to-sequence models
- **Languages**: Specialized for Uralic language family (Finnish, Estonian, Northern Sami, etc.)

These models are specifically optimized for Northern Sami (Davvisámegiella), which is spoken primarily in northern Norway, Sweden, and Finland. They provide better quality than generic multilingual models due to their focus on Uralic languages.

## Support

For issues related to:
- **Backend server**: Check logs in terminal
- **Model loading**: Ensure internet connection for first download
- **GPU issues**: Verify NVIDIA drivers and CUDA toolkit
- **Translation quality**: Report to TartuNLP/Helsinki-NLP

## License

Backend code: MIT License
Translation models: Apache 2.0 (Helsinki-NLP)
