#!/bin/bash
# Startup script for Sami Translation Backend

set -e

echo "🚀 Starting Sami Translation Backend Server"
echo "=========================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run the following commands:"
    echo "  python -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "✓ Activating virtual environment..."
source venv/bin/activate

# Check if dependencies are installed
if ! python -c "import torch" 2>/dev/null; then
    echo "❌ PyTorch not installed!"
    echo "Please install PyTorch with CUDA support first:"
    echo "  pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124"
    echo "Then install other dependencies:"
    echo "  pip install -r requirements.txt"
    exit 1
fi

if ! python -c "import transformers" 2>/dev/null; then
    echo "❌ Transformers not installed!"
    echo "Please run: pip install -r requirements.txt"
    exit 1
fi

if ! python -c "import fastapi" 2>/dev/null; then
    echo "❌ FastAPI not installed!"
    echo "Please run: pip install -r requirements.txt"
    exit 1
fi

# Check CUDA availability
echo ""
echo "🔍 Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}') if torch.cuda.is_available() else print('Running on CPU')"
echo ""

# Start the server
echo "🌐 Starting FastAPI server on http://localhost:8000"
echo "Press Ctrl+C to stop"
echo ""

# Use bf16/fp16 for efficient GPU memory usage (~9-10GB)
# Uncomment below lines if you need fp32 precision (uses ~36GB):
# export MODEL_DTYPE="fp32"
# export USE_FP32="1"

python main.py
