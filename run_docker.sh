#!/bin/bash

# Script to run DeepSeek-OCR in NVIDIA PyTorch container (GB10/Blackwell compatible)
# This avoids vLLM's Triton/ptxas issues with SM 12.1

echo "Starting NVIDIA PyTorch container (GB10/Blackwell compatible)..."
echo "========================================"
echo ""

# Try newer PyTorch containers that support GB10 GPU
# Start with 24.12, fall back to 25.01 if needed
PYTORCH_IMAGE="nvcr.io/nvidia/pytorch:24.12-py3"

docker run --gpus=all -it --rm \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v ~/Documents/deepseek-ocr:/workspace \
  -v ~/hf-cache:/root/.cache/huggingface \
  -w /workspace \
  ${PYTORCH_IMAGE} bash -c '
    echo "Verifying PyTorch and CUDA support..."
    python3 - <<EOF
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)
else:
    print("WARNING: CUDA not available!")
    exit(1)
EOF

    if [ $? -ne 0 ]; then
        echo "ERROR: PyTorch CUDA verification failed"
        exit 1
    fi

    echo ""
    echo "Installing dependencies..."
    python3 -m pip install --upgrade pip -q
    pip install -q "transformers==4.46.3" "tokenizers==0.20.3" einops addict easydict accelerate pillow pymupdf

    echo ""
    echo "Dependencies installed successfully!"
    echo "========================================"
    echo ""
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo ""
    echo "========================================"
    echo ""
    echo "Running DeepSeek-OCR batch processor..."
    echo ""

    python3 run_deepseek_transformers.py in

    echo ""
    echo "========================================"
    echo "Processing complete! Check out/ directory for results."
'
