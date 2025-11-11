# Quick Start Guide - GB10 Blackwell GPU

## Step 1: Find Compatible Container

First, let's find which NVIDIA PyTorch container works with your GB10:

```bash
./test_container.sh
```

This will test multiple container versions and tell you which one works.

## Step 2: Update run_docker.sh (if needed)

If the test found a different container than `24.12-py3`, edit `run_docker.sh`:

```bash
# Change this line to the working container:
PYTORCH_IMAGE="nvcr.io/nvidia/pytorch:XX.XX-py3"
```

## Step 3: Run the Processor

```bash
./run_docker.sh
```

This will:
1. Start the compatible PyTorch container
2. Verify GPU support
3. Install dependencies
4. Process all files in `in/` directory
5. Save results to `out/` directory

## Manual Testing (Alternative)

If you prefer to test manually:

```bash
# Try the latest container first
docker run --gpus=all -it --rm \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v ~/Documents/deepseek-ocr:/workspace \
  -v ~/hf-cache:/root/.cache/huggingface \
  -w /workspace \
  nvcr.io/nvidia/pytorch:25.01-py3 bash

# Inside container - verify GPU
python3 - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)
PY

# If GPU works, install dependencies
python3 -m pip install --upgrade pip
pip install "transformers==4.46.3" "tokenizers==0.20.3" einops addict easydict accelerate pillow pymupdf

# Run the processor
python3 run_deepseek_transformers.py in
```

## Troubleshooting

### Error: "not yet supported GPU"

This means the container version doesn't support GB10. Try:

1. Run `./test_container.sh` to find a compatible version
2. Or manually try newer containers: `25.02-py3`, `25.01-py3`, etc.

### Error: "CUDA not available"

Check Docker GPU setup:

```bash
docker run --gpus=all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

If this fails, reinstall NVIDIA Container Toolkit:

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Out of Memory

Edit `run_deepseek_transformers.py` and change:

```python
torch_dtype=torch.bfloat16  # Change to torch.float32 for less memory
```

### Slow Processing

Check GPU is being used:

```bash
# In another terminal
watch -n 1 nvidia-smi
```

You should see Python process using GPU memory.

## What Gets Processed

- **Input**: All files in `in/` directory (PDF, PNG, JPG, JPEG, TIF, TIFF)
- **Output**: Files in `out/` directory
  - PDFs: `filename-p1.csv`, `filename-p2.md`, etc. (one file per page)
  - Images: `filename-p1.csv` or `filename-p1.md`

## Performance

On DGX Spark GB10:
- **PDF rendering**: Parallel (10 workers on 20-core ARM CPU)
- **OCR inference**: Sequential (GPU)
- **Speed**: ~2-5 seconds per page (depends on complexity)

## Next Steps

Once processing completes:

1. Check `out/` directory for results
2. Verify OCR quality
3. Adjust DPI if needed (edit line in `run_deepseek_transformers.py`: `dpi=250`)

For questions or issues, check `README_TRANSFORMERS.md`.
