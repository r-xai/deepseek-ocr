# DeepSeek-OCR with Transformers (GB10/Blackwell Compatible)

This alternative implementation uses **Transformers** instead of vLLM to avoid Triton/ptxas issues with the Blackwell GB10 GPU (SM 12.1).

## Quick Start

### Option 1: Automated Docker Script (Recommended)

Simply run:

```bash
./run_docker.sh
```

This will:
1. Start NVIDIA PyTorch container
2. Install dependencies
3. Process all files in `in/` directory
4. Save results to `out/` directory

### Option 2: Manual Docker Commands

```bash
# 1. Start container
docker run --gpus=all -it --rm \
  -v ~/Documents/deepseek-ocr:/workspace \
  -v ~/hf-cache:/root/.cache/huggingface \
  -w /workspace \
  nvcr.io/nvidia/pytorch:24.10-py3 bash

# 2. Install dependencies (inside container)
python3 -m pip install --upgrade pip
pip install "transformers==4.46.3" "tokenizers==0.20.3" einops addict easydict accelerate pillow pymupdf python-slugify

# 3. Run the processor
python3 run_deepseek_transformers.py in

# 4. Exit container
exit
```

### Option 3: Process Specific File

```bash
# Inside the Docker container
python3 run_deepseek_transformers.py path/to/file.pdf
python3 run_deepseek_transformers.py path/to/image.png
```

## How It Works

- **Uses PyTorch Transformers** instead of vLLM (no Triton kernels)
- **GPU-accelerated** via native PyTorch operations
- **Parallel PDF rendering** with 10 workers (leverages 20-core ARM CPU)
- **Auto-detects** CSV vs Markdown output
- **Per-page output** with slug-based naming

## Output Format

- **PDFs**: `filename-p1.csv`, `filename-p2.md`, etc.
- **Images**: `filename-p1.csv` or `filename-p1.md`

## Why This Works

The GB10 Blackwell GPU has CUDA compute capability 12.1 (`sm_121a`). vLLM's current version uses Triton kernels that try to compile for this architecture, but the bundled `ptxas` doesn't support it yet.

This Transformers-based solution:
- ✅ Avoids all Triton compilation
- ✅ Uses standard PyTorch CUDA kernels
- ✅ Runs natively on GB10
- ✅ Still leverages GPU acceleration
- ✅ Maintains parallel PDF rendering

## Performance

While slightly slower than optimized vLLM with Triton kernels, this solution:
- Fully utilizes the GB10 GPU's 6144 CUDA cores
- Uses BF16 precision for efficiency
- Processes pages in parallel (CPU rendering)
- Runs OCR sequentially on GPU (avoids memory contention)

## Monitoring

Check GPU usage while running:

```bash
# In another terminal
docker exec -it $(docker ps -q) nvidia-smi
```

Or inside the container:

```bash
watch -n 1 nvidia-smi
```

## Troubleshooting

### Out of Memory
Reduce batch size or use FP32 instead of BF16 (edit script line with `torch_dtype`).

### Slow Processing
- Check GPU is being used: `nvidia-smi` should show Python process
- Verify DPI setting (250 is balanced, lower = faster)
- Check CPU workers (10 is optimal for 20-core CPU)

### Wrong Output Format
The script auto-detects CSV vs Markdown. To force:
- Edit `csv_mode=True/False` in the `process_file` function calls

## Migration Back to vLLM

When vLLM releases a version with SM 12.1 support, you can switch back to the original `batch_ocr.py` and `batch_tables.py` scripts for maximum performance.
