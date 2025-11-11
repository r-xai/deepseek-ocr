# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Python-based OCR (Optical Character Recognition) batch processing tool that uses DeepSeek-OCR (3B parameter vision-language model) via vLLM for extracting text from images and PDFs. The tool processes files from an `in/` directory and outputs results to an `out/` directory.

DeepSeek-OCR supports multilingual OCR, markdown conversion, and document understanding tasks.

## Architecture

### Two Main Scripts

1. **[batch_ocr.py](batch_ocr.py)** - Basic OCR extraction
   - Converts images and PDFs to markdown text
   - Single output file per input (one `.md` file per image, or one `.md` per PDF with all pages)
   - Straightforward text extraction without formatting instructions

2. **[batch_tables.py](batch_tables.py)** - Table-focused extraction
   - Specialized for extracting structured data from tables
   - Per-page output with slug-based naming (e.g., `filename-p1.csv`)
   - Two extraction modes via prompts:
     - CSV mode (`PROMPT_CSV`) - For tabular data extraction
     - Markdown mode (`PROMPT_MD`) - For structured document conversion with pipe tables
   - Auto-detects CSV vs Markdown output based on content characteristics

### Model Integration

Both scripts use:
- **Model**: `deepseek-ai/DeepSeek-OCR` via vLLM
- **GPU execution**: vLLM handles efficient GPU utilization automatically
- **Initialization**: Model loaded once at startup with DGX Spark optimizations:
  - `gpu_memory_utilization=0.9` - Uses 90% of GB10 GPU memory
  - `max_model_len=4096` - Increased context for large images
  - `enable_prefix_caching=False` - Per DeepSeek-OCR official docs
  - `tensor_parallel_size=1` - Single GPU configuration
- **Inference**: Zero-temperature sampling for deterministic output (`temperature=0`)
- **Multimodal input**: Follows DeepSeek-OCR format: `{"prompt": "<instruction>", "images": [img]}`

### Performance Optimizations for DGX Spark

The code is optimized for Nvidia DGX Spark (GB10 Blackwell GPU, 6144 CUDA cores, 128GB unified memory):

1. **Parallel PDF rendering**: Uses ThreadPoolExecutor with 10 workers to leverage the 20-core ARM CPU (10× Cortex-X925 + 10× Cortex-A725) for CPU-bound PDF-to-image conversion
2. **Sequential GPU processing**: OCR runs sequentially on the single GB10 GPU to avoid memory contention
3. **High GPU memory utilization**: 90% allocation maximizes throughput on the 128GB unified memory architecture

### Processing Flow

1. Scan `in/` directory for supported formats: `*.pdf`, `*.png`, `*.jpg`, `*.jpeg`, `*.tif`, `*.tiff`
2. For PDFs: Extract pages as images (250 DPI via PyMuPDF)
3. Run OCR via vLLM multimodal generation
4. Write output to `out/` directory

## Running the Scripts

```bash
# Basic OCR extraction (outputs markdown)
python batch_ocr.py

# Table extraction (outputs CSV or markdown per page)
python batch_tables.py
```

Both scripts:
- Process all supported files in `./in/` directory
- Create `./out/` directory if it doesn't exist
- Continue processing if individual files fail (with warning)
- Automatically use parallel PDF rendering (10 workers on DGX Spark's 20 cores)
- Show progress indicators for multi-page documents

## Dependencies

Required Python packages:
- `vllm` - For running DeepSeek-OCR model
- `Pillow` (PIL) - Image processing
- `PyMuPDF` (fitz) - PDF rendering to images
- `python-slugify` (batch_tables.py only) - Filename sanitization

Official model requirements (from HuggingFace):
- Python 3.12.9 + CUDA 11.8
- torch==2.6.0
- transformers==4.46.3
- flash-attn==2.7.3

Install with:
```bash
pip install vllm pillow pymupdf python-slugify
```

For direct model usage (non-vLLM):
```bash
pip install torch==2.6.0 transformers==4.46.3 flash-attn==2.7.3
```

## Key Parameters

### PDF Rendering
- **DPI**: 250 (configurable in `ocr_pdf()` function)
  - Trade-off between quality and processing time
  - 200-300 DPI recommended for typical documents
  - Note: DeepSeek-OCR has resolution tiers (512/640/1024/1280) that affect accuracy

### Model Inference
- **max_tokens**: 2048
- **temperature**: 0 (deterministic output, recommended for OCR)
- **vLLM optimization**: Consider `enable_prefix_caching=False` per official docs

### Prompts (batch_tables.py)
- `PROMPT_CSV`: Instructs model to output raw CSV format
- `PROMPT_MD`: Instructs model to output structured markdown with pipe tables
- Official prompt format from docs: `"<image>\n<|grounding|>Convert the document to markdown."` or `"<image>\nFree OCR."`

## Improvements Made

All previously identified issues have been fixed:

1. ✅ **Fixed missing prompt in batch_ocr.py** - Now passes proper instruction: `{"prompt": "Convert the document to markdown.", "images": [img]}`

2. ✅ **Removed unused variable in batch_tables.py** - Cleaned up dead code

3. ✅ **Optimized for DGX Spark** - Added vLLM configuration for GB10 GPU with 90% memory utilization

4. ✅ **Added parallel processing** - PDF page rendering uses 10 threads to leverage 20-core ARM CPU

5. ✅ **Better progress tracking** - Added file counters and per-page progress indicators

## GB10 Blackwell GPU Compatibility (IMPORTANT)

### Issue: vLLM Incompatibility with SM 12.1

The original vLLM-based scripts (`batch_ocr.py`, `batch_tables.py`) **do not work** on GB10 Blackwell GPU due to Triton kernel compilation errors:

```
ptxas fatal: Value 'sm_121a' is not defined for option 'gpu-name'
```

**Root cause**: The bundled `ptxas` in current vLLM versions doesn't support CUDA compute capability 12.1 (sm_121a).

**Attempted fixes that failed**:
- Setting `VLLM_TORCH_COMPILE_LEVEL=0`, `TORCHDYNAMO_DISABLE=1`, `TORCH_COMPILE_DISABLE=1`
- Using `enforce_eager=True` in vLLM initialization
- Disabling Flash Attention with `FLASH_ATTENTION_FORCE_SDPA=1`

All attempts failed because vLLM uses Triton kernels internally regardless of these settings.

### Solution: Use Transformers Library

**Working script**: [run_deepseek_transformers.py](run_deepseek_transformers.py)

This script uses PyTorch Transformers instead of vLLM, avoiding all Triton compilation:

```python
from transformers import AutoProcessor, AutoModel

# Load model (IMPORTANT: use AutoProcessor + AutoModel with trust_remote_code)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModel.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
).to(DEVICE).eval()

# Run inference
inputs = processor(text=prompt, images=img.convert("RGB"), return_tensors="pt").to(DEVICE)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=2048)
text = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
```

**Key API requirements**:
- Must use `AutoProcessor` (handles both text and images)
- Must use `AutoModel` with `trust_remote_code=True` (NOT AutoModelForCausalLM - DeepSeek-OCR uses custom model classes)
- Call pattern: `processor(text=..., images=..., return_tensors="pt")`
- Decode with: `processor.batch_decode(out, skip_special_tokens=True)[0]`

### Docker Container Requirements

GB10 requires recent NVIDIA PyTorch containers. Tested working versions:
- `nvcr.io/nvidia/pytorch:24.12-py3` ✅
- `nvcr.io/nvidia/pytorch:25.01-py3` ✅
- `nvcr.io/nvidia/pytorch:25.02-py3` ✅

Older containers (24.10 and earlier) fail with: "Detected NVIDIA GB10 GPU, which is not yet supported"

**Usage**:
```bash
# Automated approach
./run_docker.sh

# Or test container compatibility first
./test_container.sh
```

See [QUICKSTART.md](QUICKSTART.md) and [README_TRANSFORMERS.md](README_TRANSFORMERS.md) for detailed instructions.

## Potential Future Enhancements

- **Resolution configuration**: Expose DeepSeek-OCR's resolution tier settings (tiny/small/base/large/gundam) as command-line arguments
- **Batch inference**: Explore Transformers batch processing for multiple images simultaneously (may improve GPU utilization)
- **Adaptive DPI**: Auto-adjust PDF rendering DPI based on source document quality
- **vLLM migration**: When vLLM releases SM 12.1 support, consider migrating back for maximum performance
