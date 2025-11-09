# DeepSeek-OCR Batch Processing

Batch OCR processing tool using DeepSeek-OCR (3B parameter vision-language model) via vLLM for extracting text from images and PDFs. Optimized for Nvidia DGX Spark.

## Features

- **Batch processing** of images and PDFs from `in/` directory
- **Two processing modes**:
  - `batch_ocr.py` - Converts documents to markdown
  - `batch_tables.py` - Extracts tables as CSV or markdown
- **Optimized for DGX Spark**: Leverages GB10 GPU and 20-core ARM CPU
- **Parallel PDF rendering**: 10 workers for fast page extraction
- **Progress tracking**: Real-time feedback for multi-page documents

## Hardware Requirements

Optimized for Nvidia DGX Spark:
- GB10 Blackwell GPU (6144 CUDA cores)
- 128GB unified memory
- 20-core ARM CPU (10× Cortex-X925 + 10× Cortex-A725)

## Installation

```bash
pip install vllm pillow pymupdf python-slugify
```

## Usage

1. Place files in `./in/` directory (supported: PDF, PNG, JPG, JPEG, TIF, TIFF)
2. Run either script:

```bash
# Basic markdown extraction
python batch_ocr.py

# Table extraction (CSV/markdown per page)
python batch_tables.py
```

3. Find results in `./out/` directory

## Configuration

See [CLAUDE.md](CLAUDE.md) for detailed architecture documentation and development guidance.

## Model

Uses [DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR) - a 3B parameter multimodal model supporting:
- Multilingual OCR
- Markdown conversion
- Document understanding

## Performance Optimizations

- **GPU**: 90% memory utilization on GB10 GPU
- **CPU**: Parallel PDF rendering with 10 threads
- **vLLM**: Optimized inference with `enable_prefix_caching=False`
- **Context**: Extended to 4096 tokens for large images

## License

MIT
