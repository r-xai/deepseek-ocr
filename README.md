# DeepSeek-OCR Batch Processing

Local OCR pipeline using [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) (~950M active parameters) to extract text from PDFs and images as clean Markdown. Runs entirely on-device via Docker + NVIDIA GPU.

## Quick Start

```bash
# 1. Drop files into in/
cp my-document.pdf in/

# 2. Run
./run_docker.sh

# 3. Results appear in out/<document-name>/trimmed/
ls out/my-document/trimmed/
```

That's it. The script handles Docker container setup, dependency installation, model loading, and OCR processing automatically.

## What It Does

For each page of every input document, the pipeline runs **two inference passes** and then auto-trims hallucinated repetition:

1. **Grounding pass** — produces markdown with `<|ref|>/<|det|>` bounding box annotations (rich positional data)
2. **Clean pass** — produces plain markdown (human-readable, but may hallucinate on complex pages)
3. **Auto-trim** — detects repetitive hallucination in the clean output and removes it, falling back to the grounding output (with tags stripped) when the clean pass degenerated

Output is organised per-document:

```
out/
  my-document/
    grounding/    # Pass 1: markdown + bounding box annotations
      p1.md, p2.md, ...
    clean/        # Pass 2: raw clean markdown
      p1.md, p2.md, ...
    trimmed/      # Best quality: repetition removed
      p1.md, p2.md, ...
```

**Use `trimmed/` for final results.**

Supports: PDF, PNG, JPG, JPEG, TIF, TIFF.

## Example Output

```
Model loaded on cuda (32.5s)

Found 1 file(s) to process

[1/1] 2025健康体检报告 - Aug 12 2025.pdf
    Page 1 [grounding]... done (6.7s)
    Wrote out/2025健康体检报告 - Aug 12 2025/grounding/p1.md
    Page 1 [clean]... done (2.9s)
    Wrote out/2025健康体检报告 - Aug 12 2025/clean/p1.md
    Wrote out/2025健康体检报告 - Aug 12 2025/trimmed/p1.md
    Page 1 total: 9.8s
    ...
    Page 9 [grounding]... done (16.6s)
    Page 9 [clean]... done (2m 46.3s)
    Page 9 [trimmed] -- repetition detected and removed
    Page 9 total: 3m 3.2s
    ...
  File total: 12m 8.8s

Processing complete! 1 file(s) in 12m 8.8s
Check out/<document-name>/trimmed/ for best results.
```

## Requirements

- **Docker** with NVIDIA GPU support (`nvidia-container-toolkit`)
- **NVIDIA GPU** with CUDA support (tested on DGX Spark GB10 Blackwell)
- **~7 GB disk** for the model (auto-downloaded to `~/hf-cache/` on first run)

## How It Works

1. `run_docker.sh` launches an NVIDIA PyTorch container (`nvcr.io/nvidia/pytorch:25.09-py3`)
2. Installs pinned Python dependencies inside the container (`transformers==4.46.3`)
3. Runs `run_deepseek_transformers.py` which:
   - Loads the DeepSeek-OCR model from HuggingFace cache
   - Renders each PDF page at 250 DPI via PyMuPDF
   - **Pass 1**: inference with `<image>\n<|grounding|>Convert the document to markdown.`
   - **Pass 2**: inference with `<image>\nConvert the document to markdown.`
   - Auto-detects and trims repetitive hallucination
   - Writes per-page Markdown to `out/<document-name>/{grounding,clean,trimmed}/`

### Why Transformers instead of vLLM?

The GB10 Blackwell GPU (SM 12.1) is not yet supported by vLLM's Triton kernel compiler. Using PyTorch Transformers directly avoids this issue entirely while still providing GPU-accelerated inference.

## Performance (DGX Spark GB10)

| Metric | Value |
|--------|-------|
| Model load time | ~30s |
| Simple page (both passes) | ~8-15s |
| Complex page (both passes) | ~20-50s |
| Hallucinating page (hits token max) | ~3 min per pass |
| 10-page PDF total (dual-pass) | ~12 min |

## Known Limitations

- **Repetitive hallucination on complex pages**: Pages with multiple reports side-by-side can trigger repetitive output that fills the 8192 token limit. The auto-trimmer catches this and falls back to the grounding output with tags stripped. Manual review of `trimmed/` output is still recommended.
- **`transformers==4.46.3` pinned**: The model's custom code imports `LlamaFlashAttention2` which was removed in newer transformers versions. Do not upgrade.

## File Structure

```
deepseek-ocr/
├── run_docker.sh                  # Main entry point - run this
├── run_deepseek_transformers.py   # OCR processing script (runs inside Docker)
├── in/                            # Drop input files here
├── out/                           # Results appear here (per-document folders)
│   └── <document-name>/
│       ├── grounding/             # Markdown + bounding box annotations
│       ├── clean/                 # Raw clean markdown
│       └── trimmed/               # Best quality output
├── CLAUDE.md                      # Development notes for Claude Code
└── ~/hf-cache/                    # Model weights (outside repo)
```

## Processing a Specific File

```bash
# Inside the Docker container (or modify run_docker.sh):
python3 run_deepseek_transformers.py path/to/file.pdf
python3 run_deepseek_transformers.py path/to/image.png
```

## License

MIT
