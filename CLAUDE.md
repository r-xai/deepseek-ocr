# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Local OCR batch processing pipeline using DeepSeek-OCR (~950M active parameters) via PyTorch Transformers. Runs dual-pass inference (grounding + clean) per page, auto-trims repetitive hallucination, and outputs structured Markdown. Runs inside an NVIDIA Docker container on DGX Spark (GB10 Blackwell GPU).

## Architecture

### Primary Script

**`run_deepseek_transformers.py`** — the main OCR processor. Uses Transformers (not vLLM) to avoid Triton/ptxas SM 12.1 incompatibility on GB10.

Key design decisions:
- **Dual-pass inference** — each page is processed twice: first with `PROMPT_GROUNDING` (bounding box annotations), then with `PROMPT_MD` (clean markdown). The grounding pass serves as a reference for trimming.
- **`eval_mode=True`** in `model.infer()` — returns decoded text as a string. Without this, the method streams to stdout and returns `None`.
- **Prompt constants**:
  - `PROMPT_MD = "<image>\nConvert the document to markdown."` — clean Markdown
  - `PROMPT_GROUNDING = "<image>\n<|grounding|>Convert the document to markdown."` — Markdown + bounding box annotations
- **`trust_remote_code=True`** required on both `AutoProcessor` and `AutoModel` — the model uses custom classes (`DeepseekOCRForCausalLM`).
- **`transformers==4.46.3`** pinned — newer versions removed `LlamaFlashAttention2` which the model's `modeling_deepseekv2.py` imports.
- **Image reuse** — each page image is saved to a temp file once and reused for both inference passes.
- **Timing** — per-page (per-pass), per-file, and total job timing is printed to stdout.

### Repetition Detection & Trimming

The `produce_trimmed_output()` function applies two heuristics:

1. **Sliding-window detection** (`detect_repetition`): normalises each line (strip markdown, numbers, whitespace), then slides a window of 5 lines. If 4+ lines are ≥70% similar (via `SequenceMatcher`) to the anchor, a repetition block is found. Walks backwards to locate the first occurrence and truncates after it.
2. **Distinct-ratio check** (`detect_repetition_by_ratio`): if the ratio of distinct normalised lines to total non-empty lines falls below 40%, the output is flagged as degenerate.

When repetition is detected, the trimmer truncates the clean output. If the stripped grounding output has >1.3× more non-empty lines, it is used instead (with `<|ref|>/<|det|>` tags removed by `strip_grounding_tags`).

### Output Structure

```text
out/
  {document-name}/
    grounding/    # Pass 1: markdown with <|ref|>/<|det|> bounding box tags
    clean/        # Pass 2: raw clean markdown (may contain hallucination)
    trimmed/      # Post-processed: repetition removed, best quality
```

### Docker Entry Point

**`run_docker.sh`** — launches `nvcr.io/nvidia/pytorch:25.09-py3`, installs deps, runs the processor. Mounts:
- `~/Documents/deepseek-ocr` → `/workspace`
- `~/hf-cache` → `/root/.cache/huggingface`

### Legacy Scripts (do not work on GB10)

- `batch_ocr.py`, `batch_tables.py` — vLLM-based scripts that fail with `ptxas fatal: Value 'sm_121a' is not defined`

## Running

```bash
# Drop files in in/, run:
./run_docker.sh

# Or process a specific file (inside Docker container):
python3 run_deepseek_transformers.py path/to/file.pdf
```

## Dependencies (installed inside Docker container)

```text
transformers==4.46.3
tokenizers==0.20.3
einops addict easydict accelerate pillow pymupdf python-slugify
```

## Known Issues

### Repetitive hallucination on multi-panel pages
Pages with two reports side-by-side (e.g., scanned lab reports) cause the model to degenerate into repeating a single phrase until hitting the 8192 token max. This takes ~2-3 minutes per page. The auto-trimmer handles most cases, but manual review of `trimmed/` output is still recommended.

### `no_repeat_ngram_size` not fully effective
The model code sets `no_repeat_ngram_size=20` (streaming mode) or `35` (eval mode) but this doesn't prevent all repetition patterns, especially when each repeated line has a different numbering prefix.

### Grounding fallback format
When the trimmer falls back to stripped grounding output, the result may contain structural labels like `title`, `table`, `text`, `image` at the start of content blocks. These come from the grounding output format and are readable but differ in style from the clean pass output.

### Prompt modes
| Prompt | Output |
| --- | --- |
| `<image>\nConvert the document to markdown.` | Clean Markdown |
| `<image>\n<\|grounding\|>Convert the document to markdown.` | Markdown + bounding box annotations |
| `<image>\nFree OCR.` | Raw text extraction (no layout) |

## Performance (DGX Spark GB10)

- Model load: ~30s
- Simple page (both passes): ~8-15s
- Complex page (both passes): ~20-50s
- Hallucinating page (hits token max): ~3 min per pass
- 10-page PDF total (dual-pass): ~12 min
