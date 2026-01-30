#!/usr/bin/env python3
"""
DeepSeek-OCR dual-pass batch processor using Transformers (instead of vLLM).
Works on Blackwell GB10 GPU (SM 12.1) without Triton/ptxas issues.

For each page, runs two inference passes:
  1. Grounding — markdown with <|ref|>/<|det|> bounding box annotations
  2. Clean    — plain markdown (may hallucinate on complex pages)
Then auto-trims repetitive hallucination from the clean output, falling
back to tag-stripped grounding when the clean pass degenerated.

Usage:
    python3 run_deepseek_transformers.py <file-or-folder>
    python3 run_deepseek_transformers.py in/

Requires:
    pip install transformers==4.46.3 tokenizers==0.20.3 einops addict easydict accelerate pillow pymupdf
"""

import os, io, sys, re, tempfile, time
from difflib import SequenceMatcher

# Disable CUDA JIT compilation and torch.compile (GB10 SM 12.1 not supported)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

from PIL import Image
import fitz  # PyMuPDF
import torch
from transformers import AutoProcessor, AutoModel

MODEL_ID = "deepseek-ai/DeepSeek-OCR"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IN_DIR, OUT_DIR = "in", "out"
os.makedirs(OUT_DIR, exist_ok=True)

PROMPT_MD = "<image>\nConvert the document to markdown."
PROMPT_GROUNDING = "<image>\n<|grounding|>Convert the document to markdown."

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def infer_on_image(img_path: str, prompt: str) -> str:
    """Run OCR inference on a saved image file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = model.infer(
            tokenizer=processor,
            prompt=prompt,
            image_file=img_path,
            output_path=tmpdir,
            base_size=1024,
            image_size=640,
            crop_mode=True,
            save_results=False,
            eval_mode=True,
        )
        return result.strip() if isinstance(result, str) else str(result).strip()

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def ensure_output_dirs(stem: str):
    """Create the three-folder output structure for a document."""
    for subdir in ("grounding", "clean", "trimmed"):
        os.makedirs(os.path.join(OUT_DIR, stem, subdir), exist_ok=True)


def write_output(stem: str, pass_name: str, page_idx: int, text: str):
    """Write OCR output to out/{stem}/{pass_name}/p{N}.md."""
    out_path = os.path.join(OUT_DIR, stem, pass_name, f"p{page_idx + 1}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"    Wrote {out_path}")

# ---------------------------------------------------------------------------
# Repetition detection
# ---------------------------------------------------------------------------

def _normalize_line(line: str) -> str:
    """Strip markdown formatting, leading numbers, and collapse whitespace."""
    s = line.strip()
    s = re.sub(r"^\d+[\.\)]\s*", "", s)          # leading "1. " / "2) "
    s = re.sub(r"[#*_|`\-]", "", s)               # markdown chars
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _line_similarity(a: str, b: str) -> float:
    """SequenceMatcher ratio between two normalized lines (0.0–1.0)."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def detect_repetition(text: str,
                      window_size: int = 5,
                      similarity_threshold: float = 0.7,
                      min_similar: int = 4) -> "int | None":
    """
    Detect where repetitive hallucination begins.

    Slides a window of *window_size* lines through the text.  If
    *min_similar* of the non-anchor lines in the window are ≥ threshold
    similar to the anchor, a repetition block has been found.

    Returns the 0-based line index of the last "good" line (inclusive),
    or None if no repetition is detected.
    """
    lines = text.split("\n")
    if len(lines) < window_size + min_similar:
        return None

    normalized = [_normalize_line(l) for l in lines]

    for i in range(len(lines) - window_size):
        anchor = normalized[i]
        if len(anchor) < 5:
            continue

        similar_count = sum(
            1
            for j in range(i + 1, i + window_size)
            if _line_similarity(anchor, normalized[j]) >= similarity_threshold
        )

        if similar_count >= min_similar:
            # Walk backwards — the anchor itself may already be a repeat.
            first_occurrence = i
            for k in range(i - 1, -1, -1):
                if _line_similarity(anchor, normalized[k]) >= similarity_threshold:
                    first_occurrence = k
                else:
                    break
            return first_occurrence  # keep lines 0..first_occurrence

    return None


def detect_repetition_by_ratio(text: str,
                               threshold: float = 0.4) -> bool:
    """Return True if the distinct/total normalised-line ratio is below *threshold*."""
    lines = text.split("\n")
    normalized = [_normalize_line(l) for l in lines if l.strip()]
    if len(normalized) < 10:
        return False
    return len(set(normalized)) / len(normalized) < threshold

# ---------------------------------------------------------------------------
# Grounding-tag stripping
# ---------------------------------------------------------------------------

def strip_grounding_tags(text: str) -> str:
    """Remove <|ref|>/<|/ref|>/<|det|>…<|/det|> tags, keeping inner ref text."""
    text = re.sub(r"<\|det\|>.*?<\|/det\|>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|ref\|>", "", text)
    text = re.sub(r"<\|/ref\|>", "", text)
    text = re.sub(r"<\|grounding\|>", "", text)
    text = re.sub(r"  +", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# ---------------------------------------------------------------------------
# Trimming / merging
# ---------------------------------------------------------------------------

def produce_trimmed_output(clean_text: str, grounding_text: str) -> tuple:
    """
    Return (trimmed_text, was_trimmed) where *was_trimmed* is True when
    repetition was detected and removed.
    """
    trim_point = detect_repetition(clean_text)
    low_ratio = detect_repetition_by_ratio(clean_text)

    if trim_point is not None:
        lines = clean_text.split("\n")
        trimmed = "\n".join(lines[: trim_point + 1])

        # If grounding captured more content, prefer it.
        stripped = strip_grounding_tags(grounding_text)
        stripped_nonempty = [l for l in stripped.split("\n") if l.strip()]
        trimmed_nonempty = [l for l in trimmed.split("\n") if l.strip()]

        if len(stripped_nonempty) > len(trimmed_nonempty) * 1.3:
            return stripped, True
        return trimmed, True

    if low_ratio:
        return strip_grounding_tags(grounding_text), True

    return clean_text, False

# ---------------------------------------------------------------------------
# PDF rendering
# ---------------------------------------------------------------------------

def pdf_pages(path: str, dpi: int = 250):
    """Yield (page_index, PIL.Image) for each page of a PDF."""
    doc = fitz.open(path)
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        yield i, Image.open(io.BytesIO(pix.tobytes("png")))

# ---------------------------------------------------------------------------
# Per-page processing (dual-pass + trim)
# ---------------------------------------------------------------------------

def process_page(stem: str, page_idx: int, img: Image.Image):
    """Run grounding + clean passes on one page, then produce trimmed output."""
    # Save image once, reuse for both passes
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "input.png")
        img.convert("RGB").save(img_path, "PNG")

        # Pass 1 — grounding
        print(f"    Page {page_idx + 1} [grounding]...", end="", flush=True)
        t0 = time.time()
        grounding_text = infer_on_image(img_path, PROMPT_GROUNDING)
        t1 = time.time()
        print(f" done ({fmt_duration(t1 - t0)})")
        write_output(stem, "grounding", page_idx, grounding_text)

        # Pass 2 — clean
        print(f"    Page {page_idx + 1} [clean]...", end="", flush=True)
        t2 = time.time()
        clean_text = infer_on_image(img_path, PROMPT_MD)
        t3 = time.time()
        print(f" done ({fmt_duration(t3 - t2)})")
        write_output(stem, "clean", page_idx, clean_text)

    # Trim / merge
    trimmed_text, was_trimmed = produce_trimmed_output(clean_text, grounding_text)
    write_output(stem, "trimmed", page_idx, trimmed_text)
    if was_trimmed:
        print(f"    Page {page_idx + 1} [trimmed] -- repetition detected and removed")

# ---------------------------------------------------------------------------
# File-level processing
# ---------------------------------------------------------------------------

def fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    return f"{int(m)}m {s:.1f}s"


def process_path(path: str):
    """Process a single file (PDF or image) with dual-pass OCR."""
    stem = os.path.splitext(os.path.basename(path))[0]
    ensure_output_dirs(stem)

    if path.lower().endswith(".pdf"):
        for i, img in pdf_pages(path):
            page_start = time.time()
            process_page(stem, i, img)
            elapsed = time.time() - page_start
            print(f"    Page {i + 1} total: {fmt_duration(elapsed)}\n")
    else:
        page_start = time.time()
        img = Image.open(path)
        process_page(stem, 0, img)
        elapsed = time.time() - page_start
        print(f"    Total: {fmt_duration(elapsed)}\n")


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else IN_DIR
    files = []
    if os.path.isdir(target):
        for n in sorted(os.listdir(target)):
            if n.lower().endswith((".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                files.append(os.path.join(target, n))
    else:
        files = [target]

    if not files:
        print(f"No supported files found in {target}")
        return

    print(f"\nFound {len(files)} file(s) to process\n")

    job_start = time.time()
    for idx, p in enumerate(files, 1):
        file_start = time.time()
        print(f"[{idx}/{len(files)}] {os.path.basename(p)}")
        try:
            process_path(p)
        except Exception as e:
            print(f"[WARN] Failed on {p}: {e}")
        file_elapsed = time.time() - file_start
        print(f"  File total: {fmt_duration(file_elapsed)}\n")

    job_elapsed = time.time() - job_start
    print(f"Processing complete! {len(files)} file(s) in {fmt_duration(job_elapsed)}")
    print(f"Check {OUT_DIR}/<document-name>/trimmed/ for best results.")


if __name__ == "__main__":
    load_start = time.time()
    print(f"Loading DeepSeek-OCR model on {DEVICE}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    ).to(DEVICE).eval()
    load_elapsed = time.time() - load_start
    print(f"Model loaded on {DEVICE} ({fmt_duration(load_elapsed)})")
    main()
