#!/usr/bin/env python3
"""
DeepSeek-OCR batch processor using Transformers (instead of vLLM).
Works on Blackwell GB10 GPU (SM 12.1) without Triton/ptxas issues.

Usage:
    python3 run_deepseek_transformers.py <file-or-folder>
    python3 run_deepseek_transformers.py in/

Requires:
    pip install transformers==4.46.3 tokenizers==0.20.3 einops addict easydict accelerate pillow pymupdf
"""

import os, io, sys, tempfile
from PIL import Image
import fitz  # PyMuPDF
import torch
from transformers import AutoProcessor, AutoModel

MODEL_ID = "deepseek-ai/DeepSeek-OCR"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IN_DIR, OUT_DIR = "in", "out"
os.makedirs(OUT_DIR, exist_ok=True)

PROMPT_CSV = (
    "Transcribe ONLY the table as CSV.\n"
    "- First row = header names\n"
    "- One row per record\n"
    "- No commentary, no code fences"
)
PROMPT_MD = (
    "Convert the page to Markdown.\n"
    "- Use proper headings and paragraphs\n"
    "- Render tables as Markdown pipe tables with a header row\n"
    "- No extra commentary"
)

def infer_on_image(img: Image.Image, csv_preference=True, max_new_tokens=2048):
    """Run OCR inference on a single image using model's infer() method."""
    prompt = PROMPT_CSV if csv_preference else PROMPT_MD

    # Save image to temp file (model.infer expects file path)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img.convert("RGB").save(tmp.name, "PNG")
        tmp_path = tmp.name

    try:
        # Use model's native infer() method
        result = model.infer(
            tokenizer=processor.tokenizer,
            prompt=prompt,
            image_file=tmp_path,
            output_path="",  # Don't save intermediate files
            base_size=1024,
            image_size=640,
            crop_mode=True,
            save_results=False
        )
        return result.strip() if isinstance(result, str) else str(result).strip()
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def write_output(stem: str, page_idx: int, text: str, prefer_csv: bool):
    """Write OCR output to file, auto-detecting CSV vs Markdown."""
    looks_csv = prefer_csv and ("," in text and "\n" in text and "|" not in text)
    ext = ".csv" if looks_csv else ".md"
    out_path = os.path.join(OUT_DIR, f"{stem}-p{page_idx+1}{ext}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  Wrote {out_path}")

def pdf_pages(path, dpi=250):
    """Generator that yields (page_index, PIL.Image) for each PDF page."""
    doc = fitz.open(path)
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        yield i, Image.open(io.BytesIO(pix.tobytes("png")))

def process_path(path: str):
    """Process a single file (PDF or image)."""
    stem = os.path.splitext(os.path.basename(path))[0]
    if path.lower().endswith(".pdf"):
        for i, img in pdf_pages(path):
            print(f"  Page {i+1}...")
            text = infer_on_image(img, csv_preference=True)
            write_output(stem, i, text, prefer_csv=True)
    else:
        img = Image.open(path)
        text = infer_on_image(img, csv_preference=True)
        write_output(stem, 0, text, prefer_csv=True)

def main():
    """Main entry point."""
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

    for idx, p in enumerate(files, 1):
        print(f"[{idx}/{len(files)}] {os.path.basename(p)}")
        try:
            process_path(p)
        except Exception as e:
            print(f"[WARN] Failed on {p}: {e}")

    print(f"\nProcessing complete! Check {OUT_DIR}/ for results.")

if __name__ == "__main__":
    print(f"Loading DeepSeek-OCR model on {DEVICE}...")
    # Load processor + model (custom classes via trust_remote_code)
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    ).to(DEVICE).eval()
    print(f"Model loaded successfully on {DEVICE}")
    main()
