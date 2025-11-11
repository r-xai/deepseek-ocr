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

from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import torch
import fitz  # PyMuPDF
import io
import sys
import os
from pathlib import Path

MODEL_ID = "deepseek-ai/DeepSeek-OCR"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading DeepSeek-OCR model on {DEVICE}...")
# Use AutoProcessor (handles text+image) instead of AutoTokenizer
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32
).to(DEVICE).eval()
print(f"Model loaded successfully on {DEVICE}")

def ocr_image(img: Image.Image, max_new_tokens=2048, csv=False):
    """Run OCR on a single image."""
    prompt = None
    if csv:
        prompt = (
            "Transcribe ONLY the table as CSV.\n"
            "- First row = header names\n"
            "- One row per record\n"
            "- No commentary, no code fences"
        )
    else:
        prompt = "Convert the document to markdown."

    inputs = processor(text=prompt, images=img.convert("RGB"), return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)

    return processor.batch_decode(out, skip_special_tokens=True)[0].strip()

def pdf_to_pages(path, dpi=250):
    """Generator that yields (page_index, PIL.Image) for each PDF page."""
    doc = fitz.open(path)

    # Parallel rendering using ThreadPoolExecutor (leverages DGX Spark's 20 cores)
    def render_page(page_info):
        i, page = page_info
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return (i, img)

    with ThreadPoolExecutor(max_workers=10) as executor:
        yield from executor.map(render_page, enumerate(doc))

def write_output(filepath, name, page_idx, text, csv=False):
    """Write OCR output to file, auto-detecting CSV vs Markdown."""
    stem = os.path.splitext(os.path.basename(name))[0]

    # Auto-detect CSV format
    looks_csv = csv and ("," in text and "\n" in text and "|" not in text)
    ext = ".csv" if looks_csv else ".md"

    output_path = os.path.join("out", f"{stem}-p{page_idx+1}{ext}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"  Wrote {output_path}")

def process_file(filepath, csv_mode=True):
    """Process a single file (PDF or image)."""
    print(f"Processing {os.path.basename(filepath)}...")

    if filepath.lower().endswith(".pdf"):
        # Process PDF page by page
        for i, img in pdf_to_pages(filepath):
            print(f"  Page {i+1}...")
            text = ocr_image(img, csv=csv_mode)
            write_output(filepath, filepath, i, text, csv=csv_mode)
    else:
        # Process single image
        img = Image.open(filepath)
        text = ocr_image(img, csv=csv_mode)
        write_output(filepath, filepath, 0, text, csv=csv_mode)

def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python3 run_deepseek_transformers.py <file-or-folder>")
        print("Example: python3 run_deepseek_transformers.py in/")
        return

    target = sys.argv[1]
    in_dir = "in"
    out_dir = "out"

    # Create output directory
    os.makedirs(out_dir, exist_ok=True)

    # Collect files to process
    files = []
    if os.path.isdir(target):
        supported_exts = (".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff")
        for filename in sorted(os.listdir(target)):
            if filename.lower().endswith(supported_exts):
                files.append(os.path.join(target, filename))
    else:
        files = [target]

    if not files:
        print(f"No supported files found in {target}")
        return

    print(f"\nFound {len(files)} file(s) to process\n")

    # Process each file
    for idx, filepath in enumerate(files, 1):
        try:
            print(f"[{idx}/{len(files)}] {os.path.basename(filepath)}")
            process_file(filepath, csv_mode=True)
        except Exception as e:
            print(f"[WARN] Failed on {filepath}: {e}")

    print(f"\nProcessing complete! Check {out_dir}/ for results.")

if __name__ == "__main__":
    main()
