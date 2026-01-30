import os

# Disable torch.compile to avoid Triton/PTX paths unsupported on GB10 (SM 12.1)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import glob
import io
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator, Tuple

import fitz  # PyMuPDF
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

MODEL_ID = "deepseek-ai/DeepSeek-OCR"
PROMPT = "Convert the document to markdown."
PDF_DPI = 250
PDF_RENDER_WORKERS = min(10, (os.cpu_count() or 2))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IN_DIR = "in"
OUT_DIR = "out"
os.makedirs(OUT_DIR, exist_ok=True)

_processor = None
_model = None


def load_model() -> Tuple[AutoProcessor, AutoModel]:
    """Load the DeepSeek-OCR processor + model lazily."""
    global _processor, _model
    if _processor is None or _model is None:
        print(f"Loading DeepSeek-OCR ({MODEL_ID}) on {DEVICE}...")
        _processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
        _model = (
            AutoModel.from_pretrained(
                MODEL_ID,
                trust_remote_code=True,
                torch_dtype=dtype,
            )
            .to(DEVICE)
            .eval()
        )
        print("Model ready.")
    return _processor, _model


def pdf_pages(path: str, dpi: int = PDF_DPI) -> Iterator[Tuple[int, Image.Image]]:
    """Yield (page_index, PIL.Image) for each PDF page with limited memory use."""
    with fitz.open(path) as doc:
        def render(page_info):
            idx, page = page_info
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            with Image.open(io.BytesIO(pix.tobytes("png"))) as pil_img:
                return idx, pil_img.convert("RGB")

        with ThreadPoolExecutor(max_workers=PDF_RENDER_WORKERS) as executor:
            for idx, img in executor.map(render, enumerate(doc)):
                yield idx, img


def ocr_image(pil_img: Image.Image) -> str:
    """Run OCR on a single image and return markdown text."""
    processor, model = load_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_img = os.path.join(tmpdir, "page.png")
        pil_img.convert("RGB").save(tmp_img, "PNG")

        result = model.infer(
            tokenizer=processor,
            prompt=PROMPT,
            image_file=tmp_img,
            output_path=tmpdir,
            base_size=1024,
            image_size=640,
            crop_mode=True,
            save_results=False,
        )

    return result.strip() if isinstance(result, str) else str(result).strip()


def ocr_pdf(path: str) -> str:
    """Process each PDF page sequentially and stitch markdown output."""
    sections = []
    for idx, img in pdf_pages(path):
        print(f"    Page {idx + 1}...")
        try:
            text = ocr_image(img)
        finally:
            img.close()
        sections.append(f"## Page {idx + 1}\n\n{text}\n")
    return "\n".join(sections)


def ocr_path(path: str) -> None:
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    out_md = os.path.join(OUT_DIR, f"{stem}.md")

    if path.lower().endswith(".pdf"):
        text = ocr_pdf(path)
    else:
        with Image.open(path) as img:
            text = ocr_image(img)

    with open(out_md, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Wrote {out_md}")


def main():
    exts = ("*.pdf", "*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
    files = []
    for pattern in exts:
        files.extend(glob.glob(os.path.join(IN_DIR, pattern)))
    if not files:
        print("No input files found in ./in")
        return

    print(f"Found {len(files)} files to process")
    for idx, path in enumerate(sorted(files), 1):
        print(f"[{idx}/{len(files)}] Processing {os.path.basename(path)}...")
        try:
            ocr_path(path)
        except Exception as exc:
            print(f"[WARN] Failed on {path}: {exc}")


if __name__ == "__main__":
    main()
