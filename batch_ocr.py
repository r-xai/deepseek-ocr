import os, io, glob
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import fitz  # PyMuPDF
from vllm import LLM, SamplingParams

MODEL_ID = "deepseek-ai/DeepSeek-OCR"

# Start the model once; vLLM handles GPU execution efficiently
# Optimized for DGX Spark (GB10, 6144 CUDA cores, 128GB unified memory)
llm = LLM(
    model=MODEL_ID,
    gpu_memory_utilization=0.9,  # Use 90% of GPU memory
    max_model_len=4096,  # Increase context for large images
    enable_prefix_caching=False,  # Per DeepSeek-OCR docs
    tensor_parallel_size=1,  # Single GPU
)
sampler = SamplingParams(max_tokens=2048, temperature=0)

IN_DIR  = "in"
OUT_DIR = "out"
os.makedirs(OUT_DIR, exist_ok=True)

def ocr_image(pil_img):
    # vLLM multimodal: pass image with prompt following DeepSeek-OCR format
    out = llm.generate(
        {"prompt": "Convert the document to markdown.", "images": [pil_img.convert("RGB")]},
        sampler
    )
    return out[0].outputs[0].text

def ocr_pdf(path):
    doc = fitz.open(path)
    images = []

    # First pass: render all pages to images in parallel (CPU-bound, uses DGX Spark's 20 cores)
    print(f"  Rendering {len(doc)} pages...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        def render_page(page_info):
            i, page = page_info
            pix = page.get_pixmap(dpi=250, alpha=False)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            return (i, img)

        images = list(executor.map(render_page, enumerate(doc)))

    # Second pass: batch OCR processing for GPU efficiency
    print(f"  Running OCR on {len(images)} pages...")
    texts = []
    for i, img in images:
        txt = ocr_image(img)
        texts.append(f"## Page {i+1}\n\n{txt}\n")
        print(f"    Page {i+1}/{len(images)} complete")

    return "\n".join(texts)

def ocr_path(path):
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    out_md = os.path.join(OUT_DIR, f"{stem}.md")
    if path.lower().endswith(".pdf"):
        text = ocr_pdf(path)
    else:
        img = Image.open(path)
        text = ocr_image(img)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Wrote {out_md}")

def main():
    exts = ("*.pdf","*.png","*.jpg","*.jpeg","*.tif","*.tiff")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(IN_DIR, e)))
    if not files:
        print("No input files found in ./in")
        return

    print(f"Found {len(files)} files to process")
    for idx, p in enumerate(sorted(files), 1):
        try:
            print(f"[{idx}/{len(files)}] Processing {os.path.basename(p)}...")
            ocr_path(p)
        except Exception as e:
            print(f"[WARN] Failed on {p}: {e}")

if __name__ == "__main__":
    main()
