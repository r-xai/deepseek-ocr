import os, io, glob, re
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import fitz  # PyMuPDF
from slugify import slugify
from vllm import LLM, SamplingParams

MODEL_ID = "deepseek-ai/DeepSeek-OCR"
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

IN_DIR, OUT_DIR = "in", "out"
os.makedirs(OUT_DIR, exist_ok=True)

# Optimized for DGX Spark (GB10, 6144 CUDA cores, 128GB unified memory)
llm = LLM(
    model=MODEL_ID,
    gpu_memory_utilization=0.9,  # Use 90% of GPU memory
    max_model_len=4096,  # Increase context for large images
    enable_prefix_caching=False,  # Per DeepSeek-OCR docs
    tensor_parallel_size=1,  # Single GPU
)
sp_csv = SamplingParams(max_tokens=2048, temperature=0)
sp_md  = SamplingParams(max_tokens=2048, temperature=0)

def run_ocr_on_image(img: Image.Image, as_csv=False):
    prompt = PROMPT_CSV if as_csv else PROMPT_MD
    # vLLM multimodal dict form following DeepSeek-OCR format
    out = llm.generate(
        {"images": [img.convert("RGB")], "prompt": prompt},
        sp_csv if as_csv else sp_md
    )
    return out[0].outputs[0].text.strip()

def write_output(base, page_idx, text, prefer_csv):
    stem = f"{slugify(base)}-p{page_idx+1}"
    # Detect a CSV by looking for commas and multiple newline-separated rows without pipes
    looks_csv = prefer_csv and ("," in text and "\n" in text and "|" not in text)
    if looks_csv:
        fp = os.path.join(OUT_DIR, f"{stem}.csv")
    else:
        fp = os.path.join(OUT_DIR, f"{stem}.md")
    with open(fp, "w", encoding="utf-8") as f:
        f.write(text)
    print("Wrote", fp)

def ocr_pdf(path, prefer_csv=False, dpi=250):
    doc = fitz.open(path)
    base = os.path.splitext(os.path.basename(path))[0]

    # Render all pages in parallel (CPU-bound, uses DGX Spark's 20 cores)
    print(f"  Rendering {len(doc)} pages...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        def render_page(page_info):
            i, page = page_info
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            return (i, img)

        images = list(executor.map(render_page, enumerate(doc)))

    # Process OCR sequentially on GPU
    print(f"  Running OCR on {len(images)} pages...")
    for i, img in images:
        text = run_ocr_on_image(img, as_csv=prefer_csv)
        write_output(base, i, text, prefer_csv)
        print(f"    Page {i+1}/{len(images)} complete")

def ocr_image(path, prefer_csv=False):
    base = os.path.splitext(os.path.basename(path))[0]
    img = Image.open(path).convert("RGB")
    text = run_ocr_on_image(img, as_csv=prefer_csv)
    write_output(base, 0, text, prefer_csv)

def main():
    exts = ("*.pdf","*.png","*.jpg","*.jpeg","*.tif","*.tiff")
    files = [p for e in exts for p in glob.glob(os.path.join(IN_DIR, e))]
    if not files:
        print("No inputs in ./in")
        return

    print(f"Found {len(files)} files to process")
    for idx, p in enumerate(sorted(files), 1):
        try:
            print(f"[{idx}/{len(files)}] Processing {os.path.basename(p)}...")
            if p.lower().endswith(".pdf"):
                ocr_pdf(p, prefer_csv=True)  # pages of tables often → CSV
            else:
                ocr_image(p, prefer_csv=True)
        except Exception as e:
            print(f"[WARN] Failed on {p}: {e}")
if __name__ == "__main__":
    main()
