#!/usr/bin/env python3
"""
Design-to-Code Fidelity Playground — Backend
=============================================
FastAPI server exposing:
  GET  /                   → serves playground UI
  POST /api/evaluate       → accepts reference + implementation, returns job_id
  GET  /api/stream/{id}    → SSE stream of real-time progress + results
  GET  /api/screenshots/*  → serves generated PNGs
"""

import asyncio
import base64
import json
import os
import re
import tempfile
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# ── Config ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
SS_DIR   = BASE_DIR / "screenshots"
SS_DIR.mkdir(exist_ok=True)

API_BASE  = os.environ.get("API_BASE", "http://10.69.141.113:8023")
VLM_MODEL = "gpt-4o"
VIEWPORT  = {"width": 1280, "height": 900}

# In-memory job store  {job_id: asyncio.Queue}
_queues: dict[str, asyncio.Queue] = {}

SAMPLES_DIR = BASE_DIR / "samples"

app = FastAPI(title="Design Fidelity Playground")

# ── Static files ─────────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/screenshots", StaticFiles(directory=str(SS_DIR)), name="screenshots")


@app.get("/", response_class=HTMLResponse)
async def root():
    index = BASE_DIR / "static" / "index.html"
    return HTMLResponse(index.read_text())


# ── Sample library endpoints ──────────────────────────────────────────────────
SAMPLE_META = {
    "login": {
        "title":       "Login Page",
        "description": "Indigo gradient card with social login",
        "icon":        "🔐",
        "variants":    {"reference": "reference.html", "close": "v1_close.html", "degraded": "v2_degraded.html"},
        "dir":         "login",
    },
    "pricing": {
        "title":       "Pricing Cards",
        "description": "3-tier dark glassmorphism pricing",
        "icon":        "💳",
        "variants":    {"reference": "reference.html", "close": "v1_close.html", "degraded": "v2_degraded.html"},
        "dir":         "pricing",
    },
    "signup": {
        "title":       "Sign-Up Form",
        "description": "Split hero + multi-field registration",
        "icon":        "📝",
        "variants":    {"reference": "reference.html", "close": "v1_close.html", "degraded": "v2_degraded.html"},
        "dir":         "signup",
    },
}

# Symlink login sample from POC directory if not already in samples/
_login_src = BASE_DIR.parent / "poc"
_login_dst = SAMPLES_DIR / "login"
if not _login_dst.exists() and _login_src.exists():
    import shutil
    _login_dst.mkdir(parents=True, exist_ok=True)
    for name, fname in [("reference.html", "reference.html"),
                         ("v1_close.html",  "v1_close.html"),
                         ("v2_degraded.html","v2_degraded.html")]:
        src = _login_src / ("design" if name == "reference.html" else "implementations") / fname
        if src.exists():
            shutil.copy(src, _login_dst / name)


@app.get("/api/samples")
async def list_samples():
    return list(SAMPLE_META.values())


@app.get("/api/samples/{sample_id}/{variant}")
async def get_sample(sample_id: str, variant: str):
    meta = SAMPLE_META.get(sample_id)
    if not meta or variant not in meta["variants"]:
        from fastapi import HTTPException
        raise HTTPException(404, "Sample not found")
    path = SAMPLES_DIR / meta["dir"] / meta["variants"][variant]
    if not path.exists():
        from fastapi import HTTPException
        raise HTTPException(404, f"File not found: {path}")
    return {"html": path.read_text(), "name": f"{meta['title']} — {variant}"}


# ── Evaluate endpoint ─────────────────────────────────────────────────────────
@app.post("/api/evaluate")
async def evaluate(
    ref_type:  str           = Form(...),   # "upload" | "html" | "url"
    impl_type: str           = Form(...),   # "url"    | "html"
    ref_image: Optional[UploadFile] = File(None),
    ref_html:  Optional[str] = Form(None),
    ref_url:   Optional[str] = Form(None),
    impl_url:  Optional[str] = Form(None),
    impl_html: Optional[str] = Form(None),
):
    job_id = str(uuid.uuid4())
    q: asyncio.Queue = asyncio.Queue()
    _queues[job_id] = q

    # Save uploaded image if provided
    ref_image_bytes: Optional[bytes] = None
    if ref_image:
        ref_image_bytes = await ref_image.read()

    asyncio.create_task(
        _run_evaluation(
            job_id, q,
            ref_type, ref_image_bytes, ref_html, ref_url,
            impl_type, impl_url, impl_html,
        )
    )
    return {"job_id": job_id}


# ── SSE stream endpoint ───────────────────────────────────────────────────────
@app.get("/api/stream/{job_id}")
async def stream(job_id: str):
    q = _queues.get(job_id)
    if not q:
        return HTMLResponse("Job not found", status_code=404)

    async def event_gen():
        while True:
            msg = await q.get()
            yield f"data: {json.dumps(msg)}\n\n"
            if msg.get("type") in ("done", "error"):
                _queues.pop(job_id, None)
                break

    return StreamingResponse(event_gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


# ── Core evaluation logic (runs in background task) ───────────────────────────
async def _run_evaluation(
    job_id:          str,
    q:               asyncio.Queue,
    ref_type:        str,
    ref_image_bytes: Optional[bytes],
    ref_html:        Optional[str],
    ref_url:         Optional[str],
    impl_type:       str,
    impl_url:        Optional[str],
    impl_html:       Optional[str],
):
    async def emit(step: str, pct: int, msg: str, data: dict = None):
        payload = {"type": "progress", "step": step, "pct": pct, "msg": msg}
        if data:
            payload.update(data)
        await q.put(payload)

    try:
        # ── Step 1: Prepare reference ────────────────────────────────────────
        await emit("reference", 5, "Preparing reference design …")
        ref_ss = SS_DIR / f"{job_id}_ref.png"

        loop = asyncio.get_event_loop()

        if ref_type == "upload" and ref_image_bytes:
            # Save uploaded image directly as reference screenshot
            img = Image.open(__import__("io").BytesIO(ref_image_bytes)).convert("RGB")
            # Crop/resize to viewport width if too large; keep proportional
            if img.width > VIEWPORT["width"]:
                ratio = VIEWPORT["width"] / img.width
                img = img.resize((VIEWPORT["width"], int(img.height * ratio)), Image.LANCZOS)
            img.save(str(ref_ss), "PNG")
        elif ref_type == "html" and ref_html:
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
                f.write(ref_html)
                tmp_path = Path(f.name)
            await loop.run_in_executor(None, _screenshot_file, tmp_path, ref_ss)
            tmp_path.unlink(missing_ok=True)
        elif ref_type == "url" and ref_url:
            await loop.run_in_executor(None, _screenshot_url, ref_url, ref_ss)
        else:
            await q.put({"type": "error", "msg": "Invalid reference input"})
            return

        await emit("reference", 20, "Reference captured ✓", {"ref_img": _img_url(ref_ss)})

        # ── Step 2: Prepare implementation ───────────────────────────────────
        await emit("impl", 25, "Rendering implementation …")
        impl_ss = SS_DIR / f"{job_id}_impl.png"

        if impl_type == "url" and impl_url:
            await loop.run_in_executor(None, _screenshot_url, impl_url, impl_ss)
        elif impl_type == "html" and impl_html:
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
                f.write(impl_html)
                tmp_path = Path(f.name)
            await loop.run_in_executor(None, _screenshot_file, tmp_path, impl_ss)
            tmp_path.unlink(missing_ok=True)
        else:
            await q.put({"type": "error", "msg": "Invalid implementation input"})
            return

        await emit("impl", 45, "Implementation rendered ✓", {"impl_img": _img_url(impl_ss)})

        # ── Step 3: Pixel metrics ─────────────────────────────────────────────
        await emit("metrics", 50, "Computing pixel & SSIM metrics …")
        px = await loop.run_in_executor(None, _pixel_metrics, ref_ss, impl_ss)
        await emit("metrics", 65, "Pixel metrics done ✓", {"pixel": px})

        # ── Step 4: VLM judge ─────────────────────────────────────────────────
        await emit("vlm", 70, f"Asking VLM judge ({VLM_MODEL}) …")
        vlm = await loop.run_in_executor(None, _vlm_judge, ref_ss, impl_ss)
        await emit("vlm", 88, "VLM scoring done ✓", {"vlm": vlm})

        # ── Step 5: Composite score ───────────────────────────────────────────
        await emit("composite", 95, "Computing composite score …")
        total_vlm = vlm.get("total", 0)
        composite = round(
            0.40 * total_vlm
            + 0.35 * px["ssim"] * 100
            + 0.25 * px["pixel_match_20"],
            1,
        )
        grade = _grade(composite)

        await q.put({
            "type":      "done",
            "composite": composite,
            "grade":     grade,
            "pixel":     px,
            "vlm":       vlm,
            "ref_img":   _img_url(ref_ss),
            "impl_img":  _img_url(impl_ss),
        })

    except Exception as exc:
        import traceback
        await q.put({"type": "error", "msg": str(exc), "trace": traceback.format_exc()})


# ── Helpers ───────────────────────────────────────────────────────────────────
def _img_url(path: Path) -> str:
    return f"/screenshots/{path.name}"


def _screenshot_url(url: str, out: Path):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page    = browser.new_page(viewport=VIEWPORT)
        page.goto(url, wait_until="networkidle", timeout=30000)
        page.wait_for_timeout(500)
        page.screenshot(path=str(out), full_page=False)
        browser.close()


def _screenshot_file(html_path: Path, out: Path):
    from playwright.sync_api import sync_playwright
    url = f"file://{html_path.resolve()}"
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page    = browser.new_page(viewport=VIEWPORT)
        page.goto(url, wait_until="networkidle")
        page.wait_for_timeout(300)
        page.screenshot(path=str(out), full_page=False)
        browser.close()


def _load_arr(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def _pixel_metrics(ref_p: Path, cmp_p: Path) -> dict:
    ref = _load_arr(ref_p)
    cmp = _load_arr(cmp_p)
    if ref.shape != cmp.shape:
        img = Image.fromarray(cmp).resize((ref.shape[1], ref.shape[0]), Image.LANCZOS)
        cmp = np.array(img)
    s    = ssim(ref, cmp, channel_axis=2, data_range=255)
    diff = np.abs(ref.astype(np.int32) - cmp.astype(np.int32))
    mx   = diff.max(axis=2)
    tot  = mx.size
    return {
        "ssim":           round(float(s), 4),
        "pixel_match_5":  round(float((mx <= 5).sum())  / tot * 100, 2),
        "pixel_match_20": round(float((mx <= 20).sum()) / tot * 100, 2),
        "mse":            round(float(np.mean(diff ** 2)), 2),
    }


RUBRIC_PROMPT = """You are a UI/UX design quality evaluator. You are given two screenshots:
  IMAGE 1: The REFERENCE DESIGN (the golden standard).
  IMAGE 2: A CODE IMPLEMENTATION to evaluate against the reference.

Score the implementation on EACH of the 5 dimensions below from 0-20 (total = 100):

1. **Layout & Structure** (0-20): Card/container size, centering, element order, overall composition.
2. **Typography** (0-20): Heading sizes, font weights, label styles, text hierarchy, placeholder text.
3. **Color & Visual Style** (0-20): Background, button color, brand elements, borders, link colors.
4. **Component Fidelity** (0-20): Presence and styling of all expected UI elements.
5. **Spacing & Alignment** (0-20): Padding, margins, gaps, input/button heights.

Respond ONLY with valid JSON — no extra text:
{
  "layout_structure":   <0-20>,
  "typography":         <0-20>,
  "color_visual_style": <0-20>,
  "component_fidelity": <0-20>,
  "spacing_alignment":  <0-20>,
  "total":              <0-100>,
  "summary":            "<2-3 sentence verdict on the biggest wins and gaps>"
}"""


def _encode(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def _vlm_judge(ref_p: Path, cmp_p: Path) -> dict:
    messages = [{
        "role": "user",
        "content": [
            {"type": "text",      "text": RUBRIC_PROMPT},
            {"type": "text",      "text": "IMAGE 1 — REFERENCE DESIGN:"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_encode(ref_p)}"}},
            {"type": "text",      "text": "IMAGE 2 — IMPLEMENTATION TO EVALUATE:"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_encode(cmp_p)}"}},
        ],
    }]
    payload = json.dumps({"model": VLM_MODEL, "messages": messages, "max_tokens": 800}).encode()
    req = urllib.request.Request(
        f"{API_BASE}/v1/chat/completions", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = json.loads(r.read())["choices"][0]["message"]["content"].strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        return json.loads(m.group()) if m else {"error": raw, "total": 0}


def _grade(s: float) -> dict:
    if s >= 90: return {"label": "Excellent",  "color": "#16a34a", "letter": "A"}
    if s >= 75: return {"label": "Good",        "color": "#2563eb", "letter": "B"}
    if s >= 60: return {"label": "Acceptable",  "color": "#d97706", "letter": "C"}
    if s >= 40: return {"label": "Needs Work",  "color": "#ea580c", "letter": "D"}
    return              {"label": "Major Drift", "color": "#dc2626", "letter": "F"}


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=7860, reload=False)
