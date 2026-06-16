#!/usr/bin/env python3
"""
Design-to-Code Fidelity Playground — Backend (Auth-Aware v3)
=============================================================
New in v2:
  • Figma REST API screenshots  (PAT token  → export PNG via API)
  • Web form auth agent         (VLM detects login form → Playwright fills it)
  • Cookie / Basic-auth support (inject session cookies or HTTP credentials)
  • Demo protected app          (/demo/login  /demo/dashboard) for testing

Endpoints:
  GET  /                       → playground UI
  POST /api/evaluate           → submit job, returns job_id
  GET  /api/stream/{id}        → SSE real-time progress
  GET  /api/samples            → sample library list
  GET  /api/samples/{id}/{v}   → fetch sample HTML
  GET  /demo/login             → demo protected app – login page
  POST /demo/auth              → demo protected app – validates creds + sets cookie
  GET  /demo/dashboard         → demo protected app – requires auth
  GET  /screenshots/*          → serve generated PNGs
"""

import asyncio
import base64
import io
import json
import os
import re
import tempfile
import time
import urllib.request
import urllib.parse
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
SS_DIR      = BASE_DIR / "screenshots"
SAMPLES_DIR = BASE_DIR / "samples"
SS_DIR.mkdir(exist_ok=True)

API_BASE  = os.environ.get("API_BASE", "http://10.69.141.113:8023")
VLM_MODEL = "gpt-4o"
VIEWPORT  = {"width": 1280, "height": 900}

# Platform viewports
VIEWPORTS = {
    "desktop":        {"width": 1280, "height": 900},
    "android_phone":  {"width": 390,  "height": 844},
    "android_tablet": {"width": 768,  "height": 1024},
    "windows_hd":     {"width": 1366, "height": 768},
    "windows_fhd":    {"width": 1920, "height": 1080},
}

def _platform_type(platform: str) -> str:
    """Return 'mobile', 'windows', or 'desktop' rubric category."""
    if platform in ("android_phone", "android_tablet"):
        return "mobile"
    if platform in ("windows_hd", "windows_fhd"):
        return "windows"
    return "desktop"

# Windows user-agent for Playwright (makes responsive sites render in desktop mode)
WINDOWS_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# ADB binary (installed on remote server)
ADB_PATH = os.path.expanduser("~/platform-tools/adb")

# Demo app credentials (for auth testing)
DEMO_USER = "demo"
DEMO_PASS = "demo123"
DEMO_COOKIE_NAME = "playground_session"
DEMO_COOKIE_VALUE = "authenticated"

_queues: dict[str, asyncio.Queue] = {}

# ── Safety limits ──────────────────────────────────────────────────────────────
MAX_UPLOAD_BYTES  = 10 * 1024 * 1024   # 10 MB per file
MAX_CONCURRENT    = 6                  # max simultaneous Playwright instances
SS_MAX_AGE_HOURS  = 2                  # screenshots older than this are deleted
_job_semaphore    = asyncio.Semaphore(MAX_CONCURRENT)

# ── ADB serial validation (alphanumeric, colon, dot, hyphen only) ──────────────
_ADB_SERIAL_RE = re.compile(r'^[A-Za-z0-9:.\-]+$')

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(application):
    n = _cleanup_old_screenshots()
    if n:
        print(f"[startup] cleaned up {n} old screenshots")
    yield   # server runs here

app = FastAPI(title="Design Fidelity Playground v3", lifespan=lifespan)

app.mount("/static",      StaticFiles(directory=str(BASE_DIR / "static")),  name="static")
app.mount("/screenshots", StaticFiles(directory=str(SS_DIR)),               name="screenshots")


def _cleanup_old_screenshots():
    """Delete screenshots older than SS_MAX_AGE_HOURS to prevent disk bloat."""
    cutoff = time.time() - SS_MAX_AGE_HOURS * 3600
    removed = 0
    for f in SS_DIR.glob("*.png"):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
                removed += 1
        except Exception:
            pass
    return removed




# ══════════════════════════════════════════════════════════════════════════════
# DEMO PROTECTED APP  (for testing the auth agent)
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/demo/login", response_class=HTMLResponse)
async def demo_login_page(error: str = ""):
    err_html = f'<div class="error">{error}</div>' if error else ""
    return HTMLResponse(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>Demo App — Sign in</title>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
body{{min-height:100vh;display:flex;align-items:center;justify-content:center;
  background:linear-gradient(135deg,#1a1a2e,#16213e,#0f3460);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;}}
.card{{width:400px;background:#fff;border-radius:14px;padding:40px;
  box-shadow:0 20px 60px rgba(0,0,0,.35);}}
.brand{{display:flex;align-items:center;gap:10px;margin-bottom:28px;}}
.brand-icon{{width:36px;height:36px;background:linear-gradient(135deg,#4f46e5,#7c3aed);
  border-radius:9px;display:flex;align-items:center;justify-content:center;}}
.brand-name{{font-size:18px;font-weight:700;color:#111827;}}
h1{{font-size:22px;font-weight:700;color:#111827;margin-bottom:4px;}}
.sub{{font-size:13px;color:#6b7280;margin-bottom:28px;}}
.error{{background:#fef2f2;border:1px solid #fecaca;color:#dc2626;
  padding:10px 14px;border-radius:8px;font-size:13px;margin-bottom:16px;}}
.field{{margin-bottom:18px;}}
label{{display:block;font-size:13px;font-weight:600;color:#374151;margin-bottom:6px;}}
input[type=text],input[type=password]{{width:100%;height:42px;padding:0 12px;
  border:1.5px solid #e5e7eb;border-radius:8px;font-size:14px;color:#111827;outline:none;}}
input:focus{{border-color:#4f46e5;box-shadow:0 0 0 3px rgba(79,70,229,.1);}}
.hint{{font-size:11px;color:#9ca3af;margin-top:18px;text-align:center;}}
.btn{{width:100%;height:44px;background:linear-gradient(135deg,#4f46e5,#7c3aed);
  color:#fff;font-size:14px;font-weight:700;border:none;border-radius:8px;cursor:pointer;}}
</style>
</head>
<body>
<div class="card">
  <div class="brand">
    <div class="brand-icon">
      <svg width="18" height="18" fill="none" viewBox="0 0 24 24">
        <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"
              stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
    </div>
    <span class="brand-name">Synapse Demo</span>
  </div>
  <h1>Sign in</h1>
  <p class="sub">Access the protected dashboard</p>
  {err_html}
  <form method="POST" action="/demo/auth">
    <div class="field">
      <label for="username">Username</label>
      <input type="text" id="username" name="username" placeholder="demo" autocomplete="username"/>
    </div>
    <div class="field">
      <label for="password">Password</label>
      <input type="password" id="password" name="password" placeholder="demo123" autocomplete="current-password"/>
    </div>
    <button type="submit" class="btn">Sign in</button>
  </form>
  <p class="hint">Use <strong>demo / demo123</strong> to sign in</p>
</div>
</body>
</html>""")


@app.post("/demo/auth")
async def demo_auth(request: Request):
    form = await request.form()
    username = form.get("username", "")
    password = form.get("password", "")
    if username == DEMO_USER and password == DEMO_PASS:
        response = RedirectResponse("/demo/dashboard", status_code=302)
        response.set_cookie(DEMO_COOKIE_NAME, DEMO_COOKIE_VALUE,
                            httponly=True, samesite="lax")
        return response
    return RedirectResponse("/demo/login?error=Invalid+credentials", status_code=302)


@app.get("/demo/dashboard", response_class=HTMLResponse)
async def demo_dashboard(request: Request):
    if request.cookies.get(DEMO_COOKIE_NAME) != DEMO_COOKIE_VALUE:
        return RedirectResponse("/demo/login", status_code=302)
    return HTMLResponse("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>Demo Dashboard — Protected</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{background:#0f172a;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  color:#e2e8f0;min-height:100vh;}
header{background:#1e293b;border-bottom:1px solid #334155;padding:16px 32px;
  display:flex;align-items:center;justify-content:space-between;}
.brand{display:flex;align-items:center;gap:10px;font-size:16px;font-weight:700;}
.brand-icon{width:30px;height:30px;background:linear-gradient(135deg,#4f46e5,#7c3aed);
  border-radius:8px;display:flex;align-items:center;justify-content:center;}
.badge{background:rgba(34,197,94,.15);border:1px solid rgba(34,197,94,.3);
  color:#4ade80;font-size:11px;font-weight:700;padding:3px 10px;border-radius:20px;}
.main{padding:32px;}
h1{font-size:24px;font-weight:800;color:#f1f5f9;margin-bottom:8px;}
.sub{font-size:14px;color:#64748b;margin-bottom:32px;}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:32px;}
.stat{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:20px;}
.stat-val{font-size:32px;font-weight:800;color:#f1f5f9;margin-bottom:4px;}
.stat-lbl{font-size:12px;color:#64748b;}
.stat-trend{font-size:11px;color:#4ade80;margin-top:4px;}
.section{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:24px;}
.section-title{font-size:14px;font-weight:700;color:#94a3b8;
  text-transform:uppercase;letter-spacing:.8px;margin-bottom:16px;}
.row{display:flex;align-items:center;padding:10px 0;
  border-bottom:1px solid #1e293b;gap:12px;font-size:14px;}
.dot{width:8px;height:8px;border-radius:50%;flex-shrink:0;}
</style>
</head>
<body>
<header>
  <div class="brand">
    <div class="brand-icon">
      <svg width="16" height="16" fill="none" viewBox="0 0 24 24">
        <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"
              stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
    </div>
    Synapse Dashboard
  </div>
  <span class="badge">Authenticated ✓</span>
</header>
<div class="main">
  <h1>Welcome back, Demo User</h1>
  <p class="sub">This is a protected page — only visible after authentication.</p>
  <div class="stats">
    <div class="stat"><div class="stat-val">2,847</div><div class="stat-lbl">Total evaluations</div><div class="stat-trend">↑ 12% this week</div></div>
    <div class="stat"><div class="stat-val">91.4</div><div class="stat-lbl">Avg fidelity score</div><div class="stat-trend">↑ 3.2 pts</div></div>
    <div class="stat"><div class="stat-val">38</div><div class="stat-lbl">Components tracked</div><div class="stat-trend">↑ 4 new</div></div>
    <div class="stat"><div class="stat-val">99.8%</div><div class="stat-lbl">Uptime</div><div class="stat-trend">Last 30 days</div></div>
  </div>
  <div class="section">
    <div class="section-title">Recent evaluations</div>
    <div class="row"><div class="dot" style="background:#22c55e"></div><span>Login Page v3.2</span><span style="margin-left:auto;color:#4ade80">94/100</span></div>
    <div class="row"><div class="dot" style="background:#22c55e"></div><span>Pricing Cards v1.8</span><span style="margin-left:auto;color:#4ade80">89/100</span></div>
    <div class="row"><div class="dot" style="background:#f59e0b"></div><span>Checkout Flow v2.1</span><span style="margin-left:auto;color:#f59e0b">72/100</span></div>
    <div class="row"><div class="dot" style="background:#ef4444"></div><span>Settings Page v0.9</span><span style="margin-left:auto;color:#ef4444">41/100</span></div>
  </div>
</div>
</body>
</html>""")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PAGE
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse((BASE_DIR / "static" / "index.html").read_text())


# ══════════════════════════════════════════════════════════════════════════════
# SAMPLE LIBRARY
# ══════════════════════════════════════════════════════════════════════════════

SAMPLE_META = {
    "login":   {"title": "Login Page",      "description": "Indigo gradient card with social login",   "icon": "🔐", "dir": "login",   "variants": {"reference": "reference.html", "close": "v1_close.html", "degraded": "v2_degraded.html"}},
    "pricing": {"title": "Pricing Cards",   "description": "3-tier dark glassmorphism pricing",         "icon": "💳", "dir": "pricing", "variants": {"reference": "reference.html", "close": "v1_close.html", "degraded": "v2_degraded.html"}},
    "signup":  {"title": "Sign-Up Form",    "description": "Split hero + multi-field registration",     "icon": "📝", "dir": "signup",  "variants": {"reference": "reference.html", "close": "v1_close.html", "degraded": "v2_degraded.html"}},
}

_login_src = BASE_DIR.parent / "poc"
_login_dst = SAMPLES_DIR / "login"
if not _login_dst.exists() and _login_src.exists():
    import shutil
    _login_dst.mkdir(parents=True, exist_ok=True)
    for name, fname in [("reference.html","reference.html"),("v1_close.html","v1_close.html"),("v2_degraded.html","v2_degraded.html")]:
        src = _login_src / ("design" if name=="reference.html" else "implementations") / fname
        if src.exists(): shutil.copy(src, _login_dst / name)


@app.get("/api/samples")
async def list_samples():
    return list(SAMPLE_META.values())


@app.get("/api/devices")
async def list_devices():
    """Return connected ADB devices (requires ADB installed at ADB_PATH)."""
    import subprocess
    try:
        result = subprocess.run(
            [ADB_PATH, "devices"], capture_output=True, text=True, timeout=10
        )
        lines = result.stdout.strip().split("\n")[1:]  # skip "List of devices" header
        devices = []
        for line in lines:
            line = line.strip()
            if line and "\t" in line:
                device_id, status = line.split("\t", 1)
                devices.append({"id": device_id.strip(), "status": status.strip()})
        return {"devices": devices}
    except FileNotFoundError:
        return {"devices": [], "error": f"ADB not found at {ADB_PATH}"}
    except Exception as e:
        return {"devices": [], "error": str(e)}


@app.get("/api/samples/{sample_id}/{variant}")
async def get_sample(sample_id: str, variant: str):
    from fastapi import HTTPException
    meta = SAMPLE_META.get(sample_id)
    if not meta or variant not in meta["variants"]:
        raise HTTPException(404, "Sample not found")
    path = SAMPLES_DIR / meta["dir"] / meta["variants"][variant]
    if not path.exists():
        raise HTTPException(404, f"File not found: {path}")
    return {"html": path.read_text(), "name": f"{meta['title']} — {variant}"}


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATE ENDPOINT  (auth-aware v2)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/evaluate")
async def evaluate(
    # ── Reference inputs ──────────────────────────────────────────────────────
    ref_type:         str                    = Form(...),  # "upload"|"html"|"url"|"figma"
    ref_image:        Optional[UploadFile]   = File(None),
    ref_html:         Optional[str]          = Form(None),
    ref_url:          Optional[str]          = Form(None),
    ref_figma_token:  Optional[str]          = Form(None),  # Figma PAT
    # ── Implementation inputs ─────────────────────────────────────────────────
    impl_type:        str                    = Form(...),  # "url"|"html"|"adb"|"android_upload"|"windows_upload"
    impl_url:         Optional[str]          = Form(None),
    impl_html:        Optional[str]          = Form(None),
    impl_adb_device:  Optional[str]          = Form(None),  # ADB device serial
    impl_android_image: Optional[UploadFile] = File(None),  # Android or Windows screenshot upload
    # ── Auth config for implementation URL ───────────────────────────────────
    impl_auth_type:   Optional[str]          = Form(None),  # "none"|"form"|"cookies"|"basic"
    impl_auth_user:   Optional[str]          = Form(None),
    impl_auth_pass:   Optional[str]          = Form(None),
    impl_auth_cookies:Optional[str]          = Form(None),  # "name=val; name2=val2"
    impl_auth_target: Optional[str]          = Form(None),  # URL to reach after auth
    # ── Platform / viewport ───────────────────────────────────────────────────
    platform:         Optional[str]          = Form("desktop"),  # "desktop"|"android_phone"|"android_tablet"
):
    job_id = str(uuid.uuid4())
    q: asyncio.Queue = asyncio.Queue()
    _queues[job_id] = q

    ref_image_bytes: Optional[bytes] = None
    if ref_image:
        ref_image_bytes = await ref_image.read()
        if len(ref_image_bytes) > MAX_UPLOAD_BYTES:
            from fastapi import HTTPException
            raise HTTPException(413, f"Reference image exceeds {MAX_UPLOAD_BYTES//1024//1024} MB limit")

    android_image_bytes: Optional[bytes] = None
    if impl_android_image:
        android_image_bytes = await impl_android_image.read()
        if len(android_image_bytes) > MAX_UPLOAD_BYTES:
            from fastapi import HTTPException
            raise HTTPException(413, f"Implementation image exceeds {MAX_UPLOAD_BYTES//1024//1024} MB limit")

    auth_cfg = {
        "type":    impl_auth_type or "none",
        "user":    impl_auth_user or "",
        "pass":    impl_auth_pass or "",
        "cookies": impl_auth_cookies or "",
        "target":  impl_auth_target or "",
    }

    vp = VIEWPORTS.get(platform or "desktop", VIEWPORTS["desktop"])

    # Validate ADB device serial before accepting the job
    if impl_type == "adb" and impl_adb_device:
        if not _ADB_SERIAL_RE.match(impl_adb_device):
            from fastapi import HTTPException
            raise HTTPException(400, "Invalid ADB device serial format")

    asyncio.create_task(_run_evaluation(
        job_id, q,
        ref_type, ref_image_bytes, ref_html, ref_url, ref_figma_token,
        impl_type, impl_url, impl_html, impl_adb_device, android_image_bytes,
        auth_cfg, vp, platform or "desktop",
    ))
    return {"job_id": job_id}


# ══════════════════════════════════════════════════════════════════════════════
# SSE STREAM
# ══════════════════════════════════════════════════════════════════════════════

SSE_TIMEOUT_SECS = 300   # 5-minute hard deadline per evaluation job

@app.get("/api/stream/{job_id}")
async def stream(job_id: str):
    q = _queues.get(job_id)
    if not q:
        return HTMLResponse("Job not found", status_code=404)

    async def event_gen():
        deadline = time.time() + SSE_TIMEOUT_SECS
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                yield f"data: {json.dumps({'type':'error','msg':'Evaluation timed out after 5 minutes'})}\n\n"
                _queues.pop(job_id, None)
                break
            try:
                msg = await asyncio.wait_for(q.get(), timeout=min(remaining, 30))
            except asyncio.TimeoutError:
                # Send a keep-alive comment so the browser connection stays open
                yield ": keepalive\n\n"
                continue
            yield f"data: {json.dumps(msg)}\n\n"
            if msg.get("type") in ("done", "error"):
                _queues.pop(job_id, None)
                break

    return StreamingResponse(event_gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ══════════════════════════════════════════════════════════════════════════════
# CORE EVALUATION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

async def _run_evaluation(
    job_id, q,
    ref_type, ref_image_bytes, ref_html, ref_url, ref_figma_token,
    impl_type, impl_url, impl_html, impl_adb_device, android_image_bytes,
    auth_cfg, viewport, platform,
):
    async def emit(step, pct, msg, data=None):
        payload = {"type": "progress", "step": step, "pct": pct, "msg": msg}
        if data: payload.update(data)
        await q.put(payload)

    plat_type  = _platform_type(platform)
    is_mobile  = plat_type == "mobile"
    is_windows = plat_type == "windows"

    async with _job_semaphore:
        try:
            loop    = asyncio.get_running_loop()   # fixes deprecation warning in Python 3.10+
            ref_ss  = SS_DIR / f"{job_id}_ref.png"
            impl_ss = SS_DIR / f"{job_id}_impl.png"

            # ── Step 1: Reference ─────────────────────────────────────────────
            await emit("reference", 5, "Preparing reference design …")

            if ref_type == "upload" and ref_image_bytes:
                img = Image.open(io.BytesIO(ref_image_bytes)).convert("RGB")
                max_w = viewport["width"]
                if img.width > max_w:
                    r = max_w / img.width
                    img = img.resize((max_w, int(img.height * r)), Image.LANCZOS)
                img.save(str(ref_ss), "PNG")

            elif ref_type in ("figma", "url") and ref_url:
                if ref_type == "figma" and ref_figma_token:
                    await emit("reference", 10, "Fetching Figma frame via REST API …")
                    await loop.run_in_executor(None, _screenshot_figma, ref_url, ref_figma_token, ref_ss)
                elif ref_type == "figma" and not ref_figma_token:
                    await emit("reference", 10, "Screenshotting Figma URL (no token, may be partial) …")
                    await loop.run_in_executor(None, _screenshot_url, ref_url, ref_ss, viewport)
                else:
                    await loop.run_in_executor(None, _screenshot_url, ref_url, ref_ss, viewport)

            elif ref_type == "html" and ref_html:
                tmp = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
                        f.write(ref_html)
                        tmp = Path(f.name)
                    await loop.run_in_executor(None, _screenshot_file, tmp, ref_ss, viewport)
                finally:
                    if tmp:
                        tmp.unlink(missing_ok=True)
            else:
                await q.put({"type": "error", "msg": "Invalid reference input"}); return

            await emit("reference", 20, "Reference captured ✓", {"ref_img": _img_url(ref_ss)})

            # ── Step 2: Implementation ─────────────────────────────────────────
            auth_label = f" [{auth_cfg['type']} auth]" if auth_cfg["type"] != "none" else ""
            plat_label = f" [{platform}]" if platform != "desktop" else ""
            await emit("impl", 25, f"Rendering implementation{plat_label}{auth_label} …")

            if impl_type == "adb" and impl_adb_device:
                await emit("impl", 30, f"Capturing ADB screenshot from {impl_adb_device} …")
                await loop.run_in_executor(None, _screenshot_adb, impl_adb_device, impl_ss)

            elif impl_type in ("android_upload", "windows_upload") and android_image_bytes:
                img = Image.open(io.BytesIO(android_image_bytes)).convert("RGB")
                img.save(str(impl_ss), "PNG")

            elif impl_type == "url" and impl_url:
                ua = WINDOWS_UA if is_windows else None
                if auth_cfg["type"] == "none":
                    await loop.run_in_executor(None, _screenshot_url, impl_url, impl_ss, viewport, ua)
                else:
                    await emit("impl", 30, f"Auth agent starting ({auth_cfg['type']}) …")
                    result = await loop.run_in_executor(
                        None, _screenshot_with_auth, impl_url, impl_ss, auth_cfg, viewport, ua
                    )
                    if result.get("agent_log"):
                        await emit("impl", 38, result["agent_log"])

            elif impl_type == "html" and impl_html:
                tmp = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
                        f.write(impl_html)
                        tmp = Path(f.name)
                    await loop.run_in_executor(None, _screenshot_file, tmp, impl_ss, viewport)
                finally:
                    if tmp:
                        tmp.unlink(missing_ok=True)
            else:
                await q.put({"type": "error", "msg": "Invalid implementation input"}); return

            await emit("impl", 45, "Implementation rendered ✓", {"impl_img": _img_url(impl_ss)})

            # ── Step 3: Pixel metrics ──────────────────────────────────────────
            await emit("metrics", 50, "Computing pixel & SSIM metrics …")
            px = await loop.run_in_executor(None, _pixel_metrics, ref_ss, impl_ss)
            await emit("metrics", 65, "Pixel metrics done ✓", {"pixel": px})

            # ── Step 4: VLM judge ──────────────────────────────────────────────
            rubric_map = {"mobile": MOBILE_RUBRIC_PROMPT, "windows": WINDOWS_RUBRIC_PROMPT}
            rubric = rubric_map.get(plat_type, RUBRIC_PROMPT)
            await emit("vlm", 70, f"Asking VLM judge ({VLM_MODEL}, {plat_type} rubric) …")
            vlm = await loop.run_in_executor(None, _vlm_judge, ref_ss, impl_ss, rubric)
            await emit("vlm", 88, "VLM scoring done ✓", {"vlm": vlm})

            # ── Step 5: Composite ──────────────────────────────────────────────
            await emit("composite", 95, "Computing composite score …")
            ssim_safe = px["ssim"] if not (px["ssim"] != px["ssim"]) else 0.0  # NaN guard
            composite = round(
                0.40 * vlm.get("total", 0)
                + 0.35 * ssim_safe * 100
                + 0.25 * px["pixel_match_20"],
                1
            )
            # Clamp to [0, 100]
            composite = max(0.0, min(100.0, composite))
            await q.put({
                "type": "done", "composite": composite, "grade": _grade(composite),
                "pixel": px, "vlm": vlm,
                "ref_img": _img_url(ref_ss), "impl_img": _img_url(impl_ss),
                "auth_type": auth_cfg["type"],
                "platform": platform,
                "plat_type": plat_type,
                "is_mobile": is_mobile,
                "is_windows": is_windows,
            })
            # Clean stale screenshots opportunistically
            _cleanup_old_screenshots()

        except Exception as exc:
            import traceback
            await q.put({"type": "error", "msg": str(exc), "trace": traceback.format_exc()})


# ══════════════════════════════════════════════════════════════════════════════
# SCREENSHOT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _img_url(path: Path) -> str:
    return f"/screenshots/{path.name}"


def _screenshot_url(url: str, out: Path, viewport: dict = None, user_agent: str = None):
    """Plain screenshot — no auth."""
    from playwright.sync_api import sync_playwright
    vp = viewport or VIEWPORT
    with sync_playwright() as p:
        b = p.chromium.launch()
        ctx_kwargs = {"viewport": vp}
        if user_agent:
            ctx_kwargs["user_agent"] = user_agent
        pg = b.new_page(**ctx_kwargs)
        pg.goto(url, wait_until="networkidle", timeout=30000)
        pg.wait_for_timeout(500)
        pg.screenshot(path=str(out), full_page=False)
        b.close()


def _screenshot_file(html_path: Path, out: Path, viewport: dict = None):
    from playwright.sync_api import sync_playwright
    vp = viewport or VIEWPORT
    url = f"file://{html_path.resolve()}"
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport=vp)
        pg.goto(url, wait_until="networkidle")
        pg.wait_for_timeout(300)
        pg.screenshot(path=str(out), full_page=False)
        b.close()


# ── ADB screenshot ────────────────────────────────────────────────────────────
def _screenshot_adb(device_id: str, out: Path):
    """Capture screenshot from an Android device via ADB."""
    import subprocess
    subprocess.run(
        [ADB_PATH, "-s", device_id, "shell", "screencap", "-p", "/sdcard/_playground_ss.png"],
        check=True, timeout=30, capture_output=True,
    )
    subprocess.run(
        [ADB_PATH, "-s", device_id, "pull", "/sdcard/_playground_ss.png", str(out)],
        check=True, timeout=30, capture_output=True,
    )
    # Clean up temp file on device
    subprocess.run(
        [ADB_PATH, "-s", device_id, "shell", "rm", "/sdcard/_playground_ss.png"],
        timeout=10, capture_output=True,
    )


# ── Figma REST API screenshot ─────────────────────────────────────────────────
def _screenshot_figma(figma_url: str, token: str, out: Path):
    """
    Uses Figma Export API to download a frame as PNG.
    URL formats supported:
      https://www.figma.com/design/FILE_KEY/...?node-id=X-Y
      https://www.figma.com/file/FILE_KEY/...?node-id=X%3AY
    Falls back to Playwright screenshot if node-id cannot be parsed.
    """
    # Parse file key and node id from Figma URL
    m = re.search(r"figma\.com/(?:file|design|proto)/([A-Za-z0-9]+)", figma_url)
    if not m:
        raise ValueError(f"Cannot extract Figma file key from URL: {figma_url}")
    file_key = m.group(1)

    parsed    = urllib.parse.urlparse(figma_url)
    params    = urllib.parse.parse_qs(parsed.query)
    node_id   = (params.get("node-id") or params.get("node_id") or [""])[0]
    node_id   = node_id.replace("%3A", ":").replace("-", ":")  # normalise

    if not node_id:
        # No node-id → use /v1/files/{key} thumbnails endpoint (correct Figma API)
        api_url = f"https://api.figma.com/v1/files/{file_key}/thumbnails"
        req = urllib.request.Request(api_url, headers={"X-Figma-Token": token})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read())
        # Thumbnail endpoint returns {"thumbnails": {"PAGE_ID": "https://…"}}
        thumbnails = data.get("thumbnails", {})
        img_url = next(iter(thumbnails.values()), None) if thumbnails else None
    else:
        ids_param = urllib.parse.quote(node_id)
        api_url   = f"https://api.figma.com/v1/images/{file_key}?ids={ids_param}&format=png&scale=2"
        req = urllib.request.Request(api_url, headers={"X-Figma-Token": token})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read())
        # Image export endpoint returns {"images": {"NODE_ID": "https://…"}}
        images = data.get("images", {})
        img_url = next(iter(images.values()), None) if images else None

    if not img_url:
        raise ValueError(f"Figma API did not return an image URL. Check that the file key and node-id are correct. Response keys: {list(data.keys())}")

    with urllib.request.urlopen(img_url, timeout=30) as r:
        img_bytes = r.read()

    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    if img.width > VIEWPORT["width"]:
        ratio = VIEWPORT["width"] / img.width
        img = img.resize((VIEWPORT["width"], int(img.height * ratio)), Image.LANCZOS)
    img.save(str(out), "PNG")


# ── Auth-aware screenshot (form / cookies / basic) ────────────────────────────
def _screenshot_with_auth(url: str, out: Path, auth_cfg: dict, viewport: dict = None, user_agent: str = None) -> dict:
    """
    Returns {"agent_log": str} with a description of what the agent did.
    """
    from playwright.sync_api import sync_playwright

    vp        = viewport or VIEWPORT
    auth_type = auth_cfg["type"]
    user      = auth_cfg["user"]
    password  = auth_cfg["pass"]
    cookies   = auth_cfg["cookies"]
    target    = auth_cfg["target"] or url
    ua_kwargs = {"user_agent": user_agent} if user_agent else {}

    with sync_playwright() as p:

        # ── Cookie injection ──────────────────────────────────────────────────
        if auth_type == "cookies":
            browser = p.chromium.launch()
            ctx     = browser.new_context(viewport=vp, **ua_kwargs)
            parsed  = urllib.parse.urlparse(url)
            base_url = f"{parsed.scheme}://{parsed.netloc}"
            cookie_list = []
            for part in cookies.split(";"):
                part = part.strip()
                if "=" in part:
                    k, _, v = part.partition("=")
                    cookie_list.append({"name": k.strip(), "value": v.strip(), "url": base_url})
            if cookie_list:
                ctx.add_cookies(cookie_list)
            pg = ctx.new_page()
            pg.goto(target, wait_until="networkidle", timeout=30000)
            pg.wait_for_timeout(500)
            pg.screenshot(path=str(out), full_page=False)
            browser.close()
            return {"agent_log": f"Cookie auth: injected {len(cookie_list)} cookie(s) → navigated to {target}"}

        # ── HTTP Basic auth ───────────────────────────────────────────────────
        elif auth_type == "basic":
            browser = p.chromium.launch()
            ctx     = browser.new_context(viewport=vp,
                                          http_credentials={"username": user, "password": password},
                                          **ua_kwargs)
            pg      = ctx.new_page()
            pg.goto(target, wait_until="networkidle", timeout=30000)
            pg.wait_for_timeout(500)
            pg.screenshot(path=str(out), full_page=False)
            browser.close()
            return {"agent_log": f"Basic auth: {user}:*** → {target}"}

        # ── Form auth (VLM agent) ─────────────────────────────────────────────
        elif auth_type == "form":
            browser = p.chromium.launch()
            ctx     = browser.new_context(viewport=vp, **ua_kwargs)
            pg      = ctx.new_page()

            # Navigate to the login URL
            pg.goto(url, wait_until="networkidle", timeout=30000)
            pg.wait_for_timeout(400)

            # Take screenshot and ask VLM to identify the login form
            ss_bytes = pg.screenshot()
            ss_b64   = base64.b64encode(ss_bytes).decode()
            form_info = _vlm_detect_login_form(ss_b64)

            agent_log = f"VLM detected: {form_info.get('detection', 'unknown')}"

            if form_info.get("is_login_page"):
                u_sel = form_info.get("username_selector", "")
                p_sel = form_info.get("password_selector", "")
                s_sel = form_info.get("submit_selector", "")

                # Fill username
                if u_sel:
                    try:
                        pg.fill(u_sel, user, timeout=3000)
                        agent_log += f" | filled username via '{u_sel}'"
                    except Exception:
                        # Fallback selectors
                        for fb in ["input[type='email']", "input[name='username']", "input[name='email']", "input[id='username']"]:
                            try: pg.fill(fb, user, timeout=1000); agent_log += f" | username fallback '{fb}'"; break
                            except Exception: pass

                # Fill password
                if p_sel:
                    try:
                        pg.fill(p_sel, password, timeout=3000)
                        agent_log += f" | filled password via '{p_sel}'"
                    except Exception:
                        try: pg.fill("input[type='password']", password, timeout=2000); agent_log += " | password fallback"
                        except Exception: pass

                # Click submit
                if s_sel:
                    try:
                        pg.click(s_sel, timeout=3000)
                        agent_log += f" | clicked '{s_sel}'"
                    except Exception:
                        for fb in ["button[type='submit']", "input[type='submit']", "button:text('Sign in')", "button:text('Login')", "button:text('Log in')"]:
                            try: pg.click(fb, timeout=1000); agent_log += f" | submit fallback '{fb}'"; break
                            except Exception: pass

                pg.wait_for_load_state("networkidle", timeout=10000)
                pg.wait_for_timeout(600)

                # Navigate to target URL if different from login URL
                if target and target != url and pg.url != target:
                    pg.goto(target, wait_until="networkidle", timeout=30000)
                    pg.wait_for_timeout(500)

                agent_log += f" | landed on: {pg.url}"
            else:
                agent_log += " — page did not look like a login form; taking screenshot as-is"

            pg.screenshot(path=str(out), full_page=False)
            browser.close()
            return {"agent_log": agent_log}

        # ── Fallback: no auth ─────────────────────────────────────────────────
        else:
            _screenshot_url(url, out)
            return {"agent_log": "No auth applied"}


# ── VLM login form detector ───────────────────────────────────────────────────
LOGIN_DETECT_PROMPT = """You are a web automation agent. Analyze this screenshot.

Determine if this is a login / sign-in page. If it is, identify the CSS selectors
for the username/email field, password field, and submit button.

Prefer specific selectors in this order:
  1. id selector:          #email, #username, #password
  2. name attribute:       input[name='email'], input[name='username']
  3. type attribute:       input[type='email'], input[type='password']
  4. visible text/label:   button:text('Sign in')

Respond ONLY with valid JSON, no extra text:
{
  "is_login_page": <true|false>,
  "confidence": "<high|medium|low>",
  "username_selector": "<CSS selector or empty string>",
  "password_selector": "<CSS selector or empty string>",
  "submit_selector":   "<CSS selector or empty string>",
  "detection":         "<one-line summary of what you see>"
}"""


def _vlm_detect_login_form(screenshot_b64: str) -> dict:
    messages = [{
        "role": "user",
        "content": [
            {"type": "text",      "text": LOGIN_DETECT_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}},
        ],
    }]
    payload = json.dumps({"model": VLM_MODEL, "messages": messages, "max_tokens": 400}).encode()
    req = urllib.request.Request(
        f"{API_BASE}/v1/chat/completions", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = json.loads(r.read())["choices"][0]["message"]["content"].strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"): raw = raw[4:]
        return json.loads(raw)
    except Exception as e:
        return {"is_login_page": False, "detection": f"VLM detection error: {e}"}


# ══════════════════════════════════════════════════════════════════════════════
# METRICS & SCORING
# ══════════════════════════════════════════════════════════════════════════════

def _load_arr(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def _pixel_metrics(ref_p: Path, cmp_p: Path) -> dict:
    ref = _load_arr(ref_p)
    cmp = _load_arr(cmp_p)
    if ref.shape != cmp.shape:
        cmp = np.array(Image.fromarray(cmp).resize((ref.shape[1], ref.shape[0]), Image.LANCZOS))
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


MOBILE_RUBRIC_PROMPT = """You are a mobile UI/UX design quality evaluator. You are given two screenshots:
  IMAGE 1: The REFERENCE DESIGN (golden standard — could be a Figma mockup or design screenshot).
  IMAGE 2: A MOBILE APP IMPLEMENTATION to evaluate.

Score the implementation on EACH of the 5 mobile dimensions below from 0-20 (total = 100):

1. **App Bar & Navigation** (0-20): Top app bar height/style, title, back/action icons, bottom navigation bar or tab bar presence and accuracy.
2. **Touch Targets & Spacing** (0-20): Minimum 44dp-equivalent tap targets, comfortable spacing between interactive elements, no overlapping or tiny controls.
3. **Typography & Readability** (0-20): Font sizes appropriate for mobile (min ~14sp body), contrast ratios, line heights, heading hierarchy.
4. **Layout & Responsiveness** (0-20): Single-column flow, full-width use, no horizontal overflow, proper safe-area/padding at edges, scrollable lists.
5. **Component Fidelity** (0-20): Accuracy of Material Design / iOS HIG components — cards, FAB, chips, dialogs, bottom sheets, loading states.

Respond ONLY with valid JSON — no extra text:
{
  "app_bar_navigation":  <0-20>,
  "touch_targets":       <0-20>,
  "typography":          <0-20>,
  "layout_responsive":   <0-20>,
  "component_fidelity":  <0-20>,
  "total":               <0-100>,
  "summary":             "<2-3 sentence verdict on the biggest wins and gaps for mobile>"
}"""


WINDOWS_RUBRIC_PROMPT = """You are a Windows desktop application UI/UX design quality evaluator. You are given two screenshots:
  IMAGE 1: The REFERENCE DESIGN (golden standard — Figma mockup or design screenshot).
  IMAGE 2: A WINDOWS APPLICATION IMPLEMENTATION to evaluate.

Score the implementation on EACH of the 5 Windows-specific dimensions below from 0-20 (total = 100):

1. **Title Bar & Window Chrome** (0-20): Window title text, app icon, minimize/maximize/close buttons position and style, menu bar if present.
2. **Navigation & Menus** (0-20): Ribbon, toolbar, sidebar nav, context menus, breadcrumbs — presence, accuracy, and Fluent/Windows 11 styling.
3. **Controls & Components** (0-20): Accuracy of Windows controls — buttons, text boxes, checkboxes, radio buttons, dropdowns, list views, tree views, data grids.
4. **Typography & Icons** (0-20): Segoe UI (or Segoe UI Variable) font usage, icon style (Fluent/MDL2/Segoe MDL2 Assets), text density, label alignment.
5. **Layout & Spacing** (0-20): Windows HIG-compliant panel layout, status bar, acrylic/mica material effects, consistent padding and gutter sizes.

Respond ONLY with valid JSON — no extra text:
{
  "titlebar_chrome":    <0-20>,
  "navigation_menus":   <0-20>,
  "controls_components":<0-20>,
  "typography_icons":   <0-20>,
  "layout_spacing":     <0-20>,
  "total":              <0-100>,
  "summary":            "<2-3 sentence verdict on the biggest wins and gaps for the Windows app>"
}"""


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


def _parse_vlm_json(raw: str) -> dict:
    """
    Robustly parse VLM JSON response.
    Handles markdown fences (```json … ``` or ``` … ```), and falls back to
    regex extraction if the model emits prose before/after the JSON object.
    Also normalises the `total` field so it always equals the sum of dimensions.
    """
    text = raw.strip()

    # Strip markdown code fence (any number of backticks, optional language tag)
    fence_re = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
    m = fence_re.search(text)
    if m:
        text = m.group(1).strip()

    # Try direct parse first
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Fall back: find the outermost JSON object
        obj_m = re.search(r"\{[\s\S]*\}", text)
        if not obj_m:
            return {"error": raw, "total": 0, "summary": "VLM response could not be parsed"}
        try:
            data = json.loads(obj_m.group())
        except json.JSONDecodeError:
            return {"error": raw, "total": 0, "summary": "VLM response could not be parsed"}

    # Normalise total: if dimensions are present, recompute to avoid VLM hallucination
    dim_keys_desktop = ["layout_structure","typography","color_visual_style","component_fidelity","spacing_alignment"]
    dim_keys_mobile  = ["app_bar_navigation","touch_targets","typography","layout_responsive","component_fidelity"]
    dim_keys_windows = ["titlebar_chrome","navigation_menus","controls_components","typography_icons","layout_spacing"]

    for dim_keys in (dim_keys_desktop, dim_keys_mobile, dim_keys_windows):
        if all(k in data for k in dim_keys):
            computed = sum(int(data.get(k, 0)) for k in dim_keys)
            data["total"] = min(computed, 100)  # clamp to 100
            break
    else:
        # Clamp whatever total is there
        data["total"] = min(int(data.get("total", 0)), 100)

    # Ensure all dimension values are ints in [0, 20]
    for k, v in list(data.items()):
        if k not in ("total", "summary", "error") and isinstance(v, (int, float)):
            data[k] = max(0, min(20, int(v)))

    return data


def _vlm_judge(ref_p: Path, cmp_p: Path, rubric_prompt: str = None) -> dict:
    prompt = rubric_prompt or RUBRIC_PROMPT
    messages = [{"role": "user", "content": [
        {"type": "text",      "text": prompt},
        {"type": "text",      "text": "IMAGE 1 — REFERENCE DESIGN:"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_encode(ref_p)}"}},
        {"type": "text",      "text": "IMAGE 2 — IMPLEMENTATION TO EVALUATE:"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_encode(cmp_p)}"}},
    ]}]
    payload = json.dumps({"model": VLM_MODEL, "messages": messages, "max_tokens": 800}).encode()
    req = urllib.request.Request(f"{API_BASE}/v1/chat/completions", data=payload,
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            raw = json.loads(r.read())["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return {"error": str(e), "total": 0, "summary": f"VLM API error: {e}"}
    return _parse_vlm_json(raw)


def _grade(s: float) -> dict:
    if s >= 90: return {"label": "Excellent",   "color": "#16a34a", "letter": "A"}
    if s >= 75: return {"label": "Good",         "color": "#2563eb", "letter": "B"}
    if s >= 60: return {"label": "Acceptable",   "color": "#d97706", "letter": "C"}
    if s >= 40: return {"label": "Needs Work",   "color": "#ea580c", "letter": "D"}
    return              {"label": "Major Drift",  "color": "#dc2626", "letter": "F"}


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=7860, reload=False)
