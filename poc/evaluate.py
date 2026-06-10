#!/usr/bin/env python3
"""
Design-to-Code Fidelity Evaluator
==================================
POC that measures how closely a rendered UI matches a reference design using:
  1. Pixel-level metrics   — SSIM (structural similarity)
  2. Perceptual metrics    — pixel match rate at tolerance
  3. VLM judge             — gpt-4o via local OpenAI-compatible API (0-100 score)
  4. Structured rubric     — 5 dimensions scored independently

Usage:
    python3 evaluate.py
"""

import base64
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# ── Config ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DESIGN_DIR = BASE_DIR / "design"
IMPL_DIR   = BASE_DIR / "implementations"
SS_DIR     = BASE_DIR / "screenshots"
RPT_DIR    = BASE_DIR / "reports"

API_BASE   = "http://127.0.0.1:8023"
VLM_MODEL  = "gpt-4o"          # vision-capable model on local server
VIEWPORT   = {"width": 1280, "height": 900}

REFERENCE_HTML  = DESIGN_DIR  / "reference.html"
IMPLEMENTATIONS = {
    "v1_close":    IMPL_DIR / "v1_close.html",
    "v2_degraded": IMPL_DIR / "v2_degraded.html",
}

SS_DIR.mkdir(parents=True, exist_ok=True)
RPT_DIR.mkdir(parents=True, exist_ok=True)


# ── 1. Screenshot via Playwright ─────────────────────────────────────────────
def screenshot_page(html_path: Path, out_path: Path) -> Path:
    """Render an HTML file in headless Chromium and save a PNG screenshot."""
    from playwright.sync_api import sync_playwright

    file_url = f"file://{html_path.resolve()}"
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page    = browser.new_page(viewport=VIEWPORT)
        page.goto(file_url, wait_until="networkidle")
        # Let any CSS transitions settle
        page.wait_for_timeout(300)
        page.screenshot(path=str(out_path), full_page=False)
        browser.close()
    print(f"  [screenshot] {out_path.name}")
    return out_path


# ── 2. Pixel / SSIM metrics ──────────────────────────────────────────────────
def load_image_array(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)


def compute_pixel_metrics(ref_arr: np.ndarray, cmp_arr: np.ndarray) -> dict:
    """
    Returns:
      ssim_score      — structural similarity [0,1]
      pixel_match_5   — % of pixels within L1-distance ≤5 per channel
      pixel_match_20  — % of pixels within L1-distance ≤20 per channel
      mse             — mean squared error (lower = better)
    """
    # Resize comparison to match reference if dimensions differ
    if ref_arr.shape != cmp_arr.shape:
        img = Image.fromarray(cmp_arr).resize(
            (ref_arr.shape[1], ref_arr.shape[0]), Image.LANCZOS
        )
        cmp_arr = np.array(img)

    ssim_score = ssim(ref_arr, cmp_arr, channel_axis=2, data_range=255)

    diff       = np.abs(ref_arr.astype(np.int32) - cmp_arr.astype(np.int32))
    max_diff   = diff.max(axis=2)           # per-pixel max channel diff
    total_px   = max_diff.size

    pixel_match_5  = float((max_diff <= 5).sum())  / total_px * 100
    pixel_match_20 = float((max_diff <= 20).sum()) / total_px * 100

    mse = float(np.mean(diff ** 2))

    return {
        "ssim":           round(float(ssim_score), 4),
        "pixel_match_5":  round(pixel_match_5,  2),
        "pixel_match_20": round(pixel_match_20, 2),
        "mse":            round(mse, 2),
    }


# ── 3. VLM Judge ─────────────────────────────────────────────────────────────
def encode_image(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_vlm(messages: list, max_tokens: int = 1024) -> str:
    """Call the local OpenAI-compatible API."""
    payload = json.dumps({
        "model":      VLM_MODEL,
        "messages":   messages,
        "max_tokens": max_tokens,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{API_BASE}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"].strip()


RUBRIC_PROMPT = """You are a UI/UX design quality evaluator. You are given two screenshots:
  IMAGE 1: The REFERENCE DESIGN (the golden standard).
  IMAGE 2: A CODE IMPLEMENTATION to evaluate against the reference.

Score the implementation on EACH of the 5 dimensions below from 0-20 (total = 100):

1. **Layout & Structure** (0-20)
   - Card dimensions, centering, overall composition, element order

2. **Typography** (0-20)
   - Heading sizes, font weights, label styles, text hierarchy, placeholder text

3. **Color & Visual Style** (0-20)
   - Background gradient/color, button color, brand icon, input borders, link colors

4. **Component Fidelity** (0-20)
   - Presence and styling of: brand icon+name, email/password inputs, remember-me checkbox,
     forgot-password link, submit button, social login buttons (Google/GitHub), footer link

5. **Spacing & Alignment** (0-20)
   - Padding inside card, gap between elements, input heights, button height, margin consistency

Respond ONLY with valid JSON in exactly this format (no extra text):
{
  "layout_structure":    <0-20>,
  "typography":          <0-20>,
  "color_visual_style":  <0-20>,
  "component_fidelity":  <0-20>,
  "spacing_alignment":   <0-20>,
  "total":               <0-100>,
  "summary":             "<2-3 sentence verdict on the biggest wins and gaps>"
}"""


def vlm_judge(ref_ss: Path, cmp_ss: Path, label: str) -> dict:
    print(f"  [VLM judge] scoring {label} ...")
    ref_b64 = encode_image(ref_ss)
    cmp_b64 = encode_image(cmp_ss)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text",      "text": RUBRIC_PROMPT},
                {"type": "text",      "text": "IMAGE 1 — REFERENCE DESIGN:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{ref_b64}"}},
                {"type": "text",      "text": "IMAGE 2 — IMPLEMENTATION TO EVALUATE:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{cmp_b64}"}},
            ],
        }
    ]

    raw = call_vlm(messages, max_tokens=800)

    # Strip markdown fences if any
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: extract JSON block
        import re
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        result = json.loads(m.group()) if m else {"error": raw}

    return result


# ── 4. Grade helper ──────────────────────────────────────────────────────────
def grade(score: float) -> str:
    if score >= 90: return "A — Excellent"
    if score >= 75: return "B — Good"
    if score >= 60: return "C — Acceptable"
    if score >= 40: return "D — Needs work"
    return           "F — Major drift"


def ssim_grade(s: float) -> str:
    if s >= 0.92: return "Excellent"
    if s >= 0.80: return "Good"
    if s >= 0.65: return "Acceptable"
    if s >= 0.45: return "Needs work"
    return              "Major drift"


# ── 5. Main pipeline ─────────────────────────────────────────────────────────
def run():
    print("\n━━━  Design-to-Code Fidelity Evaluator  ━━━\n")

    # Step 1 — Screenshot reference
    ref_ss = SS_DIR / "reference.png"
    print("Rendering reference design ...")
    screenshot_page(REFERENCE_HTML, ref_ss)
    ref_arr = load_image_array(ref_ss)
    ref_b64_thumb = encode_image(ref_ss)   # keep for report

    all_results = {}

    for label, impl_path in IMPLEMENTATIONS.items():
        print(f"\nEvaluating: {label}")
        print("─" * 40)

        # Step 2 — Screenshot implementation
        cmp_ss = SS_DIR / f"{label}.png"
        screenshot_page(impl_path, cmp_ss)
        cmp_arr = load_image_array(cmp_ss)

        # Step 3 — Pixel metrics
        print("  [metrics] computing SSIM + pixel stats ...")
        px = compute_pixel_metrics(ref_arr, cmp_arr)
        print(f"    SSIM:            {px['ssim']} ({ssim_grade(px['ssim'])})")
        print(f"    Pixel match ≤5:  {px['pixel_match_5']}%")
        print(f"    Pixel match ≤20: {px['pixel_match_20']}%")
        print(f"    MSE:             {px['mse']}")

        # Step 4 — VLM judge
        vlm = vlm_judge(ref_ss, cmp_ss, label)
        total = vlm.get("total", 0)
        print(f"  [VLM]  total={total}/100  →  {grade(total)}")
        print(f"         {vlm.get('summary','')}")

        # Step 5 — Combined composite score
        # Weight: 40% VLM total + 35% SSIM (normalised to 100) + 25% pixel_match_20
        composite = round(
            0.40 * total
            + 0.35 * px["ssim"] * 100
            + 0.25 * px["pixel_match_20"],
            1,
        )
        print(f"  [COMPOSITE] {composite}/100 → {grade(composite)}")

        all_results[label] = {
            "pixel_metrics": px,
            "vlm_scores":    vlm,
            "composite":     composite,
            "grade":         grade(composite),
        }

    # ── Generate JSON report ─────────────────────────────────────────────────
    report = {
        "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model":       VLM_MODEL,
        "viewport":    VIEWPORT,
        "reference":   str(REFERENCE_HTML),
        "results":     all_results,
    }
    rpt_path = RPT_DIR / "eval_report.json"
    with open(rpt_path, "w") as f:
        json.dump(report, f, indent=2)

    # ── Generate HTML report ─────────────────────────────────────────────────
    html_rpt = generate_html_report(report, ref_ss)
    html_rpt_path = RPT_DIR / "eval_report.html"
    with open(html_rpt_path, "w") as f:
        f.write(html_rpt)

    print(f"\n━━━  Reports saved  ━━━")
    print(f"  JSON  →  {rpt_path}")
    print(f"  HTML  →  {html_rpt_path}\n")


# ── 6. HTML report generator ─────────────────────────────────────────────────
def img_to_data_uri(path: Path) -> str:
    return f"data:image/png;base64,{encode_image(path)}"


def generate_html_report(report: dict, ref_ss: Path) -> str:
    rows = ""
    for label, res in report["results"].items():
        px  = res["pixel_metrics"]
        vlm = res["vlm_scores"]
        cmp = res["composite"]
        cmp_ss = SS_DIR / f"{label}.png"

        color = (
            "#16a34a" if cmp >= 75 else
            "#d97706" if cmp >= 50 else
            "#dc2626"
        )

        # VLM dimension bars
        dims = ["layout_structure","typography","color_visual_style","component_fidelity","spacing_alignment"]
        dim_labels = ["Layout & Structure","Typography","Color & Visual Style","Component Fidelity","Spacing & Alignment"]
        bars = ""
        for d, dl in zip(dims, dim_labels):
            val = vlm.get(d, 0)
            pct = val / 20 * 100
            bar_color = "#4f46e5" if pct >= 70 else "#f59e0b" if pct >= 40 else "#ef4444"
            bars += f"""
            <div class="dim-row">
              <span class="dim-label">{dl}</span>
              <div class="bar-bg">
                <div class="bar-fill" style="width:{pct}%;background:{bar_color}"></div>
              </div>
              <span class="dim-score">{val}/20</span>
            </div>"""

        rows += f"""
        <section class="result-card">
          <h2>{label.replace("_"," ").title()}
            <span class="badge" style="background:{color}">{cmp}/100 · {res["grade"]}</span>
          </h2>

          <div class="screenshots">
            <figure>
              <figcaption>Reference Design</figcaption>
              <img src="{img_to_data_uri(ref_ss)}" alt="reference" />
            </figure>
            <figure>
              <figcaption>Implementation</figcaption>
              <img src="{img_to_data_uri(cmp_ss)}" alt="{label}" />
            </figure>
          </div>

          <div class="metrics-grid">
            <div class="metric-box">
              <div class="metric-val">{px['ssim']}</div>
              <div class="metric-lbl">SSIM <small>(1.0 = perfect)</small></div>
            </div>
            <div class="metric-box">
              <div class="metric-val">{px['pixel_match_5']}%</div>
              <div class="metric-lbl">Pixel Match ≤5</div>
            </div>
            <div class="metric-box">
              <div class="metric-val">{px['pixel_match_20']}%</div>
              <div class="metric-lbl">Pixel Match ≤20</div>
            </div>
            <div class="metric-box">
              <div class="metric-val">{vlm.get('total',0)}/100</div>
              <div class="metric-lbl">VLM Score</div>
            </div>
          </div>

          <div class="dims">{bars}</div>

          <blockquote class="summary">{vlm.get("summary","")}</blockquote>
        </section>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>Design Fidelity Evaluation Report</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #f1f5f9;
    color: #1e293b;
    padding: 32px 16px;
    font-size: 14px;
  }}
  h1 {{
    font-size: 22px; font-weight: 700; margin-bottom: 6px; color: #0f172a;
  }}
  .meta {{
    font-size: 12px; color: #64748b; margin-bottom: 32px;
  }}
  .result-card {{
    background: #fff; border-radius: 12px;
    padding: 28px 28px 24px; margin-bottom: 28px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
  }}
  h2 {{
    font-size: 17px; font-weight: 700; color: #0f172a;
    display: flex; align-items: center; gap: 12px; margin-bottom: 20px;
  }}
  .badge {{
    font-size: 12px; font-weight: 600; color: #fff;
    padding: 3px 10px; border-radius: 20px;
  }}
  .screenshots {{
    display: flex; gap: 16px; margin-bottom: 20px;
  }}
  figure {{
    flex: 1; border: 1.5px solid #e2e8f0; border-radius: 8px; overflow: hidden;
  }}
  figcaption {{
    font-size: 11px; font-weight: 600; color: #64748b; text-transform: uppercase;
    letter-spacing: 0.6px; padding: 8px 12px; background: #f8fafc;
    border-bottom: 1px solid #e2e8f0;
  }}
  figure img {{ width: 100%; display: block; }}
  .metrics-grid {{
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;
    margin-bottom: 20px;
  }}
  .metric-box {{
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 8px; padding: 12px 14px; text-align: center;
  }}
  .metric-val {{
    font-size: 20px; font-weight: 700; color: #0f172a; margin-bottom: 4px;
  }}
  .metric-lbl {{
    font-size: 11px; color: #64748b; line-height: 1.4;
  }}
  .metric-lbl small {{ display: block; font-size: 10px; color: #94a3b8; }}
  .dims {{ margin-bottom: 16px; }}
  .dim-row {{
    display: flex; align-items: center; gap: 10px; margin-bottom: 8px;
  }}
  .dim-label {{
    width: 160px; font-size: 12px; color: #475569; flex-shrink: 0;
  }}
  .bar-bg {{
    flex: 1; height: 8px; background: #e2e8f0; border-radius: 4px; overflow: hidden;
  }}
  .bar-fill {{ height: 100%; border-radius: 4px; transition: width 0.3s; }}
  .dim-score {{
    width: 36px; text-align: right; font-size: 12px; font-weight: 600; color: #0f172a;
  }}
  blockquote.summary {{
    background: #f8fafc; border-left: 3px solid #6366f1;
    padding: 10px 14px; border-radius: 0 6px 6px 0;
    font-size: 13px; color: #475569; line-height: 1.55; font-style: italic;
  }}
</style>
</head>
<body>
  <h1>Design-to-Code Fidelity Report</h1>
  <p class="meta">
    Generated: {report["timestamp"]} &nbsp;|&nbsp;
    VLM: {report["model"]} &nbsp;|&nbsp;
    Viewport: {report["viewport"]["width"]}×{report["viewport"]["height"]}px
  </p>

  {rows}

  <p class="meta" style="text-align:center; margin-top:8px;">
    Composite = 40% VLM score + 35% SSIM×100 + 25% pixel-match-≤20
  </p>
</body>
</html>"""


if __name__ == "__main__":
    run()
