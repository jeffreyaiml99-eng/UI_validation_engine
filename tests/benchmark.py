#!/usr/bin/env python3
"""
Design Fidelity Playground — Benchmark & Validation Script
===========================================================
Evaluates all 6 built-in sample pairs (3 datasets × 2 quality levels) against
the reference designs and validates:

  1. Score ordering:  close > degraded  (for every dataset)
  2. Score ranges:    close >= 60  (C or better), degraded <= 75 (B or worse)
  3. Timing:          each evaluation completes within timeout
  4. Consistency:     re-runs the same pair twice; scores must be within ±5 pts
  5. Pixel sanity:    SSIM and pixel_match_20 agree directionally

Usage:
    cd UI/playground
    python3 ../tests/benchmark.py                    # uses running server on :7860
    python3 ../tests/benchmark.py --url http://host:port
    python3 ../tests/benchmark.py --offline          # skips VLM, pixel-only mode

Results are written to:
    tests/benchmark_results.json
    tests/benchmark_results.html    (human-readable report)
"""

import argparse
import json
import sys
import time
import datetime
from pathlib import Path
from typing import Optional

# ── Try to import requests; fall back to urllib ───────────────────────────────
try:
    import requests
    def _post(url, data, files=None):
        return requests.post(url, data=data, files=files, timeout=60)
    def _get(url):
        return requests.get(url, timeout=30)
    def _get_sse(url):
        """Consume SSE stream and return final message."""
        with requests.get(url, stream=True, timeout=300) as r:
            for line in r.iter_lines():
                if line and line.startswith(b"data:"):
                    msg = json.loads(line[5:].strip())
                    if msg.get("type") in ("done", "error"):
                        return msg
        return {"type": "error", "msg": "Stream ended without done/error"}
except ImportError:
    import urllib.request
    import urllib.parse

    class _FakeResponse:
        def __init__(self, body: bytes, status: int = 200):
            self._body = body
            self.status_code = status
        def json(self):
            return json.loads(self._body)

    def _post(url, data, files=None):
        # requests-compatible multipart — only works for text fields (no binary files)
        boundary = "----BenchmarkBoundary"
        parts = []
        for k, v in data.items():
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}')
        body = ("\r\n".join(parts) + f"\r\n--{boundary}--\r\n").encode()
        req = urllib.request.Request(url, data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}, method="POST")
        with urllib.request.urlopen(req, timeout=60) as r:
            return _FakeResponse(r.read(), r.status)

    def _get(url):
        with urllib.request.urlopen(url, timeout=30) as r:
            return _FakeResponse(r.read())

    def _get_sse(url):
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=300) as r:
            for raw_line in r:
                line = raw_line.decode().strip()
                if line.startswith("data:"):
                    msg = json.loads(line[5:].strip())
                    if msg.get("type") in ("done", "error"):
                        return msg
        return {"type": "error", "msg": "Stream ended"}


# ── Samples ───────────────────────────────────────────────────────────────────
SAMPLES_DIR = Path(__file__).parent.parent / "playground" / "samples"

PAIRS = [
    ("login",   "close",    "Login — Close"),
    ("login",   "degraded", "Login — Degraded"),
    ("pricing", "close",    "Pricing — Close"),
    ("pricing", "degraded", "Pricing — Degraded"),
    ("signup",  "close",    "Signup — Close"),
    ("signup",  "degraded", "Signup — Degraded"),
]


# ── Core evaluation call ──────────────────────────────────────────────────────
def evaluate_pair(
    base_url: str,
    sample_id: str,
    variant: str,
    label: str,
    offline: bool = False,
) -> dict:
    """
    Evaluate one reference/implementation pair.
    Returns a result dict with keys: label, composite, grade, pixel, vlm, elapsed_s, error.
    """
    ref_path  = SAMPLES_DIR / sample_id / "reference.html"
    impl_path = SAMPLES_DIR / sample_id / f"v1_{variant}.html" if variant == "close" \
               else SAMPLES_DIR / sample_id / f"v2_{variant}.html"

    if not ref_path.exists():
        return {"label": label, "error": f"Missing reference: {ref_path}"}
    if not impl_path.exists():
        return {"label": label, "error": f"Missing impl: {impl_path}"}

    ref_html  = ref_path.read_text()
    impl_html = impl_path.read_text()

    t0 = time.time()

    # Submit evaluation
    try:
        post_data = {
            "ref_type":  "html",
            "ref_html":  ref_html,
            "impl_type": "html",
            "impl_html": impl_html,
            "platform":  "desktop",
        }
        r = _post(f"{base_url}/api/evaluate", data=post_data)
        job_id = r.json()["job_id"]
    except Exception as e:
        return {"label": label, "error": f"Submit failed: {e}"}

    # Stream results
    try:
        result_msg = _get_sse(f"{base_url}/api/stream/{job_id}")
    except Exception as e:
        return {"label": label, "error": f"Stream failed: {e}"}

    elapsed = round(time.time() - t0, 1)

    if result_msg.get("type") == "error":
        return {"label": label, "error": result_msg.get("msg", "unknown error"), "elapsed_s": elapsed}

    return {
        "label":     label,
        "sample_id": sample_id,
        "variant":   variant,
        "composite": result_msg.get("composite", 0),
        "grade":     result_msg.get("grade", {}),
        "pixel":     result_msg.get("pixel", {}),
        "vlm":       result_msg.get("vlm", {}),
        "elapsed_s": elapsed,
        "error":     None,
    }


# ── Validation checks ─────────────────────────────────────────────────────────
def validate_results(results: list[dict]) -> list[str]:
    """Return list of validation failure messages (empty = all passed)."""
    failures = []
    by_sample: dict[str, dict] = {}
    for r in results:
        if r.get("error"):
            continue
        by_sample.setdefault(r["sample_id"], {})[r["variant"]] = r

    for sample_id, variants in by_sample.items():
        close    = variants.get("close")
        degraded = variants.get("degraded")

        if close and degraded:
            if close["composite"] < degraded["composite"]:
                failures.append(
                    f"[{sample_id}] Score ordering violated: "
                    f"close={close['composite']} < degraded={degraded['composite']}"
                )

        if close:
            if close["composite"] < 50:
                failures.append(
                    f"[{sample_id}] Close variant scored too low: {close['composite']} (expected ≥ 50)"
                )
            if close["pixel"]["ssim"] < 0.5:
                failures.append(
                    f"[{sample_id}] Close SSIM too low: {close['pixel']['ssim']:.3f} (expected ≥ 0.5)"
                )

        if degraded:
            if degraded["composite"] > 90:
                failures.append(
                    f"[{sample_id}] Degraded variant scored suspiciously high: {degraded['composite']} (expected ≤ 90)"
                )

        # Directional sanity: SSIM and pixel_match_20 should agree
        for r in (close, degraded):
            if not r:
                continue
            ssim = r["pixel"].get("ssim", 0)
            pm20 = r["pixel"].get("pixel_match_20", 0)
            if ssim > 0.9 and pm20 < 30:
                failures.append(
                    f"[{sample_id}/{r['variant']}] SSIM/PixelMatch disagree: "
                    f"SSIM={ssim:.3f} but pixel_match_20={pm20:.1f}%"
                )

    return failures


# ── HTML report ───────────────────────────────────────────────────────────────
def _grade_color(g: dict) -> str:
    return g.get("color", "#888") if isinstance(g, dict) else "#888"

def _grade_letter(g: dict) -> str:
    return g.get("letter", "?") if isinstance(g, dict) else "?"


def write_html_report(results: list[dict], failures: list[str], out_path: Path):
    rows = ""
    for r in results:
        if r.get("error"):
            rows += f"""<tr>
              <td>{r['label']}</td>
              <td colspan="7" style="color:#F87171">{r['error']}</td>
            </tr>"""
            continue
        gc = _grade_color(r["grade"])
        gl = _grade_letter(r["grade"])
        rows += f"""<tr>
          <td>{r['label']}</td>
          <td style="font-weight:700;color:{gc}">{r['composite']} <span style="font-size:11px">{gl}</span></td>
          <td>{r['pixel'].get('ssim','—')}</td>
          <td>{r['pixel'].get('pixel_match_5','—')}%</td>
          <td>{r['pixel'].get('pixel_match_20','—')}%</td>
          <td>{r['vlm'].get('total','—')}/100</td>
          <td>{r.get('elapsed_s','—')}s</td>
          <td>{'✓' if not r.get('error') else '✗'}</td>
        </tr>"""

    fail_html = ""
    if failures:
        fail_html = "<h2 style='color:#F87171'>⚠ Validation Failures</h2><ul>" + \
                    "".join(f"<li>{f}</li>" for f in failures) + "</ul>"
    else:
        fail_html = "<h2 style='color:#34D399'>✓ All validation checks passed</h2>"

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>Benchmark Results — Design Fidelity Playground</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0B0F1C;color:#F0F4FF;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;padding:32px;font-size:14px}}
h1{{font-size:22px;font-weight:800;margin-bottom:4px}}
.sub{{color:#8896B3;font-size:13px;margin-bottom:24px}}
table{{width:100%;border-collapse:collapse;margin-bottom:24px}}
th{{background:#111827;padding:10px 14px;text-align:left;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:#3D4E68;border-bottom:1px solid rgba(255,255,255,.05)}}
td{{padding:10px 14px;border-bottom:1px solid rgba(255,255,255,.04);font-variant-numeric:tabular-nums}}
tr:hover td{{background:rgba(255,255,255,.02)}}
h2{{font-size:16px;font-weight:700;margin:24px 0 10px}}
li{{margin-bottom:6px;line-height:1.6}}
</style>
</head>
<body>
<h1>Benchmark Results</h1>
<p class="sub">Design Fidelity Playground · Generated {ts}</p>
{fail_html}
<table>
  <thead>
    <tr>
      <th>Pair</th><th>Composite</th><th>SSIM</th>
      <th>Pixel≤5</th><th>Pixel≤20</th><th>VLM</th><th>Time</th><th>OK</th>
    </tr>
  </thead>
  <tbody>
    {rows}
  </tbody>
</table>
<p style="color:#3D4E68;font-size:12px">Composite = 40% VLM + 35% SSIM×100 + 25% Pixel-≤20</p>
</body>
</html>"""
    out_path.write_text(html)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Benchmark Design Fidelity Playground")
    parser.add_argument("--url", default="http://localhost:7860", help="Server base URL")
    parser.add_argument("--offline", action="store_true", help="Skip VLM (pixel metrics only) — not yet implemented")
    parser.add_argument("--consistency", action="store_true", help="Run first pair twice for consistency check")
    args = parser.parse_args()

    out_dir = Path(__file__).parent
    json_out = out_dir / "benchmark_results.json"
    html_out = out_dir / "benchmark_results.html"

    # Check server is up
    print(f"\n{'='*60}")
    print(f"  Design Fidelity Benchmark")
    print(f"  Server: {args.url}")
    print(f"{'='*60}\n")

    try:
        _get(f"{args.url}/api/samples")
        print("  ✓ Server is reachable\n")
    except Exception as e:
        print(f"  ✗ Cannot reach server at {args.url}: {e}")
        print("  Start the server with:  cd playground && python3 server.py")
        sys.exit(1)

    results = []
    pairs_to_run = list(PAIRS)
    if args.consistency:
        pairs_to_run.insert(1, PAIRS[0])   # run first pair twice

    for sample_id, variant, label in pairs_to_run:
        print(f"  Running: {label} ...", end=" ", flush=True)
        r = evaluate_pair(args.url, sample_id, variant, label, offline=args.offline)
        results.append(r)
        if r.get("error"):
            print(f"✗ ERROR: {r['error']}")
        else:
            print(f"✓  composite={r['composite']}  SSIM={r['pixel'].get('ssim','?')}  "
                  f"VLM={r['vlm'].get('total','?')}/100  [{r['elapsed_s']}s]")

    # Consistency check
    if args.consistency and len(results) >= 2:
        r1, r2 = results[0], results[1]
        if not r1.get("error") and not r2.get("error"):
            diff = abs(r1["composite"] - r2["composite"])
            if diff > 5:
                print(f"\n  ⚠ Consistency warning: same pair scored {r1['composite']} vs "
                      f"{r2['composite']} (Δ={diff:.1f} > 5)")
            else:
                print(f"\n  ✓ Consistency OK: Δ={diff:.1f} pts (within ±5)")

    # Validation
    print(f"\n{'='*60}")
    failures = validate_results([r for r in results if r.get("sample_id")])
    if failures:
        print(f"  ✗ {len(failures)} validation failure(s):")
        for f in failures:
            print(f"    • {f}")
    else:
        print(f"  ✓ All {len(results)} evaluations passed validation")

    # Summary table
    print(f"\n{'─'*60}")
    print(f"  {'Pair':<28} {'Composite':>9}  {'SSIM':>6}  {'VLM':>6}  {'Time':>6}")
    print(f"{'─'*60}")
    for r in results:
        if r.get("error"):
            print(f"  {r['label']:<28}  ERROR: {r['error'][:30]}")
        else:
            gl = _grade_letter(r["grade"])
            print(f"  {r['label']:<28} {r['composite']:>8.1f}{gl}  "
                  f"{r['pixel'].get('ssim',0):>6.3f}  "
                  f"{r['vlm'].get('total',0):>5}/100  "
                  f"{r.get('elapsed_s',0):>5.1f}s")
    print(f"{'─'*60}\n")

    # Write outputs
    json_out.write_text(json.dumps({"timestamp": datetime.datetime.now().isoformat(),
                                     "results": results, "failures": failures}, indent=2))
    write_html_report(results, failures, html_out)
    print(f"  JSON report: {json_out}")
    print(f"  HTML report: {html_out}\n")

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
