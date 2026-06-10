# UI Validation Engine

A design-to-code fidelity evaluation system that quantitatively measures how closely a rendered UI implementation matches its reference design.

## Architecture

```
UI/
├── poc/                          # Proof-of-concept CLI evaluator
│   ├── design/reference.html     # Reference design (login page)
│   ├── implementations/          # Test implementations (close / degraded)
│   ├── evaluate.py               # CLI evaluation script
│   └── reports/                  # Generated JSON + HTML reports
│
└── playground/                   # Interactive web playground
    ├── server.py                 # FastAPI backend (SSE streaming)
    ├── static/index.html         # Single-page playground UI
    └── samples/                  # Ready-made sample pairs
        ├── login/                # Login page (reference + 2 impls)
        ├── pricing/              # Pricing cards (reference + 2 impls)
        └── signup/               # Sign-up form (reference + 2 impls)
```

## Evaluation Metrics

| Metric | Type | What it catches |
|--------|------|-----------------|
| **SSIM** | Pixel-level | Structural similarity, color/contrast shifts |
| **Pixel Match ≤5 / ≤20** | Pixel-level | Fine-grained element drift |
| **MSE** | Pixel-level | Mean squared error between renders |
| **VLM Judge (gpt-4o)** | Semantic | Layout, typography, color, components, spacing |

**Composite score** = 40% VLM + 35% SSIM×100 + 25% Pixel-match-≤20

## Grade scale

| Score | Grade |
|-------|-------|
| 90–100 | A — Excellent |
| 75–89  | B — Good |
| 60–74  | C — Acceptable |
| 40–59  | D — Needs Work |
| 0–39   | F — Major Drift |

---

## Quick start — POC CLI

```bash
cd poc
pip install playwright scikit-image pillow numpy
playwright install chromium
python3 evaluate.py
# Reports written to poc/reports/
```

---

## Quick start — Playground (local)

```bash
cd playground
pip install fastapi "uvicorn[standard]" python-multipart pillow numpy scikit-image playwright
playwright install chromium
python3 server.py
# Open http://localhost:7860
```

---

## Deployment (remote server)

The playground is live at:  
**`http://zlabsml-t6.csez.zohocorpin.com:8090`**

To restart:
```bash
ssh test@zlabsml-t6.csez.zohocorpin.com
bash ~/design-playground/start.sh
```

### Environment variable

| Variable | Default | Purpose |
|----------|---------|---------|
| `API_BASE` | `http://10.69.141.113:8023` | OpenAI-compatible VLM API endpoint |

---

## How to use the Playground

### Option 1 — Sample Library (instant demo)
Click any **▶ Quick run** button in the Sample Library strip to auto-load a reference + implementation pair and immediately trigger evaluation.

### Option 2 — Upload your Figma export
1. Export a Figma frame as PNG (File → Export)
2. Drag-drop it into the **Reference Design** panel
3. Paste a live URL or HTML into the **Implementation** panel
4. Click **Evaluate Fidelity**

### Option 3 — Live URL
Switch the implementation panel to **Live URL**, paste `http://your-app/page`, and evaluate against any reference.

---

## References

- [Design2Code (Stanford, 2024)](https://arxiv.org/abs/2403.03163) — foundational benchmark
- [UI2Code^N](https://arxiv.org/abs/2511.08195) — VLM judge protocol
- [ScreenCoder](https://arxiv.org/abs/2507.22827) — CLIP + OCR block matching
- [MLLM-as-a-Judge](https://mllm-judge.github.io/) — judge reliability research
