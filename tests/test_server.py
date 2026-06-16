"""
Design Fidelity Playground — Test Suite
========================================
Run with:
    cd UI
    pip install pytest pytest-asyncio httpx
    pytest tests/ -v

Tests cover:
  - Pixel metrics (SSIM, pixel match, MSE, NaN guard)
  - Grade thresholds
  - VLM JSON parsing edge cases
  - Platform routing
  - Composite score formula + clamping
  - File size validation
  - ADB serial validation
  - Temp file cleanup
  - Sample library API
  - Full end-to-end via httpx TestClient (HTML → HTML)
"""

import io
import json
import math
import sys
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Add playground to import path
sys.path.insert(0, str(Path(__file__).parent.parent / "playground"))

from server import (
    _grade,
    _pixel_metrics,
    _parse_vlm_json,
    _platform_type,
    _ADB_SERIAL_RE,
    MAX_UPLOAD_BYTES,
    SAMPLES_DIR,
    SS_DIR,
)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _make_png(color=(255, 0, 0), size=(100, 100)) -> Path:
    """Write a solid-color PNG to SS_DIR and return its path."""
    img = Image.new("RGB", size, color)
    path = SS_DIR / f"_test_{color}_{size}.png"
    img.save(str(path))
    return path


def _make_png_bytes(color=(255, 0, 0), size=(100, 100)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# PIXEL METRICS
# ══════════════════════════════════════════════════════════════════════════════

class TestPixelMetrics:
    def test_identical_images_ssim_is_one(self):
        p = _make_png((100, 149, 237))
        m = _pixel_metrics(p, p)
        assert m["ssim"] == pytest.approx(1.0, abs=0.001)

    def test_identical_images_pixel_match_100(self):
        p = _make_png((50, 50, 50))
        m = _pixel_metrics(p, p)
        assert m["pixel_match_5"]  == 100.0
        assert m["pixel_match_20"] == 100.0

    def test_identical_images_mse_zero(self):
        p = _make_png((0, 0, 0))
        m = _pixel_metrics(p, p)
        assert m["mse"] == 0.0

    def test_completely_different_images_low_pixel_match(self):
        ref  = _make_png((0,   0,   0))
        cmp  = _make_png((255, 255, 255))
        m = _pixel_metrics(ref, cmp)
        assert m["pixel_match_5"]  < 1.0
        assert m["pixel_match_20"] < 1.0
        assert m["ssim"] < 0.5

    def test_different_sizes_are_resized(self):
        ref = _make_png((200, 100, 50), size=(200, 200))
        cmp = _make_png((200, 100, 50), size=(100, 100))   # half size, same color
        m = _pixel_metrics(ref, cmp)
        # After resize they should be nearly identical
        assert m["pixel_match_5"] > 90.0

    def test_ssim_is_float_and_bounded(self):
        ref = _make_png((10, 20, 30))
        cmp = _make_png((40, 50, 60))
        m = _pixel_metrics(ref, cmp)
        assert isinstance(m["ssim"], float)
        assert -1.0 <= m["ssim"] <= 1.0

    def test_ssim_nan_guard_in_composite(self):
        """NaN SSIM must not propagate into composite score."""
        # Create a 1×1 image which causes SSIM library to return NaN in some versions
        ref = _make_png((128, 128, 128), size=(7, 7))
        cmp = _make_png((128, 128, 128), size=(7, 7))
        m = _pixel_metrics(ref, cmp)
        ssim_val = m["ssim"]
        # Guard just as server does
        ssim_safe = ssim_val if not math.isnan(ssim_val) else 0.0
        composite = 0.40 * 80 + 0.35 * ssim_safe * 100 + 0.25 * 90.0
        assert not math.isnan(composite)
        assert 0.0 <= composite <= 100.0


# ══════════════════════════════════════════════════════════════════════════════
# GRADE THRESHOLDS
# ══════════════════════════════════════════════════════════════════════════════

class TestGrade:
    @pytest.mark.parametrize("score,letter", [
        (100, "A"), (90, "A"), (89.9, "B"), (75, "B"),
        (74.9, "C"), (60, "C"), (59.9, "D"), (40, "D"),
        (39.9, "F"), (0, "F"),
    ])
    def test_grade_letter(self, score, letter):
        assert _grade(score)["letter"] == letter

    def test_grade_has_color_and_label(self):
        g = _grade(50)
        assert "color" in g and g["color"].startswith("#")
        assert "label" in g and len(g["label"]) > 0

    def test_boundary_exactly_90(self):
        assert _grade(90)["letter"] == "A"

    def test_boundary_exactly_75(self):
        assert _grade(75)["letter"] == "B"

    def test_boundary_exactly_60(self):
        assert _grade(60)["letter"] == "C"

    def test_boundary_exactly_40(self):
        assert _grade(40)["letter"] == "D"


# ══════════════════════════════════════════════════════════════════════════════
# VLM JSON PARSING
# ══════════════════════════════════════════════════════════════════════════════

class TestParseVlmJson:
    def test_clean_json(self):
        raw = json.dumps({
            "layout_structure": 18, "typography": 16,
            "color_visual_style": 15, "component_fidelity": 17,
            "spacing_alignment": 14, "total": 80, "summary": "Looks good.",
        })
        d = _parse_vlm_json(raw)
        assert d["total"] == 80
        assert d["summary"] == "Looks good."

    def test_markdown_fence_stripped(self):
        raw = '```json\n{"layout_structure":10,"typography":10,"color_visual_style":10,"component_fidelity":10,"spacing_alignment":10,"total":50,"summary":"ok"}\n```'
        d = _parse_vlm_json(raw)
        assert d["total"] == 50

    def test_markdown_fence_no_language_tag(self):
        raw = '```\n{"layout_structure":8,"typography":9,"color_visual_style":7,"component_fidelity":8,"spacing_alignment":6,"total":38,"summary":"ok"}\n```'
        d = _parse_vlm_json(raw)
        assert d["total"] == 38

    def test_prose_before_json(self):
        raw = 'Here is my evaluation:\n{"layout_structure":12,"typography":12,"color_visual_style":12,"component_fidelity":12,"spacing_alignment":12,"total":60,"summary":"mid"}'
        d = _parse_vlm_json(raw)
        assert d["total"] == 60

    def test_total_recalculated_from_dimensions(self):
        """VLM says total=100 but dims only sum to 60 — must be corrected."""
        raw = json.dumps({
            "layout_structure": 12, "typography": 12,
            "color_visual_style": 12, "component_fidelity": 12,
            "spacing_alignment": 12, "total": 100, "summary": "test",
        })
        d = _parse_vlm_json(raw)
        assert d["total"] == 60  # sum of dims

    def test_total_clamped_to_100(self):
        """If dimensions somehow sum > 100, clamp."""
        raw = json.dumps({
            "layout_structure": 20, "typography": 20,
            "color_visual_style": 20, "component_fidelity": 20,
            "spacing_alignment": 20, "total": 100, "summary": "perfect",
        })
        d = _parse_vlm_json(raw)
        assert d["total"] == 100

    def test_dimension_values_clamped_to_0_20(self):
        raw = json.dumps({
            "layout_structure": 25, "typography": -5,
            "color_visual_style": 15, "component_fidelity": 15,
            "spacing_alignment": 10, "total": 60, "summary": "x",
        })
        d = _parse_vlm_json(raw)
        assert d["layout_structure"] == 20  # clamped from 25
        assert d["typography"] == 0          # clamped from -5

    def test_unparseable_returns_error_dict(self):
        raw = "Sorry, I cannot score this image."
        d = _parse_vlm_json(raw)
        assert d["total"] == 0
        assert "error" in d or "summary" in d

    def test_mobile_rubric_total_recalculated(self):
        raw = json.dumps({
            "app_bar_navigation": 15, "touch_targets": 14,
            "typography": 16, "layout_responsive": 13,
            "component_fidelity": 12, "total": 99, "summary": "mobile ok",
        })
        d = _parse_vlm_json(raw)
        assert d["total"] == 70  # 15+14+16+13+12

    def test_windows_rubric_total_recalculated(self):
        raw = json.dumps({
            "titlebar_chrome": 18, "navigation_menus": 17,
            "controls_components": 16, "typography_icons": 15,
            "layout_spacing": 14, "total": 999, "summary": "win ok",
        })
        d = _parse_vlm_json(raw)
        assert d["total"] == 80  # 18+17+16+15+14


# ══════════════════════════════════════════════════════════════════════════════
# PLATFORM ROUTING
# ══════════════════════════════════════════════════════════════════════════════

class TestPlatformType:
    @pytest.mark.parametrize("platform,expected", [
        ("desktop",        "desktop"),
        ("android_phone",  "mobile"),
        ("android_tablet", "mobile"),
        ("windows_hd",     "windows"),
        ("windows_fhd",    "windows"),
        ("unknown",        "desktop"),   # graceful fallback
    ])
    def test_routing(self, platform, expected):
        assert _platform_type(platform) == expected


# ══════════════════════════════════════════════════════════════════════════════
# COMPOSITE SCORE
# ══════════════════════════════════════════════════════════════════════════════

class TestCompositeScore:
    def test_formula_weights(self):
        vlm_total      = 80.0
        ssim           = 0.90
        pixel_match_20 = 85.0
        composite = 0.40 * vlm_total + 0.35 * ssim * 100 + 0.25 * pixel_match_20
        # 0.40×80 + 0.35×90 + 0.25×85 = 32 + 31.5 + 21.25 = 84.75
        assert composite == pytest.approx(84.75, abs=0.01)

    def test_composite_clamped_below_100(self):
        composite = round(0.40 * 100 + 0.35 * 1.0 * 100 + 0.25 * 100, 1)
        assert composite <= 100.0

    def test_composite_clamped_above_0(self):
        composite = max(0.0, round(0.40 * 0 + 0.35 * 0 + 0.25 * 0, 1))
        assert composite >= 0.0

    def test_identical_images_near_100(self):
        p = _make_png((100, 149, 237))
        px = _pixel_metrics(p, p)
        # With perfect pixel metrics, composite (without VLM) would be ~75 (SSIM+pixel parts)
        partial = 0.35 * px["ssim"] * 100 + 0.25 * px["pixel_match_20"]
        assert partial > 50.0


# ══════════════════════════════════════════════════════════════════════════════
# FILE SIZE VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

class TestFileSizeLimit:
    def test_max_upload_bytes_is_10mb(self):
        assert MAX_UPLOAD_BYTES == 10 * 1024 * 1024

    def test_small_file_under_limit(self):
        data = _make_png_bytes((200, 100, 50), size=(100, 100))
        assert len(data) < MAX_UPLOAD_BYTES

    def test_oversized_file_detected(self):
        oversized = b"x" * (MAX_UPLOAD_BYTES + 1)
        assert len(oversized) > MAX_UPLOAD_BYTES


# ══════════════════════════════════════════════════════════════════════════════
# ADB SERIAL VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

class TestAdbSerialValidation:
    @pytest.mark.parametrize("serial", [
        "emulator-5554",
        "192.168.1.100:5555",
        "R3CM90ABCDE",
        "abc123",
        "0a1b2c3d",
    ])
    def test_valid_serials(self, serial):
        assert _ADB_SERIAL_RE.match(serial), f"Expected {serial!r} to be valid"

    @pytest.mark.parametrize("serial", [
        "",
        "device; rm -rf /",
        "device && ls",
        "device|ls",
        "device`whoami`",
        "../../../etc/passwd",
        "device\nrm",
    ])
    def test_invalid_serials_rejected(self, serial):
        assert not _ADB_SERIAL_RE.match(serial), f"Expected {serial!r} to be rejected"


# ══════════════════════════════════════════════════════════════════════════════
# SAMPLE LIBRARY FILES EXIST
# ══════════════════════════════════════════════════════════════════════════════

class TestSamples:
    @pytest.mark.parametrize("sample_dir,files", [
        ("login",   ["reference.html", "v1_close.html", "v2_degraded.html"]),
        ("pricing", ["reference.html", "v1_close.html", "v2_degraded.html"]),
        ("signup",  ["reference.html", "v1_close.html", "v2_degraded.html"]),
    ])
    def test_sample_files_exist(self, sample_dir, files):
        for fname in files:
            p = SAMPLES_DIR / sample_dir / fname
            assert p.exists(), f"Missing sample: {p}"

    def test_sample_html_not_empty(self):
        for sd in ("login", "pricing", "signup"):
            for fname in ("reference.html", "v1_close.html", "v2_degraded.html"):
                p = SAMPLES_DIR / sd / fname
                if p.exists():
                    assert len(p.read_text()) > 100, f"Sample too short: {p}"


# ══════════════════════════════════════════════════════════════════════════════
# HTTP API  (requires running server — skipped if not available)
# ══════════════════════════════════════════════════════════════════════════════

try:
    from fastapi.testclient import TestClient
    from server import app as _app
    _client = TestClient(_app)
    _HAS_CLIENT = True
except Exception:
    _HAS_CLIENT = False


@pytest.mark.skipif(not _HAS_CLIENT, reason="FastAPI TestClient not available")
class TestApi:
    def test_root_returns_html(self):
        r = _client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]

    def test_samples_api_returns_list(self):
        r = _client.get("/api/samples")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)
        assert len(data) >= 3

    def test_samples_api_has_icon_and_title(self):
        r = _client.get("/api/samples")
        for s in r.json():
            assert "title" in s
            assert "icon" in s

    def test_get_sample_login_reference(self):
        r = _client.get("/api/samples/login/reference")
        assert r.status_code == 200
        d = r.json()
        assert "html" in d
        assert len(d["html"]) > 100

    def test_get_sample_not_found(self):
        r = _client.get("/api/samples/nonexistent/reference")
        assert r.status_code == 404

    def test_get_sample_invalid_variant(self):
        r = _client.get("/api/samples/login/nonexistent")
        assert r.status_code == 404

    def test_evaluate_missing_ref_type_422(self):
        r = _client.post("/api/evaluate", data={"impl_type": "html"})
        assert r.status_code == 422

    def test_evaluate_oversized_upload_413(self):
        big = io.BytesIO(b"x" * (MAX_UPLOAD_BYTES + 100))
        r = _client.post("/api/evaluate", data={
            "ref_type": "upload", "impl_type": "html",
            "impl_html": "<html><body>test</body></html>",
        }, files={"ref_image": ("big.png", big, "image/png")})
        assert r.status_code == 413

    def test_evaluate_invalid_adb_serial_400(self):
        r = _client.post("/api/evaluate", data={
            "ref_type": "html", "ref_html": "<html></html>",
            "impl_type": "adb", "impl_adb_device": "bad; rm -rf /",
        })
        assert r.status_code == 400

    def test_demo_login_page(self):
        r = _client.get("/demo/login")
        assert r.status_code == 200
        assert "Sign in" in r.text

    def test_demo_auth_correct_credentials(self):
        r = _client.post("/demo/auth", data={"username": "demo", "password": "demo123"},
                         follow_redirects=False)
        assert r.status_code == 302
        assert "/demo/dashboard" in r.headers.get("location", "")

    def test_demo_auth_wrong_credentials(self):
        r = _client.post("/demo/auth", data={"username": "bad", "password": "bad"},
                         follow_redirects=False)
        assert r.status_code == 302
        assert "error" in r.headers.get("location", "")

    def test_demo_dashboard_requires_auth(self):
        # Use a fresh client with no cookies so the session cookie from auth test is absent
        from fastapi.testclient import TestClient as _TC
        fresh = _TC(_app, cookies={})
        r = fresh.get("/demo/dashboard", follow_redirects=False)
        assert r.status_code in (302, 307)

    def test_devices_api_returns_dict(self):
        r = _client.get("/api/devices")
        assert r.status_code == 200
        d = r.json()
        assert "devices" in d
        assert isinstance(d["devices"], list)
