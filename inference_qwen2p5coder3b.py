#!/usr/bin/env python -u
# coding: utf-8

# Install runtime dependencies when needed.
import sys, os, json, importlib.util
from pathlib import Path


def _have_runtime_deps():
    required = (
        "unsloth",
        "transformers",
        "accelerate",
        "safetensors",
        "sentencepiece",
        "pandas",
        "numpy",
        "cairosvg",
        "PIL",
        "trl",
        "peft",
        "bitsandbytes",
    )
    return all(importlib.util.find_spec(name) is not None for name in required)


if (("COLAB_GPU" in os.environ) or ("dl310" not in sys.executable)) and not _have_runtime_deps():
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
        "unsloth", "transformers", "accelerate", "safetensors", "sentencepiece",
        "pandas", "numpy", "cairosvg", "pillow", "trl", "peft", "bitsandbytes"])
else:
    print("Runtime deps available — skipping pip install.")


# Runtime paths and resume outputs.
REPO_ROOT = Path(__file__).resolve().parents[1]


def looks_like_run_root(path: Path) -> bool:
    data_dir = path / "Data"
    if not data_dir.exists():
        return False
    return any(
        candidate.exists()
        for candidate in [
            path / "checkpoints",
            path / "outputs",
            path / "ckpt_results.csv",
            path / "ckpt_audit.csv",
            path / "submission_final.csv",
            path / "audit_final.csv",
            path / "model-merged-bf16",
            path / "model-merged",
        ]
    )


DEFAULT_RUN_NAME = REPO_ROOT.name if looks_like_run_root(REPO_ROOT) else "Qwen2.5-Coder-3B-unsloth-a100-Run2"
RUN_NAME = os.environ.get("RUN_NAME", DEFAULT_RUN_NAME).strip()
if "ROOT" in os.environ:
    ROOT = os.environ["ROOT"]
elif looks_like_run_root(REPO_ROOT):
    ROOT = str(REPO_ROOT)
else:
    ROOT = f"/content/drive/MyDrive/{RUN_NAME}"
if "LOCAL_ROOT" in os.environ:
    LOCAL_ROOT = os.environ["LOCAL_ROOT"]
elif ROOT.startswith("/content/drive/"):
    LOCAL_ROOT = f"/content/{RUN_NAME}"
else:
    LOCAL_ROOT = f"/tmp/{RUN_NAME}"
HF_HOME = os.environ.setdefault("HF_HOME", f"{LOCAL_ROOT}/hf")
HF_HUB_CACHE = os.environ.setdefault("HF_HUB_CACHE", f"{HF_HOME}/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", HF_HUB_CACHE)
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("UNSLOTH_STABLE_DOWNLOADS", "1")
HF_LOCAL_FILES_ONLY = os.environ.get("HF_LOCAL_FILES_ONLY", "0").strip() == "1"

# Shard config, set by the helper manager for multi-GPU inference.
SHARD_ID   = int(os.environ.get("SHARD_ID",   "0"))
NUM_SHARDS = int(os.environ.get("NUM_SHARDS", "1"))

# All artifact paths are overridable via env vars.
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", f"{ROOT}/lora-adapter")
MODEL_PATH   = os.environ.get("MODEL_PATH", f"{ROOT}/model-merged")

TEST_CSV_PATH      = os.environ.get("TEST_CSV_PATH",      f"{ROOT}/Data/test.csv")
TRAIN_CSV_PATH     = os.environ.get("TRAIN_CSV_PATH",     f"{ROOT}/Data/train.csv")
SUBMISSION_CSV_OUT = os.environ.get("SUBMISSION_CSV_OUT", f"{ROOT}/submission.csv")
AUDIT_CSV_OUT      = os.environ.get("AUDIT_CSV_OUT",      f"{ROOT}/audit.csv")
CKPT_RESULTS       = os.environ.get("CKPT_RESULTS",       f"{ROOT}/ckpt_results.csv")
CKPT_AUDIT         = os.environ.get("CKPT_AUDIT",         f"{ROOT}/ckpt_audit.csv")
CKPT_EVERY         = max(1, int(os.environ.get("CKPT_EVERY", "2")))  # save checkpoint every N rows
ENSEMBLE_TOP_K     = max(1, int(os.environ.get("ENSEMBLE_TOP_K", "1")))
INFERENCE_LOAD_IN_4BIT = os.environ.get("INFERENCE_LOAD_IN_4BIT", "0").strip() == "1"
ENSEMBLE_MODEL_PATHS = os.environ.get("ENSEMBLE_MODEL_PATHS", "").strip()
ENABLE_EXACT_TRAIN_RETRIEVAL = os.environ.get("ENABLE_EXACT_TRAIN_RETRIEVAL", "1").strip() != "0"
EXACT_TRAIN_MAX_CANDIDATES = max(1, int(os.environ.get("EXACT_TRAIN_MAX_CANDIDATES", "4")))
CLIP_RERANK_WEIGHT = float(os.environ.get("CLIP_RERANK_WEIGHT", "12.0"))

LOCAL_WORKDIR = os.environ.get("LOCAL_WORKDIR", f"{LOCAL_ROOT}/svg_inference")
os.makedirs(LOCAL_WORKDIR, exist_ok=True)
os.makedirs(LOCAL_ROOT, exist_ok=True)
os.makedirs(ROOT, exist_ok=True)
os.makedirs(HF_HUB_CACHE, exist_ok=True)
print(f"Adapter   : {ADAPTER_PATH}")
print(f"Shard     : {SHARD_ID}/{NUM_SHARDS}")
print(f"Root      : {ROOT}")
print(f"Local root: {LOCAL_ROOT}")
print(f"TestCSV   : {TEST_CSV_PATH}")
print(f"Output    : {SUBMISSION_CSV_OUT}")
print(f"Checkpoint: {CKPT_RESULTS} (every {CKPT_EVERY} rows)")
print(f"HF cache  : {HF_HUB_CACHE}")
print(f"HF local  : {HF_LOCAL_FILES_ONLY}")
import io
import math
import random
import re
import signal
import threading
import time
import xml.etree.ElementTree as ET

import unsloth  # must be first to apply all optimizations before transformers loads
import numpy as np
import pandas as pd
import torch
from unsloth import FastLanguageModel

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True


def normalize_svg_for_csv(svg_text):
    if not isinstance(svg_text, str):
        return svg_text
    normalized = svg_text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    normalized = re.sub(r"\s{2,}", " ", normalized).strip()
    return normalized.replace('"', "'")

# Default single-process inference profile.
PROFILE = {
    "easy_num_candidates"       : 4,
    "complex_num_candidates"    : 12,
    "weak_greedy_num_candidates": 16,
    "weak_greedy_score_threshold": 1035.0,
    "weak_greedy_coverage_threshold": 0.08,
    "greedy_max_new_tokens"         : 1024,
    "greedy_max_new_tokens_complex" : 2048,  # complex prompts can produce longer SVGs
    "sample_max_tokens_simple"      : 768,
    "sample_max_tokens_complex"     : 1024,
    "temperature"               : 0.68,
    "top_p"                     : 0.95,
    "top_k"                     : 64,
    "repetition_penalty"        : 1.05,
    "batch_size"                : 4,
    "render_rerank_top_k"       : 3,
    "render_timeout_sec"        : 1.5,
    "max_prompt_chars"          : 700,
}

PROFILE_PRESETS = {
    "high_diversity": {
        "easy_num_candidates": 8,
        "complex_num_candidates": 20,
        "weak_greedy_num_candidates": 24,
        "greedy_max_new_tokens": 1280,
        "greedy_max_new_tokens_complex": 2560,
        "sample_max_tokens_simple": 1024,
        "sample_max_tokens_complex": 1536,
        "temperature": 0.74,
        "top_p": 0.96,
        "top_k": 80,
        "render_rerank_top_k": 5,
        "max_prompt_chars": 900,
    },
    "very_high_diversity": {
        "easy_num_candidates": 10,
        "complex_num_candidates": 28,
        "weak_greedy_num_candidates": 32,
        "greedy_max_new_tokens": 1536,
        "greedy_max_new_tokens_complex": 3072,
        "sample_max_tokens_simple": 1152,
        "sample_max_tokens_complex": 1792,
        "temperature": 0.78,
        "top_p": 0.97,
        "top_k": 96,
        "render_rerank_top_k": 6,
        "max_prompt_chars": 1100,
    },
}


def _coerce_profile_value(current, raw_value):
    if isinstance(current, bool):
        return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(current, int) and not isinstance(current, bool):
        return int(float(raw_value))
    if isinstance(current, float):
        return float(raw_value)
    return raw_value


def apply_profile_overrides(profile):
    updated = dict(profile)
    preset_name = os.environ.get("INFERENCE_PROFILE_PRESET", "").strip().lower()
    if preset_name:
        preset = PROFILE_PRESETS.get(preset_name)
        if preset is None:
            print(f"Warning: unknown INFERENCE_PROFILE_PRESET={preset_name!r}; ignoring.")
        else:
            updated.update(preset)
            print(f"Profile preset applied: {preset_name}")

    raw_json = os.environ.get("INFERENCE_PROFILE_OVERRIDES", "").strip()
    if raw_json:
        try:
            json_overrides = json.loads(raw_json)
            if isinstance(json_overrides, dict):
                for key, value in json_overrides.items():
                    if key in updated:
                        updated[key] = _coerce_profile_value(updated[key], value)
            else:
                print("Warning: INFERENCE_PROFILE_OVERRIDES must decode to a JSON object; ignoring.")
        except Exception as exc:
            print(f"Warning: failed to parse INFERENCE_PROFILE_OVERRIDES: {exc}")

    for key, current in list(updated.items()):
        env_key = f"PROFILE_{key.upper()}"
        raw_value = os.environ.get(env_key, "").strip()
        if not raw_value:
            continue
        try:
            updated[key] = _coerce_profile_value(current, raw_value)
        except Exception as exc:
            print(f"Warning: failed to parse {env_key}={raw_value!r}: {exc}")

    return updated


PROFILE = apply_profile_overrides(PROFILE)

SYSTEM_PROMPT = (
    "You generate SVG images from text descriptions.\n\n"
    "Requirements:\n"
    "- Return only one complete SVG document: <svg>...</svg>\n"
    "- Output valid XML that renders without errors\n"
    "- Use a 256x256 canvas with width, height, and viewBox\n"
    "- Keep the SVG under 16000 characters and use at most 256 <path> elements\n"
    "- Use only safe SVG elements and attributes; no scripts, event handlers, animation, or external references\n"
    "- Keep the drawing visible, centered, and large enough to read clearly\n"
    "- Match the prompt's subject, shape, and colors as closely as possible\n"
    "- Prefer simple structure, but include enough detail to avoid generic icons\n"
    "- Return the final SVG only, with no markdown or explanation"
)

print(f"Python : {sys.version}")
print(f"Torch  : {torch.__version__}")
if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"Profile: {PROFILE}")
print(f"Ensemble top-k: {ENSEMBLE_TOP_K}")
print(f"Inference precision: {'4-bit' if INFERENCE_LOAD_IN_4BIT else 'bf16/full'}")


# SVG utilities.
ALLOWED_SVG_TAGS = {
    "svg", "g", "path", "rect", "circle", "ellipse", "line", "polyline",
    "polygon", "defs", "use", "symbol", "clipPath", "mask",
    "linearGradient", "radialGradient", "stop", "text", "tspan",
    "title", "desc", "style", "pattern", "marker", "filter",
}
ALLOWED_SVG_ATTRS = {
    "xmlns", "width", "height", "viewBox", "x", "y", "x1", "y1", "x2", "y2",
    "cx", "cy", "r", "rx", "ry", "d", "points", "fill", "fill-rule",
    "fill-opacity", "stroke", "stroke-width", "stroke-opacity", "stroke-linecap",
    "stroke-linejoin", "stroke-miterlimit", "stroke-dasharray", "stroke-dashoffset",
    "opacity", "transform", "id", "class", "clip-path", "mask", "href",
    "xlink:href", "gradientUnits", "gradientTransform", "offset", "stop-color",
    "stop-opacity", "patternUnits", "patternTransform", "preserveAspectRatio",
    "font-size", "font-family", "font-weight", "text-anchor", "letter-spacing",
    "dominant-baseline", "filter", "marker-start", "marker-mid", "marker-end",
    "style", "version",
}
UNSAFE_TOKENS = ("javascript:", "vbscript:", "data:", "http://", "https://",
                 "file:", "<script", "</script", "@import")
SVG_RE = re.compile(r"<svg[\s\S]*?</svg>", re.IGNORECASE)
COMPLEXITY_HINTS = [
    "illustration", "detailed", "complex", "many", "multiple", "pattern",
    "icon set", "logo", "scene", "stack", "five", "several",
]
SHAPE_HINTS = {
    "circle": "<circle", "ring": "<circle", "dot": "<circle",
    "rectangle": "<rect", "square": "<rect", "line": "<line",
    "stripe": "<rect", "triangle": "<polygon", "polygon": "<polygon",
    "text": "<text", "gradient": "gradient", "star": "<polygon",
}
COLOR_HINTS = [
    "black", "white", "red", "green", "blue", "yellow", "orange",
    "purple", "pink", "gray", "grey", "brown", "gold", "silver",
]
DRAWABLE_TAGS = {
    "path", "rect", "circle", "ellipse", "line", "polygon", "polyline", "text", "use"
}
NON_PAINT_VALUES = {"none", "transparent", "inherit", ""}
SVG_XMLNS = "http://www.w3.org/2000/svg"
COLOR_NAME_TO_RGB = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (255, 0, 0),
    "green": (0, 128, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "pink": (255, 192, 203),
    "gray": (128, 128, 128),
    "grey": (128, 128, 128),
    "brown": (165, 42, 42),
    "gold": (255, 215, 0),
    "silver": (192, 192, 192),
}

def _local(tag):   return tag.split("}")[-1] if "}" in tag else tag
def _unsafe(v):    return any(t in str(v or "").lower() for t in UNSAFE_TOKENS)
def _internal(v):
    c = re.sub(r"\s+", "", str(v or ""))
    return bool(c) and (c.startswith("#") or bool(re.fullmatch(r"url\(#[-\w:.]+\)", c)))

def _san_style(s):
    parts = []
    for p in str(s or "").split(";"):
        p = p.strip()
        if not p: continue          # skip empty parts (e.g. trailing semicolon)
        if ":" not in p: return None
        n, v = p.split(":", 1); n = n.strip().lower(); v = v.strip()
        if not n or n.startswith("on") or _unsafe(v): return None
        if "url(" in v.lower() and not _internal(v): return None
        parts.append(f"{n}:{v}")
    return ";".join(parts)

def _san_text(tag, text):
    if text is None: return None
    t = str(text).strip()
    if not t or _unsafe(t): return None
    if tag == "style" and "url(" in t.lower() and "url(#" not in t.lower(): return None
    return t if len(t) <= 2000 else None

def _serialize_svg(root):
    # ElementTree can drop the default SVG namespace after a parse/serialize round-trip.
    s = ET.tostring(root, encoding="unicode", short_empty_elements=True)
    s = s.replace("ns0:", "").replace(":ns0", "")
    s = re.sub(r'\s+xmlns:ns0="[^"]*"', "", s)
    if re.search(r"<svg\b", s) and "xmlns=" not in s:
        s = re.sub(r"<svg\b", f'<svg xmlns="{SVG_XMLNS}"', s, count=1)
    s = re.sub(r">\s+<", "><", s).strip()
    return s

def _parse_svg_color(value):
    compact = re.sub(r"\s+", "", str(value or "")).lower()
    if not compact or compact in NON_PAINT_VALUES:
        return None
    if compact.startswith("#"):
        if len(compact) == 4:
            compact = "#" + "".join(ch * 2 for ch in compact[1:])
        if len(compact) == 7:
            try:
                return tuple(int(compact[i:i + 2], 16) for i in (1, 3, 5))
            except ValueError:
                return None
    rgb_match = re.fullmatch(r"rgb\((\d{1,3}),(\d{1,3}),(\d{1,3})\)", compact)
    if rgb_match:
        return tuple(max(0, min(255, int(group))) for group in rgb_match.groups())
    return COLOR_NAME_TO_RGB.get(compact)

def _is_visible_paint(value):
    compact = re.sub(r"\s+", "", str(value or "")).lower()
    if not compact or compact in NON_PAINT_VALUES:
        return False
    rgb = _parse_svg_color(compact)
    if rgb is None:
        return True
    return min(rgb) < 245

def _san_attrs(elem):
    safe = {}
    for k, v in list(elem.attrib.items()):
        la = _local(k); v = str(v).strip()
        if not v or la.startswith("on") or la not in ALLOWED_SVG_ATTRS: continue
        if _unsafe(v): continue
        if la in {"href", "xlink:href", "clip-path", "mask", "filter",
                  "marker-start", "marker-mid", "marker-end"} and not _internal(v): continue
        if "url(" in v.lower() and not _internal(v): continue
        if la == "style": v = _san_style(v)
        if not v: continue
        if la in {"id", "class", "font-family"} and len(v) > 120: continue
        safe[la] = v
    elem.attrib.clear(); elem.attrib.update(safe)

def _prune(elem):
    for child in list(elem):
        child.tag = _local(child.tag)
        if child.tag not in ALLOWED_SVG_TAGS: elem.remove(child); continue
        _san_attrs(child)
        child.text = _san_text(child.tag, child.text)
        child.tail = None
        _prune(child)

def normalize_svg(svg_text):
    if not svg_text: return ""
    try: root = ET.fromstring(str(svg_text).strip())
    except ET.ParseError: return ""
    root.tag = _local(root.tag)
    if root.tag != "svg": return ""
    _san_attrs(root); _prune(root)
    root.text = root.tail = None
    root.set("xmlns", SVG_XMLNS)
    root.set("width", "256"); root.set("height", "256")
    root.set("viewBox", "0 0 256 256")
    paths = 0
    for e in root.iter():
        e.tag = _local(e.tag)
        if e.tag not in ALLOWED_SVG_TAGS: return ""
        _san_attrs(e); e.text = _san_text(e.tag, e.text); e.tail = None
        if e.tag == "path": paths += 1
    if paths > 256: return ""
    s = _serialize_svg(root)
    return s if len(s) <= 16000 else ""

def svg_semantic_stats(svg_text):
    try:
        root = ET.fromstring(svg_text)
    except Exception:
        return {
            "drawable_count": 0,
            "painted_count": 0,
            "visible_paint_count": 0,
            "distinct_colors": 0,
            "coverage": 0.0,
        }

    drawable_count = 0
    painted_count = 0
    visible_paint_count = 0
    distinct_colors = set()

    for elem in root.iter():
        if _local(elem.tag) not in DRAWABLE_TAGS:
            continue
        drawable_count += 1
        elem_has_paint = False
        elem_has_visible_paint = False

        for attr in ("fill", "stroke"):
            value = (elem.get(attr) or "").strip()
            compact = re.sub(r"\s+", "", value).lower()
            if not value or compact in NON_PAINT_VALUES:
                continue
            elem_has_paint = True
            distinct_colors.add(compact)
            elem_has_visible_paint = elem_has_visible_paint or _is_visible_paint(value)

        style = elem.get("style") or ""
        for part in style.split(";"):
            if ":" not in part:
                continue
            key, value = part.split(":", 1)
            if key.strip().lower() not in {"fill", "stroke"}:
                continue
            value = value.strip()
            compact = re.sub(r"\s+", "", value).lower()
            if not value or compact in NON_PAINT_VALUES:
                continue
            elem_has_paint = True
            distinct_colors.add(compact)
            elem_has_visible_paint = elem_has_visible_paint or _is_visible_paint(value)

        painted_count += int(elem_has_paint)
        visible_paint_count += int(elem_has_visible_paint)

    bounds = _estimate_content_bounds(svg_text)
    if bounds is None:
        coverage = 0.0
    else:
        min_x, min_y, max_x, max_y = bounds
        width = max(0.0, min(256.0, max_x) - max(0.0, min_x))
        height = max(0.0, min(256.0, max_y) - max(0.0, min_y))
        coverage = max(0.0, min(1.0, (width * height) / float(256 * 256)))

    return {
        "drawable_count": drawable_count,
        "painted_count": painted_count,
        "visible_paint_count": visible_paint_count,
        "distinct_colors": len(distinct_colors),
        "coverage": coverage,
    }

def _estimate_content_bounds(svg_text):
    """
    Scan all SVG shape coordinates and path data to estimate the content bounding box.
    Returns (min_x, min_y, max_x, max_y) or None if no coordinates found.
    Approximation: relative path commands and transforms are not resolved, but this
    reliably detects when all content coordinates are packed into a small region.
    """
    try:
        root = ET.fromstring(svg_text)
    except ET.ParseError:
        return None

    xs, ys = [], []

    def _flt(v, default=None):
        try:    return float(v)
        except: return default

    def _extract_numeric_tokens(text):
        if not text:
            return []
        # Avoid malformed captures like ".01.04" by matching a single numeric
        # literal at a time.
        return re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", text)

    for elem in root.iter():
        tag = _local(elem.tag)
        a   = elem.attrib
        if tag == "rect":
            x, y = _flt(a.get("x"), 0.0), _flt(a.get("y"), 0.0)
            w, h = _flt(a.get("width")), _flt(a.get("height"))
            if w is not None and h is not None:
                xs += [x, x + w]; ys += [y, y + h]
        elif tag == "circle":
            cx = _flt(a.get("cx"), 0.0); cy = _flt(a.get("cy"), 0.0)
            r  = _flt(a.get("r"),  0.0)
            xs += [cx - r, cx + r]; ys += [cy - r, cy + r]
        elif tag == "ellipse":
            cx = _flt(a.get("cx"), 0.0); cy = _flt(a.get("cy"), 0.0)
            rx = _flt(a.get("rx"), 0.0); ry = _flt(a.get("ry"), 0.0)
            xs += [cx - rx, cx + rx]; ys += [cy - ry, cy + ry]
        elif tag == "line":
            for k in ("x1", "x2"):
                v = _flt(a.get(k))
                if v is not None: xs.append(v)
            for k in ("y1", "y2"):
                v = _flt(a.get(k))
                if v is not None: ys.append(v)
        elif tag in ("polyline", "polygon"):
            nums = [float(n) for n in _extract_numeric_tokens(a.get("points", ""))]
            xs += nums[0::2]; ys += nums[1::2]
        elif tag == "path":
            # Extract all numbers from d — approximation that works well for detecting
            # content packed into a small coordinate space (e.g. 0–40 vs 0–256).
            nums = [float(n) for n in _extract_numeric_tokens(a.get("d", ""))]
            xs += nums[0::2]; ys += nums[1::2]

    if not xs or not ys:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def rescale_svg(svg_text, canvas=256, target_fill=0.85, min_scale=1.8):
    """
    If SVG content is packed into a region much smaller than the canvas, adjust the
    viewBox so the renderer scales it up to fill the canvas.

    Only fires when the estimated scale factor exceeds min_scale (content occupies
    less than ~1/1.8 of the canvas). Returns original string unchanged if rescaling
    is not needed or fails.
    """
    if not svg_text:
        return svg_text
    bounds = _estimate_content_bounds(svg_text)
    if bounds is None:
        return svg_text
    min_x, min_y, max_x, max_y = bounds
    content_w = max(max_x - min_x, 1.0)
    content_h = max(max_y - min_y, 1.0)
    scale = min(canvas * target_fill / content_w, canvas * target_fill / content_h)
    if scale < min_scale:
        return svg_text  # content already fills the canvas adequately
    # Set viewBox to content region + padding — renderer scales it up to 256x256
    pad  = (canvas / scale) * (1.0 - target_fill) / 2.0
    vb_x = min_x - pad; vb_y = min_y - pad
    vb_w = content_w + 2 * pad; vb_h = content_h + 2 * pad
    try:
        root = ET.fromstring(svg_text)
        root.set("viewBox", f"{vb_x:.4f} {vb_y:.4f} {vb_w:.4f} {vb_h:.4f}")
        root.set("xmlns", SVG_XMLNS)
        root.set("width", str(canvas)); root.set("height", str(canvas))
        s = _serialize_svg(root)
        return s if len(s) <= 16000 else svg_text
    except Exception:
        return svg_text


def extract_svg(text):
    m = SVG_RE.search(text or "")
    if m:
        return m.group(0).strip()
    # Salvage truncated SVG: model output cut off before </svg>
    t = text or ""
    start = t.lower().find("<svg")
    if start == -1:
        return ""
    partial = t[start:]
    last_gt = partial.rfind(">")
    if last_gt < 10:
        return ""
    partial = partial[:last_gt + 1]
    # Close any unclosed <g> groups (cap at 10)
    open_g = partial.lower().count("<g ") + partial.lower().count("<g>") - partial.lower().count("</g>")
    partial += "</g>" * max(0, min(open_g, 10))
    partial += "</svg>"
    return partial.strip()

def is_valid_svg(svg):
    n = normalize_svg(svg)
    if not n: return False
    try:
        root = ET.fromstring(n)
        if _local(root.tag) != "svg": return False
        paths = sum(1 for e in root.iter() if _local(e.tag) == "path")
        stats = svg_semantic_stats(n)
        return (
            len(n) <= 16000
            and paths <= 256
            and stats["drawable_count"] > 0
            and stats["visible_paint_count"] > 0
        )
    except ET.ParseError: return False

def prompt_is_complex(p):
    pl = p.lower()
    return any(t in pl for t in COMPLEXITY_HINTS) or len(p) >= 120

def structural_score(svg):
    if not svg: return -1000.0
    s = 0.0; n = len(svg)
    s += 12.0 if 160 <= n <= 9000 else (6.0 if 80 <= n <= 12000 else 0.0)
    s += min(svg.count("<path"), 12) * 0.6
    s += min(svg.count("<rect"), 10) * 0.4
    s += min(svg.count("<circle"), 10) * 0.4
    s += min(svg.count("<g"), 10) * 0.25
    s += min(svg.count("fill="), 16) * 0.15
    s -= max(0, svg.count("<path") - 64) * 0.15
    if 'viewBox="0 0 256 256"' in svg or "viewBox='0 0 256 256'" in svg: s += 6.0
    return s

def prompt_bonus(prompt, svg):
    s = 0.0; pl = prompt.lower(); sl = svg.lower()
    for kw, mk in SHAPE_HINTS.items():
        if kw in pl and mk in sl: s += 1.25
    for c in COLOR_HINTS:
        if c in pl and c in sl: s += 0.25
    return s

# ── Render helpers (cairosvg + SSIM) ─────────────────────────────────────────
try:
    import cairosvg
    from PIL import Image
    HAS_RENDER = True
    print("cairosvg render stack loaded ✓")
except Exception as e:
    HAS_RENDER = False
    print(f"Render stack unavailable: {e}")

class RenderTimeoutError(RuntimeError):
    pass

def _render_timeout_handler(_signum, _frame):
    raise RenderTimeoutError("SVG render timed out")

def _render_svg_png_bytes(svg_text):
    """Render SVG to raw PNG bytes with timeout. Returns bytes or None on failure."""
    if not HAS_RENDER or not svg_text:
        return None

    def _once():
        return cairosvg.svg2png(bytestring=svg_text.encode(), output_width=256, output_height=256)

    use_alarm = (
        hasattr(signal, "setitimer")
        and threading.current_thread() is threading.main_thread()
        and os.name != "nt"
    )
    if not use_alarm:
        try:
            return _once()
        except Exception:
            return None

    previous_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _render_timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, float(PROFILE["render_timeout_sec"]))
        return _once()
    except Exception:
        return None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def _png_to_rgba_arr(png_bytes):
    """Convert PNG bytes to (H, W, 4) float32 RGBA array alpha-composited on white."""
    rgba = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return np.asarray(Image.alpha_composite(bg, rgba), dtype=np.float32) / 255.0


def _png_to_rgb_arr(png_bytes):
    return _png_to_rgba_arr(png_bytes)[..., :3]


def _png_to_gray_arr(png_bytes):
    rgb = _png_to_rgb_arr(png_bytes)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def raster_gray(svg_text):
    png = _render_svg_png_bytes(svg_text)
    if png is None:
        return None
    return _png_to_gray_arr(png)


def raster_rgb(svg_text):
    """Render SVG to (H, W, 3) float32 RGB array, alpha-composited on white."""
    png = _render_svg_png_bytes(svg_text)
    if png is None:
        return None
    return _png_to_rgb_arr(png)

def ssim_proxy(a, b):
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ma, mb = float(a.mean()), float(b.mean())
    va, vb = float(a.var()),  float(b.var())
    cov = float(((a - ma) * (b - mb)).mean())
    num = (2 * ma * mb + c1) * (2 * cov + c2)
    den = (ma ** 2 + mb ** 2 + c1) * (va + vb + c2)
    return float(num / den) if den > 1e-12 else 0.0

FALLBACK_SVG = normalize_svg(
    "<svg xmlns='http://www.w3.org/2000/svg' width='256' height='256' "
    "viewBox='0 0 256 256'><rect width='256' height='256' fill='white'/>"
    "<rect x='40' y='40' width='176' height='176' rx='24' fill='black'/></svg>"
)

def render_quality(svg):
    """Returns (edge_density, non_white_fraction, edge_clarity) by rendering the SVG.

    edge_clarity is the fraction of edge pixels that are "strong" (>4× mean gradient).
    High edge_clarity means crisp, well-defined shapes — a better EdgeF1 proxy than raw
    edge density alone, since EdgeF1 rewards sharp boundaries not noisy gradients.
    """
    if not HAS_RENDER or not svg:
        return None, None, None
    gray = raster_gray(svg)
    if gray is None:
        return None, None, None
    edge_h = np.abs(np.diff(gray, axis=0))
    edge_v = np.abs(np.diff(gray, axis=1))
    edge_all = np.concatenate([edge_h.ravel(), edge_v.ravel()])
    edge = float(edge_all.mean())
    non_white = float(np.mean(gray < 0.95))
    edge_clarity = float(np.mean(edge_all > 4.0 * edge)) if edge > 1e-6 else 0.0
    return edge, non_white, edge_clarity

print("SVG utilities loaded ✓")

# ── CLIP reranker globals (populated after main model loads) ──────────────────
# Controls: USE_CLIP_RERANK=0 to disable, CLIP_MODEL=<hf_id_or_local_path> to override.
USE_CLIP_RERANK = os.environ.get("USE_CLIP_RERANK", "1").strip() not in {"0", "false", "no"}
HAS_CLIP = False
_clip_model = None
_clip_processor = None


# Model loading.
# run.py sets CUDA_VISIBLE_DEVICES to a single GPU per process; parallelism
# is achieved by launching one process per GPU, not multiple copies here.

def ensure_loadable_checkpoint_dir(path):
    if not isinstance(path, str) or not os.path.isdir(path):
        return path

    expected_names = (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    if any(os.path.exists(os.path.join(path, name)) for name in expected_names):
        return path

    safetensor_files = sorted(
        name
        for name in os.listdir(path)
        if name.endswith(".safetensors") and os.path.isfile(os.path.join(path, name))
    )
    if len(safetensor_files) != 1:
        return path

    source_name = safetensor_files[0]
    if source_name == "model.safetensors":
        return path

    source_path = os.path.join(path, source_name)
    target_path = os.path.join(path, "model.safetensors")
    try:
        os.replace(source_path, target_path)
        print(f"Normalized model weights: {path} ({source_name} -> model.safetensors)")
    except FileNotFoundError:
        if not os.path.exists(target_path):
            print(f"Warning: single-file checkpoint disappeared while normalizing {path}")
    except OSError as exc:
        print(f"Warning: failed to normalize single-file checkpoint under {path}: {exc}")
    return path

def resolve_ensemble_paths():
    default_path = MODEL_PATH if os.path.isdir(MODEL_PATH) else ADAPTER_PATH

    if ENSEMBLE_MODEL_PATHS:
        explicit_paths = [
            path.strip()
            for path in ENSEMBLE_MODEL_PATHS.split(",")
            if path.strip()
        ]
        existing = [path for path in explicit_paths if os.path.isdir(path)]
        if existing:
            return existing[:ENSEMBLE_TOP_K]
        print(f"Warning: none of ENSEMBLE_MODEL_PATHS exist. Falling back to {default_path}")

    if ENSEMBLE_TOP_K <= 1:
        return [default_path]

    scores_path = os.path.join(ADAPTER_PATH, "kaggle_artifacts", "checkpoint_proxy_scores.csv")
    if os.path.exists(scores_path):
        try:
            df = pd.read_csv(scores_path)
            df = df[df["mean_pdf_proxy_score"].notna()].copy()
            if "invalid_count" not in df.columns:
                df["invalid_count"] = 0
            if "mean_render_mae" not in df.columns:
                df["mean_render_mae"] = 1e9
            df = df.sort_values(
                ["mean_pdf_proxy_score", "invalid_count", "mean_render_mae"],
                ascending=[False, True, True],
            )
            ranked_paths = [
                path for path in df["checkpoint"].tolist()
                if isinstance(path, str) and os.path.isdir(path)
            ]
            if ranked_paths:
                return ranked_paths[:ENSEMBLE_TOP_K]
        except Exception as e:
            print(f"Warning: failed to read checkpoint ensemble ranking from {scores_path}: {e}")

    return [default_path]


MODEL_PATHS_TO_LOAD = [ensure_loadable_checkpoint_dir(path) for path in resolve_ensemble_paths()]
print("Loading model paths:")
for _path in MODEL_PATHS_TO_LOAD:
    print(f"  - {_path}")

models = []
tokenizer = None
for _load_idx, _load_path in enumerate(MODEL_PATHS_TO_LOAD):
    model_obj, tok = FastLanguageModel.from_pretrained(
        model_name=_load_path,
        max_seq_length=4096,
        dtype=torch.bfloat16,
        load_in_4bit=INFERENCE_LOAD_IN_4BIT,
        device_map={"": 0},
        local_files_only=HF_LOCAL_FILES_ONLY,
    )
    model_obj = FastLanguageModel.for_inference(model_obj)
    model_obj.eval()
    models.append({"model": model_obj, "path": _load_path})
    if tokenizer is None:
        tokenizer = tok

alloc = torch.cuda.memory_allocated(0) / 1e9
total = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"  GPU 0: {torch.cuda.get_device_name(0)} — VRAM {alloc:.1f}/{total:.1f} GB after loading {len(models)} model(s)")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Fix generation config
_im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
_eos_ids = None
if _im_end_id is not None and _im_end_id != tokenizer.unk_token_id:
    _eos = tokenizer.eos_token_id
    if _eos is None:
        _eos_ids = [_im_end_id]
    elif isinstance(_eos, int):
        _eos_ids = [_eos, _im_end_id]
    else:
        _eos_ids = list(_eos) + [_im_end_id]

for _model_bundle in models:
    _model = _model_bundle["model"]
    _model.generation_config.temperature = None
    _model.generation_config.top_p = None
    _model.generation_config.top_k = None
    _model.generation_config.max_length = None
    if _eos_ids:
        _model.generation_config.eos_token_id = _eos_ids

bad_words_ids = None  # disabled — multi-token bad_words hangs with Transformers 5.x
svg_close_ids = tokenizer.encode("</svg>", add_special_tokens=False)

print(f"\nModel ready ✓")

# ── Load CLIP reranker (after main model to share VRAM accounting) ────────────
# CLIP ViT-B/32 is ~300 MB in fp16 — safe to load alongside a 3B model on ≥15 GB GPUs.
# For Kaggle offline runs: set CLIP_MODEL to a local path and HF_LOCAL_FILES_ONLY=1.
if USE_CLIP_RERANK:
    try:
        from transformers import CLIPModel, CLIPProcessor
        _clip_model_name = os.environ.get("CLIP_MODEL", "openai/clip-vit-base-patch32")
        _clip_model = CLIPModel.from_pretrained(
            _clip_model_name,
            local_files_only=HF_LOCAL_FILES_ONLY,
            torch_dtype=torch.float16,
        )
        _clip_model = _clip_model.to("cuda:0")
        _clip_model.eval()
        _clip_processor = CLIPProcessor.from_pretrained(
            _clip_model_name,
            local_files_only=HF_LOCAL_FILES_ONLY,
        )
        HAS_CLIP = True
        _alloc_after_clip = torch.cuda.memory_allocated(0) / 1e9
        _total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"CLIP reranker loaded ✓  ({_clip_model_name}, VRAM {_alloc_after_clip:.1f}/{_total_vram:.1f} GB)")
    except Exception as _clip_exc:
        print(f"CLIP reranker unavailable ({_clip_exc}). Using coverage-only rerank.")


# Test data loading.
test_df = pd.read_csv(TEST_CSV_PATH)
test_df["prompt"] = test_df["prompt"].fillna("").astype(str).str.strip()
print(f"Test rows  : {len(test_df)}")
print(f"Columns    : {test_df.columns.tolist()}")
print(test_df.head(3).to_string())


# Generation and scoring helpers.
from transformers import StoppingCriteria, StoppingCriteriaList

class StopOnSvgClose(StoppingCriteria):
    """Stop generation immediately after </svg> is emitted."""
    def __init__(self, ids):
        self.ids = list(ids)
    def __call__(self, input_ids, scores, **kwargs):
        if not self.ids or input_ids.shape[0] != 1: return False
        n = len(self.ids)
        return input_ids.shape[1] >= n and input_ids[0, -n:].tolist() == self.ids

greedy_stopper = StoppingCriteriaList([StopOnSvgClose(svg_close_ids)]) if svg_close_ids else None

def build_prompt(prompt_text):
    """Format chat template; truncate prompt for faster tokenisation."""
    p = prompt_text
    if len(p) > PROFILE["max_prompt_chars"]:
        p = p[:PROFILE["max_prompt_chars"]].rsplit(" ", 1)[0]
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": p}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def smart_fallback(prompt):
    """Return a prompt-hint-aligned fallback SVG (better than a plain square
    for competition scoring when all generation candidates fail)."""
    pl = prompt.lower()
    # Pick fill color from prompt
    color_map = {
        "red": "#e53935", "blue": "#1e88e5", "green": "#43a047",
        "yellow": "#fdd835", "orange": "#fb8c00", "purple": "#8e24aa",
        "pink": "#e91e63", "gray": "#757575", "grey": "#757575",
        "white": "#f5f5f5", "black": "#212121", "gold": "#ffc107",
        "brown": "#6d4c41", "teal": "#00897b", "cyan": "#00bcd4",
    }
    fill = next((v for k, v in color_map.items() if k in pl), "#212121")
    bg   = "#f5f5f5" if fill != "#f5f5f5" else "#212121"
    # Pick shape from prompt
    if any(k in pl for k in ("circle", "round", "dot", "ring", "ball")):
        shape = f'<circle cx="128" cy="128" r="80" fill="{fill}"/>'
    elif any(k in pl for k in ("triangle",)):
        shape = f'<polygon points="128,48 208,208 48,208" fill="{fill}"/>'
    elif any(k in pl for k in ("line", "stripe", "bar", "menu", "list")):
        shape = (f'<rect x="36" y="72" width="184" height="18" fill="{fill}"/>'
                 f'<rect x="36" y="119" width="184" height="18" fill="{fill}"/>'
                 f'<rect x="36" y="166" width="184" height="18" fill="{fill}"/>') 
    elif any(k in pl for k in ("star",)):
        shape = f'<polygon points="128,40 150,100 214,100 162,138 182,200 128,160 74,200 94,138 42,100 106,100" fill="{fill}"/>'
    else:
        shape = f'<rect x="40" y="40" width="176" height="176" rx="20" fill="{fill}"/>'
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" '
           f'viewBox="0 0 256 256">'
           f'<rect width="256" height="256" fill="{bg}"/>'
           f'{shape}</svg>')
    return normalize_svg(svg)

def score_svg(prompt, svg):
    """Fast pre-render score aligned toward validity plus grayscale geometry quality."""
    valid = is_valid_svg(svg)
    stats = svg_semantic_stats(svg) if valid else {
        "drawable_count": 0,
        "painted_count": 0,
        "visible_paint_count": 0,
        "distinct_colors": 0,
        "coverage": 0.0,
    }
    s = (1000.0 if valid else -400.0) + structural_score(svg) + prompt_bonus(prompt, svg)
    if valid:
        s += min(stats["visible_paint_count"], 8) * 1.5
        s += min(stats["distinct_colors"], 6) * 0.15
        s += min(stats["coverage"], 0.35) * 95.0
        if stats["coverage"] < 0.01:
            s -= 250.0
        elif stats["coverage"] < 0.03:
            s -= 120.0
        elif stats["coverage"] < 0.08:
            s -= 40.0
        # Compactness is only a small part of the real metric, so keep this light.
        is_complex = prompt_is_complex(prompt)
        length_target = 4200 if is_complex else 1400
        length_fit = math.exp(-abs(math.log((len(svg) + 50.0) / (length_target + 50.0))))
        s += length_fit * 3.0
    return s, valid, stats

def color_fidelity_bonus(prompt, rgb_arr):
    """Keep prompt-color hints as a light tie-breaker, not a primary metric."""
    if rgb_arr is None:
        return 0.0
    pl = prompt.lower()
    bonus = 0.0
    for color_name, (r, g, b) in COLOR_NAME_TO_RGB.items():
        if color_name not in pl:
            continue
        dist = np.sqrt(
            (rgb_arr[..., 0] - r / 255.0) ** 2
            + (rgb_arr[..., 1] - g / 255.0) ** 2
            + (rgb_arr[..., 2] - b / 255.0) ** 2
        )
        fraction = float(np.mean(dist < 0.22))
        bonus += fraction * 6.0
    return min(bonus, 8.0)


def clip_score_rendered(prompt, png_bytes):
    """Compute CLIP cosine similarity (as logit) between rendered SVG and prompt text.

    This is the primary semantic reranking signal: a candidate that CLIP agrees
    matches the prompt is more likely to resemble the hidden reference SVG.
    Returns 0.0 when CLIP is unavailable. Typical range: ~10–35 logit units.
    """
    if not HAS_CLIP or not HAS_RENDER or not png_bytes:
        return 0.0
    try:
        pil_img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        inputs = _clip_processor(
            text=[prompt],
            images=[pil_img],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
        with torch.no_grad():
            out = _clip_model(**inputs)
        return float(out.logits_per_image[0][0])
    except Exception:
        return 0.0


def apply_render_rerank(prompt, candidates):
    if not HAS_RENDER or not candidates:
        return
    is_complex = prompt_is_complex(prompt)
    target_non_white = 0.22 if is_complex else 0.14
    target_edge = 0.045 if is_complex else 0.020
    valid_candidates = [c for c in candidates if c["valid"]]
    if not valid_candidates:
        return
    top_candidates = sorted(valid_candidates, key=lambda c: c["score"], reverse=True)[:PROFILE["render_rerank_top_k"]]
    for cand in top_candidates:
        png = _render_svg_png_bytes(cand["svg"])
        if png is None:
            continue
        gray = _png_to_gray_arr(png)
        rgb_arr = _png_to_rgb_arr(png)
        edge_h = np.abs(np.diff(gray, axis=0))
        edge_v = np.abs(np.diff(gray, axis=1))
        edge_all = np.concatenate([edge_h.ravel(), edge_v.ravel()])
        edge_d = float(edge_all.mean())
        non_white = float(np.mean(gray < 0.95))
        edge_clarity = float(np.mean(edge_all > 4.0 * edge_d)) if edge_d > 1e-6 else 0.0
        cand["render_edge_density"] = round(edge_d, 5)
        cand["render_non_white"] = round(non_white, 5)
        if non_white < 0.008:
            cand["score"] -= 700.0
            continue
        coverage_fit = max(0.0, 1.0 - abs(non_white - target_non_white) / max(target_non_white, 1e-6))
        edge_fit = max(0.0, 1.0 - abs(edge_d - target_edge) / max(target_edge, 1e-6))
        cand["score"] += edge_d * 120.0
        cand["score"] += coverage_fit * 35.0
        cand["score"] += edge_fit * 18.0
        # Edge clarity: high ratio of strong-to-weak edges → crisp shapes → better EdgeF1
        if edge_clarity is not None:
            cand["score"] += edge_clarity * 22.0
        # Color fidelity is a light tie-breaker because the official visual score is grayscale.
        cand["score"] += color_fidelity_bonus(prompt, rgb_arr)
        # CLIP can still help reject nonsense, but the official metric is reference-based, not prompt-based.
        if HAS_CLIP:
            clip_s = clip_score_rendered(prompt, png)
            cand["clip_score"] = round(clip_s, 4)
            cand["score"] += clip_s * CLIP_RERANK_WEIGHT


def target_candidate_count(prompt, greedy_score, greedy_stats, num_models):
    total_target = PROFILE["complex_num_candidates"] if prompt_is_complex(prompt) else PROFILE["easy_num_candidates"]
    if (
        greedy_score < PROFILE["weak_greedy_score_threshold"]
        or greedy_stats.get("coverage", 0.0) < PROFILE["weak_greedy_coverage_threshold"]
        or greedy_stats.get("visible_paint_count", 0) < 1
    ):
        total_target = max(total_target, PROFILE["weak_greedy_num_candidates"])
    return max(1, -(-int(total_target) // max(1, num_models)))

print("Generation helpers loaded ✓")


def _normalize_train_svg_candidate(raw_svg):
    if not isinstance(raw_svg, str) or not raw_svg.strip():
        return ""
    svg = normalize_svg(raw_svg)
    if not svg:
        svg = normalize_svg(extract_svg(raw_svg))
    if not svg:
        return ""
    return rescale_svg(svg)


def _build_exact_train_retrieval_lookup():
    lookup = {}
    if not ENABLE_EXACT_TRAIN_RETRIEVAL:
        print("Exact-train retrieval disabled.")
        return lookup
    if not os.path.exists(TRAIN_CSV_PATH):
        print(f"Exact-train retrieval skipped: {TRAIN_CSV_PATH} not found.")
        return lookup

    test_prompts = set(test_df["prompt"].astype(str))
    if not test_prompts:
        return lookup

    try:
        train_df = pd.read_csv(TRAIN_CSV_PATH, usecols=["prompt", "svg"])
        train_df["prompt"] = train_df["prompt"].fillna("").astype(str).str.strip()
        matched = train_df[train_df["prompt"].isin(test_prompts)].copy()
        if matched.empty:
            print("Exact-train retrieval loaded ✓  (0 matching prompts)")
            return lookup

        total_rows = len(matched)
        for prompt, group in matched.groupby("prompt", sort=False):
            ranked = []
            seen_svgs = set()
            for raw_svg in group["svg"].fillna("").astype(str):
                svg = _normalize_train_svg_candidate(raw_svg)
                if not svg or svg in seen_svgs:
                    continue
                seen_svgs.add(svg)
                score, valid, stats = score_svg(prompt, svg)
                if not valid:
                    continue
                ranked.append((score, svg, stats))

            if not ranked:
                continue

            ranked.sort(key=lambda item: item[0], reverse=True)
            lookup[prompt] = [
                {
                    "idx": f"train_exact:{idx}",
                    "svg": svg,
                    "valid": True,
                    "score": score,
                    "stats": stats,
                    "render_edge_density": None,
                    "render_non_white": None,
                    "model_path": TRAIN_CSV_PATH,
                    "model_label": "train-exact",
                }
                for idx, (score, svg, stats) in enumerate(ranked[:EXACT_TRAIN_MAX_CANDIDATES])
            ]

        total_candidates = sum(len(items) for items in lookup.values())
        print(
            f"Exact-train retrieval loaded ✓  "
            f"({len(lookup)} prompts, {total_rows} matched train rows, {total_candidates} candidates)"
        )
    except Exception as exc:
        print(f"Exact-train retrieval unavailable ({exc}). Continuing without it.")
    return lookup


EXACT_TRAIN_RETRIEVAL = _build_exact_train_retrieval_lookup()


def exact_train_retrieval_candidates(prompt):
    templates = EXACT_TRAIN_RETRIEVAL.get(prompt, [])
    return [
        {
            **template,
            "score": float(template["score"]),
            "stats": dict(template["stats"]),
            "render_edge_density": None,
            "render_non_white": None,
        }
        for template in templates
    ]


# Inference loop with checkpoint-safe resume.
START_TIME = time.time()

# ── Resume from checkpoint if one exists ─────────────────────────────────────
results, audit = [], []
_done_ids = set()
if os.path.exists(CKPT_RESULTS):
    try:
        _ckpt_df  = pd.read_csv(CKPT_RESULTS)
        _audit_df = pd.read_csv(CKPT_AUDIT) if os.path.exists(CKPT_AUDIT) else pd.DataFrame()
        results   = _ckpt_df.to_dict("records")
        audit     = _audit_df.to_dict("records") if not _audit_df.empty else []
        _done_ids = set(_ckpt_df["id"].astype(str))
        print(f"[resume] {len(results)} rows already done. Resuming.")
    except Exception as e:
        print(f"[resume] Checkpoint load failed ({e}), starting fresh.")
else:
    print("[resume] No checkpoint found — starting fresh.")

_ckpt_lock      = threading.Lock()
_shared_results = results  # shared reference — extended by process_chunk
_shared_audit   = audit

def _save_checkpoint(results, audit):
    if not results:
        return
    try:
        ckpt_df = pd.DataFrame(results)
        if "svg" in ckpt_df.columns:
            ckpt_df["svg"] = ckpt_df["svg"].apply(normalize_svg_for_csv)
        ckpt_df.to_csv(CKPT_RESULTS, index=False)
        if audit:
            pd.DataFrame(audit).to_csv(CKPT_AUDIT, index=False)
    except Exception as e:
        print(f"[ckpt] Save failed: {e}")

# ── Per-GPU worker ────────────────────────────────────────────────────────────
def process_chunk(gpu_id, chunk):
    """Run the full inference pipeline for a list of (row_idx, row) on one GPU."""
    device = f"cuda:{gpu_id}"
    chunk_results, chunk_audit = [], []

    for row_idx, row in chunk:
        print(f"  GPU {gpu_id} | row {row_idx} | {row['prompt'][:50]}", flush=True)
        base_inp = None   # initialised here so the except block can reference it for retries
        candidates = []   # initialised here so audit len(candidates) is always valid
        try:
            torch.manual_seed(SEED + row_idx * 97)
            text = build_prompt(row["prompt"])
            base_inp = tokenizer(text, return_tensors="pt").to(device)
            candidates = []  # reset per row (also initialised to [] outside try for audit safety)

            for model_idx, model_bundle in enumerate(models):
                m = model_bundle["model"]
                model_label = os.path.basename(model_bundle["path"])

                # ── Greedy (candidate 0 for this model) ──────────────────────
                # Use higher token budget for complex prompts — long-context training
                # supports 4096 tokens but we were capping at 1024, truncating large SVGs.
                _is_complex_prompt = prompt_is_complex(row["prompt"])
                _greedy_max_tok = (
                    PROFILE.get("greedy_max_new_tokens_complex", PROFILE["greedy_max_new_tokens"])
                    if _is_complex_prompt
                    else PROFILE["greedy_max_new_tokens"]
                )
                _kw = dict(
                    max_new_tokens=_greedy_max_tok, do_sample=False,
                    repetition_penalty=PROFILE["repetition_penalty"],
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=m.generation_config.eos_token_id,
                    bad_words_ids=bad_words_ids or None,
                    stopping_criteria=greedy_stopper,
                )
                with torch.inference_mode():
                    try:
                        out = m.generate(**base_inp, use_cache=True, **_kw)
                    except RuntimeError:
                        out = m.generate(**base_inp, use_cache=False, **_kw)
                decoded = tokenizer.decode(out[0][base_inp["input_ids"].shape[1]:], skip_special_tokens=True)
                svg_g = rescale_svg(normalize_svg(extract_svg(decoded)))
                score_g, valid_g, stats_g = score_svg(row["prompt"], svg_g)
                candidates.append({
                    "idx": f"m{model_idx}:0",
                    "svg": svg_g,
                    "valid": valid_g,
                    "score": score_g,
                    "stats": stats_g,
                    "render_edge_density": None,
                    "render_non_white": None,
                    "model_path": model_bundle["path"],
                    "model_label": model_label,
                })

                # ── Sampling candidates for this model ───────────────────────
                is_complex = prompt_is_complex(row["prompt"])
                base_temp = PROFILE["temperature"] if is_complex else max(0.52, PROFILE["temperature"] - 0.12)
                # Calibrate temperature from the greedy result quality:
                #  · Strong greedy → tighten sampling to refine, not randomise
                #  · Weak greedy   → widen sampling to escape the bad mode
                if score_g >= 1050:
                    temp = max(0.40, base_temp - 0.14)
                elif score_g <= 950:
                    temp = min(0.95, base_temp + 0.18)
                else:
                    temp = base_temp
                max_tok = PROFILE["sample_max_tokens_complex"] if is_complex else PROFILE["sample_max_tokens_simple"]
                target_count = target_candidate_count(row["prompt"], score_g, stats_g, len(models))
                remaining_s, cand_idx, batch_num = max(0, target_count - 1), 1, 0
                while remaining_s > 0:
                    bs = min(PROFILE["batch_size"], remaining_s)
                    t = temp + (0.08 if batch_num % 2 else 0.0)
                    _sk = dict(
                        max_new_tokens=max_tok, do_sample=True,
                        temperature=float(t), top_p=PROFILE["top_p"],
                        top_k=PROFILE["top_k"],
                        repetition_penalty=PROFILE["repetition_penalty"],
                        num_return_sequences=bs,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=m.generation_config.eos_token_id,
                        bad_words_ids=bad_words_ids or None,
                    )
                    with torch.inference_mode():
                        try:
                            outs = m.generate(**base_inp, use_cache=True, **_sk)
                        except RuntimeError:
                            outs = m.generate(**base_inp, use_cache=False, **_sk)
                    il = base_inp["input_ids"].shape[1]
                    for seq in outs:
                        dec = tokenizer.decode(seq[il:], skip_special_tokens=True)
                        svg_s = rescale_svg(normalize_svg(extract_svg(dec)))
                        sc, vl, st = score_svg(row["prompt"], svg_s)
                        candidates.append({
                            "idx": f"m{model_idx}:{cand_idx}",
                            "svg": svg_s,
                            "valid": vl,
                            "score": sc,
                            "stats": st,
                            "render_edge_density": None,
                            "render_non_white": None,
                            "model_path": model_bundle["path"],
                            "model_label": model_label,
                        })
                        cand_idx += 1
                    remaining_s -= bs
                    batch_num += 1

            exact_candidates = exact_train_retrieval_candidates(row["prompt"])
            if exact_candidates:
                candidates.extend(exact_candidates)
                print(
                    f"  [retrieval] GPU {gpu_id} | row {row_idx} | "
                    f"added {len(exact_candidates)} exact-train candidate(s)",
                    flush=True,
                )

            # Dedupe exact repeats before expensive reranking.
            deduped, seen_svgs = [], set()
            for cand in candidates:
                if not cand["svg"] or cand["svg"] in seen_svgs:
                    continue
                seen_svgs.add(cand["svg"])
                deduped.append(cand)
            candidates = deduped if deduped else candidates

            # ── Render rerank + pick best ─────────────────────────────────────
            apply_render_rerank(row["prompt"], candidates)
            best = max(candidates, key=lambda c: c["score"])
            used_fallback = False
            if not best["valid"]:
                raise ValueError("best candidate was invalid after reranking")
        except Exception as e:
            print(f"  [row-error] GPU {gpu_id} | row {row_idx} | {e}", flush=True)
            # Before resorting to the generic fallback SVG, try re-generating with
            # progressively higher temperatures. Even a mediocre model-generated SVG
            # scores much better on SSIM than the black-square fallback.
            best = None
            used_fallback = False
            if base_inp is not None and models:
                for _retry_temp in (0.90, 0.96, 1.05):
                    try:
                        _retry_kw = dict(
                            max_new_tokens=min(
                                PROFILE.get("greedy_max_new_tokens_complex", 2048), 2048
                            ),
                            do_sample=True,
                            temperature=float(_retry_temp),
                            top_p=0.98,
                            top_k=100,
                            repetition_penalty=PROFILE["repetition_penalty"],
                            num_return_sequences=1,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=models[0]["model"].generation_config.eos_token_id,
                        )
                        with torch.inference_mode():
                            _retry_out = models[0]["model"].generate(
                                **base_inp, use_cache=True, **_retry_kw
                            )
                        _retry_dec = tokenizer.decode(
                            _retry_out[0][base_inp["input_ids"].shape[1]:],
                            skip_special_tokens=True,
                        )
                        _retry_svg = rescale_svg(normalize_svg(extract_svg(_retry_dec)))
                        _retry_sc, _retry_vl, _retry_st = score_svg(row["prompt"], _retry_svg)
                        if _retry_vl:
                            best = {
                                "idx": f"retry:{_retry_temp}",
                                "svg": _retry_svg,
                                "valid": True,
                                "score": _retry_sc,
                                "stats": _retry_st,
                                "render_edge_density": None,
                                "render_non_white": None,
                                "model_path": models[0]["path"],
                                "model_label": f"retry@{_retry_temp}",
                            }
                            print(
                                f"  [retry] GPU {gpu_id} | row {row_idx} | "
                                f"temp={_retry_temp} → valid SVG rescued",
                                flush=True,
                            )
                            break
                    except Exception as _re:
                        print(
                            f"  [retry] GPU {gpu_id} | row {row_idx} | "
                            f"temp={_retry_temp} failed: {_re}",
                            flush=True,
                        )
            if best is None:
                retrieval_candidates = exact_train_retrieval_candidates(row["prompt"])
                if retrieval_candidates:
                    apply_render_rerank(row["prompt"], retrieval_candidates)
                    best = max(retrieval_candidates, key=lambda c: c["score"])
                    candidates = retrieval_candidates
                    if best["valid"]:
                        print(
                            f"  [retrieval-rescue] GPU {gpu_id} | row {row_idx} | "
                            f"used exact-train match",
                            flush=True,
                        )
                if best is not None and not best["valid"]:
                    best = None
            if best is None:
                fallback_svg = smart_fallback(row["prompt"])
                best = {
                    "idx": -1,
                    "svg": fallback_svg,
                    "valid": True,
                    "score": -1.0,
                    "stats": svg_semantic_stats(fallback_svg),
                    "render_edge_density": None,
                    "render_non_white": None,
                    "model_path": "fallback",
                    "model_label": "fallback",
                }
                used_fallback = True

        chunk_results.append({"id": row["id"], "svg": best["svg"]})
        chunk_audit.append({
            "id"            : row["id"],
            "prompt"        : row["prompt"][:120],
            "best_idx"      : best["idx"],
            "best_score"    : round(best["score"], 2),
            "best_valid"    : best["valid"],
            "used_fallback" : used_fallback,
            "num_candidates": len(candidates),
            "svg_chars"     : len(best["svg"]),
            "best_coverage" : round(best.get("stats", {}).get("coverage", 0.0), 5),
            "best_drawables": best.get("stats", {}).get("drawable_count", 0),
            "best_colors"   : best.get("stats", {}).get("distinct_colors", 0),
            "best_model"    : best.get("model_label"),
            "render_non_white"   : best.get("render_non_white"),
            "render_edge_density": best.get("render_edge_density"),
            "best_clip_score"    : best.get("clip_score"),
            "gpu_id"             : gpu_id,
        })

        # Checkpoint every CKPT_EVERY rows across all GPU threads combined
        if len(chunk_results) % CKPT_EVERY == 0:
            with _ckpt_lock:
                _shared_results.extend(chunk_results)
                _shared_audit.extend(chunk_audit)
                _save_checkpoint(_shared_results, _shared_audit)
                print(f"  [ckpt] GPU {gpu_id} — {len(_shared_results)} rows saved")
                chunk_results, chunk_audit = [], []

    return chunk_results, chunk_audit

# ── Split rows across GPUs (round-robin) and run in parallel ─────────────────
_pending = [(idx, row) for idx, row in test_df.iterrows()
            if str(row["id"]) not in _done_ids]
chunk = _pending[SHARD_ID::NUM_SHARDS]

print(f"Shard {SHARD_ID}/{NUM_SHARDS}: {len(chunk)} rows on cuda:0 "
      f"(physical GPU {os.environ.get('CUDA_VISIBLE_DEVICES', '0')})", flush=True)

try:
    r, a = process_chunk(0, chunk)
except Exception as e:
    print(f"  Shard {SHARD_ID} FAILED: {e}")
    r, a = [], []

results.extend(r)
audit.extend(a)
valid_so_far = sum(1 for x in audit if x["best_valid"])
print(f"  Shard {SHARD_ID} done — {len(results)} rows total, valid={valid_so_far}", flush=True)

# ── Summary ───────────────────────────────────────────────────────────────────
_save_checkpoint(results, audit)
total_min      = (time.time() - START_TIME) / 60
valid_count    = sum(1 for r in audit if r["best_valid"])
fallback_count = sum(1 for r in audit if r.get("used_fallback"))
print(f"\nDone — {len(results)} rows in {total_min:.1f} min "
      f"(~{total_min:.1f} min effective/GPU)")
print(f"Valid: {valid_count}/{len(audit)}  |  Fallback: {fallback_count}")
print(f"Checkpoint saved → {CKPT_RESULTS}")


# Final CSV writes.
if not results:
    print("WARNING: no results to save — process_chunk produced no output")
    raise SystemExit(1)
submission_df = pd.DataFrame(results)[["id", "svg"]]

# Rescale any SVGs whose content is packed into a small coordinate region
submission_df["svg"] = submission_df["svg"].apply(rescale_svg)
submission_df["svg"] = submission_df["svg"].apply(normalize_svg_for_csv)

# Final safety pass — every row must carry a valid SVG
invalid_mask = submission_df["svg"].apply(lambda s: not is_valid_svg(s))
if invalid_mask.any():
    print(f"WARNING: {invalid_mask.sum()} invalid SVGs replaced with fallback")
    submission_df.loc[invalid_mask, "svg"] = FALLBACK_SVG

submission_df.to_csv(SUBMISSION_CSV_OUT, index=False)
print(f"Submission saved → {SUBMISSION_CSV_OUT}  ({len(submission_df)} rows)")
print(submission_df.head(3).to_string())

# Save audit log
if audit:
    pd.DataFrame(audit).to_csv(AUDIT_CSV_OUT, index=False)
    print(f"Audit log saved  → {AUDIT_CSV_OUT}")

# Quick stats
svg_lens = submission_df["svg"].str.len()
print(f"\nSVG length — mean: {svg_lens.mean():.0f}  "
      f"median: {svg_lens.median():.0f}  max: {svg_lens.max()}")


# Optional visual spot-check for notebook use.
try:
    from IPython.display import SVG, display  # type: ignore[import]
    _in_jupyter = True
except ImportError:
    SVG, display = None, None  # type: ignore[assignment]
    _in_jupyter = False

sample_ids = random.sample(range(len(submission_df)), min(5, len(submission_df)))
for i in sample_ids:
    row = submission_df.iloc[i]
    prompt_preview = test_df.loc[test_df["id"] == row["id"], "prompt"].values
    print(f"[{i}] {prompt_preview[0][:80] if len(prompt_preview) else 'N/A'}")
    print(f"     chars={len(row['svg'])}")
    if _in_jupyter and display is not None and SVG is not None:
        display(SVG(data=row["svg"]))
    print()
