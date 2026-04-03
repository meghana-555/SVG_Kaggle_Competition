#!/usr/bin/env python3
# Single-node training entry script.
# Default usage:
#   python train_qwen2p5coder3b.py
# Optional multi-GPU usage:
#   torchrun --nproc_per_node=<N> train_qwen2p5coder3b.py

import os
import sys
import subprocess
from pathlib import Path

if "COLAB_GPU" in os.environ:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
        "unsloth", "transformers", "accelerate", "safetensors", "sentencepiece",
        "pandas", "numpy", "cairosvg", "pillow", "trl", "peft", "bitsandbytes"])

import torch

print(f"Python: {sys.version}")
print(f"Torch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")
if torch.cuda.is_available():
    n_gpus = torch.cuda.device_count()
    print(f"GPUs available: {n_gpus}")
    for _i in range(n_gpus):
        _mem = torch.cuda.get_device_properties(_i).total_memory / 1e9
        print(f"  GPU {_i}: {torch.cuda.get_device_name(_i)} — {_mem:.1f} GB")


import shutil

DEFAULT_BASE_MODEL_NAME = "Qwen/Qwen2.5-Coder-3B-Instruct"
DEFAULT_TRAIN_MODEL_NAME = "unsloth/qwen2.5-coder-3b-instruct-bnb-4bit"
DEFAULT_GRADIENT_ACCUMULATION_STEPS = (
    4 if int(os.environ.get("WORLD_SIZE", "1")) == 1 else 1
)

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
            path / "lora-adapter",
            path / "model-merged-bf16",
            path / "model-merged",
        ]
    )


DEFAULT_RUN_NAME = REPO_ROOT.name if looks_like_run_root(REPO_ROOT) else "Qwen2.5-Coder-3B-unsloth-h100-Run1"
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
BASE_MODEL_NAME_OR_PATH = os.environ.get(
    "BASE_MODEL_NAME_OR_PATH",
    os.environ.get("BASE_MODEL_NAME", DEFAULT_BASE_MODEL_NAME),
).strip()
TRAIN_MODEL_NAME_OR_PATH = os.environ.get(
    "TRAIN_MODEL_NAME_OR_PATH",
    os.environ.get("TRAIN_MODEL_NAME", DEFAULT_TRAIN_MODEL_NAME),
).strip()
TRAINING_STAGE = os.environ.get("TRAINING_STAGE", "main").strip().lower()
RUN_ONLY_POST_TRAIN_SELECTION = os.environ.get("RUN_ONLY_POST_TRAIN_SELECTION", "0").strip() == "1"
DEFER_POST_TRAIN_SELECTION = os.environ.get("DEFER_POST_TRAIN_SELECTION", "0").strip() == "1"
SKIP_HOLDOUT_SELECTION = os.environ.get("SKIP_HOLDOUT_SELECTION", "0").strip() == "1"
MAIN_LR = float(os.environ.get("MAIN_LR", "1e-4"))
LONG_CONTEXT_LR = float(os.environ.get("LONG_CONTEXT_LR", "7e-5"))
LONG_CONTEXT_EPOCHS = max(1, int(os.environ.get("LONG_CONTEXT_EPOCHS", "1")))
PER_DEVICE_BATCH_SIZE = max(1, int(os.environ.get("PER_DEVICE_BATCH_SIZE", "8")))
GRADIENT_ACCUMULATION_STEPS = max(
    1,
    int(
        os.environ.get(
            "GRADIENT_ACCUMULATION_STEPS",
            str(DEFAULT_GRADIENT_ACCUMULATION_STEPS),
        )
    ),
)
MAX_EVAL_SAMPLES = max(128, int(os.environ.get("MAX_EVAL_SAMPLES", "1000")))
CHECKPOINT_PROXY_EVAL_SAMPLES = max(128, int(os.environ.get("CHECKPOINT_PROXY_EVAL_SAMPLES", "1000")))
CHECKPOINT_PROXY_BATCH_SIZE = max(1, int(os.environ.get("CHECKPOINT_PROXY_BATCH_SIZE", "8")))
GENERATION_AUDIT_BATCH_SIZE = max(1, int(os.environ.get("GENERATION_AUDIT_BATCH_SIZE", "8")))
CHECKPOINT_STEPS = max(100, int(os.environ.get("CHECKPOINT_STEPS", "500")))
SAVE_TOTAL_LIMIT = max(2, int(os.environ.get("SAVE_TOTAL_LIMIT", "10")))
MAX_CKPTS_TO_EVAL = max(2, int(os.environ.get("MAX_CKPTS_TO_EVAL", "8")))
HF_HOME = os.environ.setdefault("HF_HOME", f"{LOCAL_ROOT}/hf")
HF_HUB_CACHE = os.environ.setdefault("HF_HUB_CACHE", f"{HF_HOME}/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", HF_HUB_CACHE)
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("UNSLOTH_STABLE_DOWNLOADS", "1")
HF_LOCAL_FILES_ONLY = os.environ.get("HF_LOCAL_FILES_ONLY", "0").strip() == "1"
ALLOW_LONG_CONTEXT_RESUME = os.environ.get("ALLOW_LONG_CONTEXT_RESUME", "1").strip() == "1"

# Primary data and artifact locations.
TRAIN_CSV_PATH = f"{ROOT}/Data/train.csv"
SAVE_DIR = f"{ROOT}/lora-adapter/"
MERGE_OUTPUT = f"{ROOT}/model-merged/"
 
# Keep hot data on local disk. Cache is stage-specific so the long-context pass
# never reuses the main-stage dataset by accident.
DATASET_CACHE = f"{LOCAL_ROOT}/{TRAINING_STAGE}-dataset-cache/"
DATASET_CACHE_READY = f"{LOCAL_ROOT}/{TRAINING_STAGE}-dataset-cache.ready"
LOCAL_CKPT = (
    f"{LOCAL_ROOT}/checkpoints/"
    if TRAINING_STAGE == "main"
    else f"{LOCAL_ROOT}/{TRAINING_STAGE}-checkpoints/"
)
FINAL_ADAPTER_DIR = (
    f"{LOCAL_ROOT}/final-adapters/"
    if TRAINING_STAGE == "main"
    else f"{LOCAL_ROOT}/{TRAINING_STAGE}-final-adapters/"
)
DRIVE_CHECKPOINT_DIR = (
    SAVE_DIR
    if TRAINING_STAGE == "main"
    else os.path.join(SAVE_DIR, f"{TRAINING_STAGE}-checkpoints")
)
MAIN_STAGE_FINAL_ADAPTER_DIR = f"{LOCAL_ROOT}/final-adapters/"

os.makedirs(LOCAL_CKPT, exist_ok=True)
os.makedirs(FINAL_ADAPTER_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(DRIVE_CHECKPOINT_DIR, exist_ok=True)
os.makedirs(HF_HUB_CACHE, exist_ok=True)


# GPU / process setup for single-GPU or torchrun launches.

NUM_GPUS = torch.cuda.device_count()
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
IS_MAIN_PROCESS = (LOCAL_RANK == 0)

print(f"Detected GPUs : {NUM_GPUS}")
print(f"Local rank    : {LOCAL_RANK}")
print(f"Main process  : {IS_MAIN_PROCESS}")
print(f"Root          : {ROOT}")
print(f"Local root    : {LOCAL_ROOT}")
print(f"HF cache      : {HF_HUB_CACHE}")
print(f"HF local only : {HF_LOCAL_FILES_ONLY}")
print(f"Long-context resume enabled: {ALLOW_LONG_CONTEXT_RESUME}")

if NUM_GPUS < 1:
    import warnings
    warnings.warn(
        "No CUDA GPUs detected. This training script requires a GPU runtime."
    )


# Config and imports.
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import inspect
import json
import random
import re
import time
import xml.etree.ElementTree as ET
import torch.distributed as _dist
from importlib.metadata import PackageNotFoundError, version as pkg_version
from io import BytesIO

import numpy as np
import pandas as pd
import unsloth  # must be imported first for all optimizations
from unsloth import FastLanguageModel
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# ── Epoch-by-epoch mode ──────────────────────────────────────────────────────
# Set EPOCH_TO_TRAIN to 1, 2, or 3 to train exactly one epoch at a time.
#   Epoch 1 → fresh start.
#   Epoch 2 → resumes from the epoch-1 checkpoint and trains until epoch 2 total.
#   Epoch 3 → resumes from the epoch-2 checkpoint and trains until epoch 3 total.
# Set to None (default) to run all epochs in one go.
EPOCH_TO_TRAIN = None


def collect_package_versions():
    packages = [
        "torch",
        "transformers",
        "trl",
        "datasets",
        "peft",
        "accelerate",
        "bitsandbytes",
        "unsloth",
        "pandas",
        "numpy",
        "cairosvg",
        "Pillow",
    ]
    versions = {}
    for package in packages:
        try:
            versions[package] = pkg_version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


PACKAGE_VERSIONS = collect_package_versions()

SYSTEM_PROMPT = (
    "You generate SVG images from text descriptions.\n\n"
    "Requirements:\n"
    "- Return only one complete SVG document: <svg>...</svg>\n"
    "- Output valid XML that renders without errors\n"
    "- Use a 256x256 canvas with width, height, and viewBox\n"
    "- Keep the drawing visible, centered, and large enough to read clearly\n"
    "- Match the prompt's subject, shape, and colors as closely as possible\n"
    "- Prefer simple structure, but include enough detail to avoid generic icons\n"
    "- No scripts, animation, or external references"
)


def format_chat_messages(prompt, svg=None):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    if svg is not None:
        messages.append({"role": "assistant", "content": svg})
    return messages


CONFIG = {
    "model_name": BASE_MODEL_NAME_OR_PATH,
    "train_model_name_or_path": TRAIN_MODEL_NAME_OR_PATH,
    "training_stage": TRAINING_STAGE,
    # LoRA
    "lora_r": 96,
    "lora_alpha": 192,
    "lora_dropout": 0.05,
    # Training
    "learning_rate": MAIN_LR,
    "num_train_epochs": 2,
    "per_device_train_batch_size": PER_DEVICE_BATCH_SIZE,
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "warmup_ratio": 0.05,
    "weight_decay": 0.01,
    "lr_scheduler_type": "cosine",
    "packing": False,  # always False — DataCollatorForCompletionOnlyLM requires packing disabled
    "group_by_length": True,
    # Validation / runtime guardrails
    "eval_fraction": 0.02,
    "max_eval_samples": MAX_EVAL_SAMPLES,
    "logging_steps": 20,
    "save_total_limit": SAVE_TOTAL_LIMIT,
    # Batch-size benchmark on real examples
    "benchmark_batch_sizes": [6, 8, 10, 12, 14],
    "benchmark_sample_size": 64,
    "benchmark_steps": 2,
    # Post-train generation audit / Kaggle inference defaults
    "generation_eval_samples": 64,
    "checkpoint_proxy_eval_samples": CHECKPOINT_PROXY_EVAL_SAMPLES,
    "checkpoint_proxy_batch_size": CHECKPOINT_PROXY_BATCH_SIZE,
    "generation_audit_batch_size": GENERATION_AUDIT_BATCH_SIZE,
    "max_checkpoints_to_eval": MAX_CKPTS_TO_EVAL,
    "max_new_tokens_eval": 1536,        # proxy ranking only; longer cap reduces truncation on complex SVGs
    "kaggle_num_candidates": 8,
    "kaggle_temperature": 0.68,
    "kaggle_top_p": 0.95,
    "kaggle_top_k": 64,
    "kaggle_repetition_penalty": 1.05,
    "kaggle_adaptive_extra_candidates": 7,
    "kaggle_min_greedy_score_keep": 1030.0,
    "kaggle_enable_clip_rerank": False,
    # Data
    "train_csv": TRAIN_CSV_PATH,
    "min_svg_chars": 50,
    "max_svg_chars": 16000,
    "dedupe_pairs": True,
    "quality_filter": False,     # keep one-color / simple icons; only filter truly degenerate SVGs
    "color_balance_oversample_factor": 0.30,
    "color_balance_power": 0.5,
    "require_masked_loss": True,
    "long_context_top_fraction": 0.30,
    # Output
    "output_dir": LOCAL_CKPT,
    "final_adapter_dir": FINAL_ADAPTER_DIR,
    "drive_checkpoint_dir": DRIVE_CHECKPOINT_DIR,
    "inference_artifact_dir": os.path.join(SAVE_DIR, "kaggle_artifacts"),
}

if TRAINING_STAGE == "main":
    CONFIG.update(
        {
            "max_seq_length": 2048,          # full context for complex SVGs
            "num_train_epochs": 3,           # extra epoch with 1.5B trains fast
            "learning_rate": MAIN_LR,
            "benchmark_batch_sizes": [6, 8, 10, 12, 14],
        }
    )
elif TRAINING_STAGE == "long_context":
    CONFIG.update(
        {
            "max_seq_length": 4096,
            "num_train_epochs": LONG_CONTEXT_EPOCHS,
            "learning_rate": LONG_CONTEXT_LR,
            "benchmark_batch_sizes": [4, 6, 8, 10, 12],
        }
    )
else:
    raise ValueError(f"Unknown TRAINING_STAGE: {TRAINING_STAGE}")

print(f"Seed: {SEED}")
print(f"Training stage: {TRAINING_STAGE}")
if TRAINING_STAGE == "main":
    print("Run plan: stage 1/2. Train the main model for 3 epochs at 2048 context, then run long_context as a separate follow-up pass.")
elif TRAINING_STAGE == "long_context":
    print(f"Run plan: stage 2/2. Resume from the best main-stage checkpoint and run {LONG_CONTEXT_EPOCHS} follow-up epoch(s) at 4096 context.")
_n = torch.cuda.device_count()
effective_batch_size = (
    CONFIG["per_device_train_batch_size"]
    * max(1, _n)
    * CONFIG["gradient_accumulation_steps"]
)
print(f"GPUs: {_n}")
for _i in range(_n):
    print(f"  GPU {_i}: {torch.cuda.get_device_name(_i)} — "
          f"{torch.cuda.get_device_properties(_i).total_memory / 1e9:.1f} GB")
print(f"Max sequence length: {CONFIG['max_seq_length']}")
print(f"Initial batch size: {CONFIG['per_device_train_batch_size']}")
print(f"Gradient accumulation: {CONFIG['gradient_accumulation_steps']}")
print(f"Effective global batch size: {effective_batch_size}")
print(f"Train model path: {CONFIG['train_model_name_or_path']}")
print(f"Merge base path: {CONFIG['model_name']}")
print(f"Packing enabled: {CONFIG['packing']}")
print(f"TF32 matmul: {getattr(torch.backends.cuda.matmul, 'allow_tf32', None)}")
print("Package versions:")
for package_name, package_version in PACKAGE_VERSIONS.items():
    print(f"  {package_name}: {package_version}")


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

UNSAFE_VALUE_TOKENS = (
    "javascript:",
    "vbscript:",
    "data:",
    "http://",
    "https://",
    "file:",
    "<script",
    "</script",
    "@import",
)

SVG_REGEX = re.compile(r"<svg[\s\S]*?</svg>", flags=re.IGNORECASE)
DRAWABLE_SVG_TAGS = {
    "path", "rect", "circle", "ellipse", "line", "polygon", "polyline", "text", "use"
}
NON_PAINT_VALUES = {"none", "transparent", "inherit", ""}
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


def local_name(tag):
    return tag.split("}")[-1] if "}" in tag else tag


def compact_attr_value(value):
    return re.sub(r"\s+", "", str(value or ""))


def has_unsafe_value(value):
    lowered = str(value or "").lower()
    return any(token in lowered for token in UNSAFE_VALUE_TOKENS)


def is_internal_reference(value):
    compact = compact_attr_value(value)
    return bool(compact) and (
        compact.startswith("#")
        or re.fullmatch(r"url\(#[-\w:.]+\)", compact) is not None
    )


def sanitize_style_attr(style_text):
    cleaned_parts = []
    for raw_part in str(style_text or "").split(";"):
        part = raw_part.strip()
        if not part:
            continue
        if ":" not in part:
            return None
        name, value = part.split(":", 1)
        name = name.strip().lower()
        value = value.strip()
        if not name or name.startswith("on") or has_unsafe_value(value):
            return None
        if "url(" in value.lower() and not is_internal_reference(value):
            return None
        cleaned_parts.append(f"{name}:{value}")
    return ";".join(cleaned_parts)


def sanitize_text_content(tag, text):
    if text is None:
        return None
    cleaned = str(text).strip()
    if not cleaned:
        return None
    if has_unsafe_value(cleaned):
        return None
    if tag == "style" and "url(" in cleaned.lower() and "url(#" not in compact_attr_value(cleaned).lower():
        return None
    if len(cleaned) > 2000:
        return None
    return cleaned


def sanitize_attributes(elem):
    safe_attrib = {}
    for attr_name, attr_value in list(elem.attrib.items()):
        local_attr = local_name(attr_name)
        value = str(attr_value).strip()
        if not value:
            continue
        if local_attr.startswith("on"):
            continue
        if local_attr not in ALLOWED_SVG_ATTRS:
            continue
        if has_unsafe_value(value):
            continue
        if local_attr in {"href", "xlink:href", "clip-path", "mask", "filter", "marker-start", "marker-mid", "marker-end"}:
            if not is_internal_reference(value):
                continue
        if "url(" in value.lower() and not is_internal_reference(value):
            continue
        if local_attr == "style":
            value = sanitize_style_attr(value)
            if not value:
                continue
        if local_attr in {"id", "class", "font-family"} and len(value) > 120:
            continue
        safe_attrib[local_attr] = value
    elem.attrib.clear()
    elem.attrib.update(safe_attrib)


def prune_disallowed_children(elem):
    for child in list(elem):
        child.tag = local_name(child.tag)
        if child.tag not in ALLOWED_SVG_TAGS:
            elem.remove(child)
            continue
        sanitize_attributes(child)
        child.text = sanitize_text_content(child.tag, child.text)
        child.tail = None
        prune_disallowed_children(child)


def normalize_svg(svg_text):
    if not svg_text or not str(svg_text).strip():
        return ""
    try:
        root = ET.fromstring(str(svg_text).strip())
    except ET.ParseError:
        return ""

    root.tag = local_name(root.tag)
    if root.tag != "svg":
        return ""

    sanitize_attributes(root)
    prune_disallowed_children(root)
    root.text = None
    root.tail = None
    root.set("xmlns", "http://www.w3.org/2000/svg")
    root.set("width", "256")
    root.set("height", "256")
    root.set("viewBox", "0 0 256 256")

    path_count = 0
    for elem in root.iter():
        elem.tag = local_name(elem.tag)
        if elem.tag not in ALLOWED_SVG_TAGS:
            return ""
        sanitize_attributes(elem)
        elem.text = sanitize_text_content(elem.tag, elem.text)
        elem.tail = None
        if elem.tag == "path":
            path_count += 1

    if path_count > 256:
        return ""

    svg_str = ET.tostring(root, encoding="unicode", short_empty_elements=True)
    svg_str = svg_str.replace("ns0:", "").replace(":ns0", "")
    svg_str = re.sub(r'\s+xmlns:ns0="[^"]*"', "", svg_str)
    svg_str = re.sub(r">\s+<", "><", svg_str).strip()
    if len(svg_str) > 16000:
        return ""
    return svg_str


def parse_svg_color(value):
    compact = compact_attr_value(value).lower()
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


def is_visible_paint(value):
    compact = compact_attr_value(value).lower()
    if not compact or compact in NON_PAINT_VALUES:
        return False
    rgb = parse_svg_color(compact)
    if rgb is None:
        return True
    return min(rgb) < 245


def svg_semantic_stats(svg_text):
    try:
        root = ET.fromstring(svg_text)
    except Exception:
        return {
            "drawable_count": 0,
            "painted_count": 0,
            "visible_paint_count": 0,
            "distinct_colors": 0,
        }

    drawable_count = 0
    painted_count = 0
    visible_paint_count = 0
    distinct_colors = set()

    for elem in root.iter():
        if local_name(elem.tag) not in DRAWABLE_SVG_TAGS:
            continue
        drawable_count += 1
        elem_has_paint = False
        elem_has_visible_paint = False

        for attr in ("fill", "stroke"):
            value = (elem.get(attr) or "").strip()
            if not value or compact_attr_value(value).lower() in NON_PAINT_VALUES:
                continue
            elem_has_paint = True
            distinct_colors.add(compact_attr_value(value).lower())
            elem_has_visible_paint = elem_has_visible_paint or is_visible_paint(value)

        style = elem.get("style") or ""
        for part in style.split(";"):
            if ":" not in part:
                continue
            key, value = part.split(":", 1)
            if key.strip().lower() not in {"fill", "stroke"}:
                continue
            value = value.strip()
            if not value or compact_attr_value(value).lower() in NON_PAINT_VALUES:
                continue
            elem_has_paint = True
            distinct_colors.add(compact_attr_value(value).lower())
            elem_has_visible_paint = elem_has_visible_paint or is_visible_paint(value)

        painted_count += int(elem_has_paint)
        visible_paint_count += int(elem_has_visible_paint)

    return {
        "drawable_count": drawable_count,
        "painted_count": painted_count,
        "visible_paint_count": visible_paint_count,
        "distinct_colors": len(distinct_colors),
    }


def extract_svg(text):
    match = SVG_REGEX.search(text or "")
    if match:
        return match.group(0).strip()
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
    open_g = partial.lower().count("<g ") + partial.lower().count("<g>") - partial.lower().count("</g>")
    partial += "</g>" * max(0, min(open_g, 10))
    partial += "</svg>"
    return partial.strip()


def is_valid_svg(svg_text):
    normalized = normalize_svg(svg_text)
    if not normalized:
        return False
    try:
        root = ET.fromstring(normalized)
        if local_name(root.tag) != "svg":
            return False
        path_count = sum(1 for elem in root.iter() if local_name(elem.tag) == "path")
        stats = svg_semantic_stats(normalized)
        return (
            len(normalized) <= 16000
            and path_count <= 256
            and stats["drawable_count"] > 0
            and stats["visible_paint_count"] > 0
        )
    except ET.ParseError:
        return False


def describe_lengths(series, label):
    vals = series.astype(str).str.len()
    print(
        f"{label}: mean={vals.mean():.1f}, median={vals.median():.0f}, "
        f"p90={vals.quantile(0.90):.0f}, p95={vals.quantile(0.95):.0f}, max={vals.max():.0f}"
    )


SPLIT_COMPLEXITY_HINTS = [
    "illustration", "detailed", "complex", "many", "multiple", "pattern",
    "icon set", "logo", "scene", "stack", "five", "several",
]
COLOR_BALANCE_TOKENS = [
    "black", "white", "red", "green", "blue", "yellow", "orange",
    "purple", "pink", "gray", "grey", "brown", "gold", "silver",
]


def prompt_complexity_score(prompt):
    prompt_lower = str(prompt or "").lower()
    score = sum(token in prompt_lower for token in SPLIT_COMPLEXITY_HINTS)
    score += int(len(str(prompt or "")) >= 120)
    return min(score, 2)


def svg_complexity_score(svg_text):
    svg_text = str(svg_text or "")
    score = 0
    score += int(len(svg_text) >= 1800)
    score += int(svg_text.count("<path") >= 12)
    score += int(svg_text.count("<g") >= 4)
    score += int((svg_text.count("<linearGradient") + svg_text.count("<radialGradient")) > 0)
    score += int(
        svg_text.count("<circle") + svg_text.count("<rect") + svg_text.count("<polygon") >= 8
    )
    return min(score, 2)


def prompt_color_tokens(prompt):
    prompt_lower = str(prompt or "").lower()
    return tuple(
        color
        for color in COLOR_BALANCE_TOKENS
        if re.search(rf"\b{re.escape(color)}\b", prompt_lower)
    )


def apply_color_balancing(frame):
    oversample_factor = float(CONFIG.get("color_balance_oversample_factor", 0.0))
    if oversample_factor <= 0 or frame.empty:
        return frame

    color_sets = frame["prompt"].apply(prompt_color_tokens)
    color_counts = {}
    for colors in color_sets:
        for color in colors:
            color_counts[color] = color_counts.get(color, 0) + 1

    if not color_counts:
        print("Color balancing skipped: no color-tagged prompts found.")
        return frame

    max_count = max(color_counts.values())
    power = float(CONFIG.get("color_balance_power", 0.5))
    weights = []
    for colors in color_sets:
        if not colors:
            weights.append(1.0)
            continue
        rarity_boost = max((max_count / max(color_counts[color], 1)) ** power for color in colors)
        weights.append(max(1.0, rarity_boost))

    extra_rows = int(len(frame) * oversample_factor)
    if extra_rows < 1:
        return frame

    extra = frame.sample(
        n=extra_rows,
        replace=True,
        weights=weights,
        random_state=SEED,
    ).reset_index(drop=True)
    balanced = pd.concat([frame, extra], ignore_index=True)
    top_counts = sorted(color_counts.items(), key=lambda kv: kv[1], reverse=True)[:8]
    print(
        f"Color balancing added {extra_rows:,} oversampled rows "
        f"({len(frame):,} -> {len(balanced):,})."
    )
    print(f"Prompt color counts: {top_counts}")
    return balanced


def build_stratified_eval_split(frame, eval_size):
    frame = frame.sort_values("combined_chars").reset_index(drop=True).copy()
    num_length_bins = min(5, max(2, int(frame["combined_chars"].nunique())))
    frame["length_bin"] = pd.qcut(
        frame["combined_chars"],
        q=num_length_bins,
        labels=False,
        duplicates="drop",
    ).astype(int)
    frame["prompt_complexity_bin"] = frame["prompt"].apply(prompt_complexity_score).astype(int)
    frame["svg_complexity_bin"] = frame["svg_norm"].apply(svg_complexity_score).astype(int)
    frame["split_bucket"] = frame.apply(
        lambda row: f"L{row['length_bin']}_P{row['prompt_complexity_bin']}_S{row['svg_complexity_bin']}",
        axis=1,
    )

    bucket_sizes = frame["split_bucket"].value_counts().sort_index()
    desired = bucket_sizes / bucket_sizes.sum() * eval_size
    bucket_eval = {}
    remainders = {}
    minima = {}
    capacities = {}

    for bucket, size in bucket_sizes.items():
        min_count = 1 if size >= 20 else 0
        count = max(min_count, int(np.floor(desired[bucket])))
        count = min(count, size - 1)
        bucket_eval[bucket] = count
        remainders[bucket] = float(desired[bucket] - np.floor(desired[bucket]))
        minima[bucket] = min_count
        capacities[bucket] = size - 1

    allocated = sum(bucket_eval.values())
    if allocated < eval_size:
        for bucket in sorted(remainders, key=lambda b: (remainders[b], bucket_sizes[b]), reverse=True):
            if allocated >= eval_size:
                break
            if bucket_eval[bucket] < capacities[bucket]:
                bucket_eval[bucket] += 1
                allocated += 1
    elif allocated > eval_size:
        for bucket in sorted(remainders, key=lambda b: (remainders[b], bucket_sizes[b])):
            if allocated <= eval_size:
                break
            removable = bucket_eval[bucket] - minima[bucket]
            if removable > 0:
                bucket_eval[bucket] -= 1
                allocated -= 1

    eval_indices = []
    for bucket, count in bucket_eval.items():
        if count <= 0:
            continue
        bucket_df = frame[frame["split_bucket"] == bucket].sort_values("combined_chars")
        positions = np.linspace(0, len(bucket_df) - 1, num=count, dtype=int)
        eval_indices.extend(bucket_df.iloc[np.unique(positions)].index.tolist())

    if len(eval_indices) < eval_size:
        remaining = [idx for idx in frame.index.tolist() if idx not in set(eval_indices)]
        extra_needed = eval_size - len(eval_indices)
        if extra_needed > 0:
            extra_positions = np.linspace(0, len(remaining) - 1, num=extra_needed, dtype=int)
            eval_indices.extend([remaining[pos] for pos in np.unique(extra_positions)])

    eval_indices = sorted(set(eval_indices))[:eval_size]
    eval_df = frame.loc[eval_indices].copy().reset_index(drop=True)
    train_df = frame.drop(index=eval_indices).reset_index(drop=True)
    print(
        f"Stratified eval split across {frame['split_bucket'].nunique()} buckets "
        f"using length + prompt/SVG complexity."
    )
    return train_df, eval_df


print("SVG utilities loaded.")


# Data loading and preparation.
print("--- Loading data ---")
train_df = pd.read_csv(CONFIG["train_csv"])
train_df["prompt"] = train_df["prompt"].fillna("").astype(str).str.strip()
train_df["svg"] = train_df["svg"].fillna("").astype(str)
train_df["prompt_norm"] = train_df["prompt"].str.lower().str.replace(r"\s+", " ", regex=True)
print(f"Raw rows: {len(train_df)}")

describe_lengths(train_df["prompt"], "Prompt chars")
describe_lengths(train_df["svg"], "Raw SVG chars")

train_df = train_df[
    (train_df["svg"].str.len() >= CONFIG["min_svg_chars"])
    & (train_df["svg"].str.len() <= CONFIG["max_svg_chars"])
    & (train_df["prompt"].str.len() > 0)
].reset_index(drop=True)

train_df["svg_norm"] = train_df["svg"].apply(normalize_svg)
train_df = train_df[train_df["svg_norm"].str.len() > 0].reset_index(drop=True)
train_df["valid"] = train_df["svg_norm"].apply(is_valid_svg)
train_df = train_df[train_df["valid"]].reset_index(drop=True)

# ── Optional visual quality filter ─────────────────────────────────────
# Disabled by default. The public train set contains many good single-color or
# single-shape icons, so we only use this if we explicitly want a narrower,
# higher-detail subset for an ablation.
def is_visually_rich(svg_text):
    try:
        root = ET.fromstring(svg_text)
    except Exception:
        return False

    DRAWABLE = {"rect", "circle", "ellipse", "path", "line",
                "polygon", "polyline", "text", "use"}
    NON_COLORS = {"none", "transparent", "inherit", ""}

    elements = [
        e for e in root.iter()
        if local_name(e.tag) in DRAWABLE
    ]
    if len(elements) < 2:
        return False

    colors = set()
    for e in elements:
        for attr in ("fill", "stroke"):
            val = (e.get(attr) or "").strip().lower()
            if val and val not in NON_COLORS:
                colors.add(val)
        style = e.get("style") or ""
        for part in style.split(";"):
            if ":" in part:
                k, v = part.split(":", 1)
                if k.strip() in ("fill", "stroke"):
                    val = v.strip().lower()
                    if val and val not in NON_COLORS:
                        colors.add(val)
    if len(colors) < 2:
        return False

    return True

if CONFIG.get("quality_filter", True):
    before_quality = len(train_df)
    train_df["visually_rich"] = train_df["svg_norm"].apply(is_visually_rich)
    train_df = train_df[train_df["visually_rich"]].reset_index(drop=True)
    print(f"Quality filter removed {before_quality - len(train_df):,} low-fidelity rows "
          f"({len(train_df):,} remain)")

before_dedupe = len(train_df)
if CONFIG["dedupe_pairs"]:
    train_df = train_df.drop_duplicates(subset=["prompt_norm", "svg_norm"]).reset_index(drop=True)
    print(f"Removed duplicates: {before_dedupe - len(train_df)}")

train_df["combined_chars"] = train_df["prompt"].str.len() + train_df["svg_norm"].str.len()
train_df = train_df.sort_values("combined_chars").reset_index(drop=True)

print(f"After filtering: {len(train_df)}")
describe_lengths(train_df["svg_norm"], "Normalized SVG chars")

raw_eval_size = max(128, int(len(train_df) * CONFIG["eval_fraction"]))
eval_size = min(CONFIG["max_eval_samples"], raw_eval_size)
eval_size = min(eval_size, len(train_df) - 1)
if eval_size < 1:
    raise ValueError("Need at least 2 valid rows to create a train/eval split.")

train_df, eval_df = build_stratified_eval_split(train_df, eval_size=eval_size)

if TRAINING_STAGE == "long_context":
    keep_fraction = float(CONFIG["long_context_top_fraction"])
    train_keep = min(len(train_df), max(4096, int(len(train_df) * keep_fraction)))
    eval_keep = min(len(eval_df), max(128, int(len(eval_df) * max(keep_fraction, 0.25))))
    train_df = train_df.nlargest(train_keep, "combined_chars").reset_index(drop=True)
    eval_df = eval_df.nlargest(eval_keep, "combined_chars").reset_index(drop=True)
    print(
        f"Long-context stage keeps {len(train_df)} train / {len(eval_df)} eval rows "
        f"from the longest examples (top {keep_fraction:.0%})."
    )

train_df = apply_color_balancing(train_df)
train_df = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

print(f"Train rows: {len(train_df)}")
print(f"Eval rows: {len(eval_df)}")
print("Sample prompt:")
print(train_df.iloc[0]["prompt"])

LONG_CONTEXT_START_SOURCE = None


try:
    import cairosvg
    from PIL import Image
    HAS_RASTER_METRIC = True
except Exception as e:
    cairosvg = None
    Image = None
    HAS_RASTER_METRIC = False
    print(f"Raster metric unavailable: {e}")


def raster_rgba(svg_text):
    png = cairosvg.svg2png(
        bytestring=svg_text.encode("utf-8"),
        output_width=256,
        output_height=256,
    )
    rgba = Image.open(BytesIO(png)).convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return np.asarray(Image.alpha_composite(background, rgba), dtype=np.float32) / 255.0


def grayscale(arr):
    return 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]


def edge_map(gray):
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1])
    gy[1:, :] = np.abs(gray[1:, :] - gray[:-1, :])
    return np.sqrt(gx**2 + gy**2)


def svg_tag_counts(svg_text):
    root = ET.fromstring(svg_text)
    out = {}
    for elem in root.iter():
        tag = local_name(elem.tag)
        out[tag] = out.get(tag, 0) + 1
    return tuple(sorted(out.items()))


@lru_cache(maxsize=4096)
def cached_reference_proxy(svg_text):
    arr_ref = raster_rgba(svg_text).astype(np.float16)
    gray_ref = grayscale(arr_ref).astype(np.float16)
    return {
        "arr_ref": arr_ref,
        "gray_ref": gray_ref,
        "tag_counts": svg_tag_counts(svg_text),
        "svg_len": len(svg_text),
    }


def ssim_proxy(gray_a, gray_b):
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_a = float(gray_a.mean())
    mu_b = float(gray_b.mean())
    var_a = float(gray_a.var())
    var_b = float(gray_b.var())
    cov_ab = float(((gray_a - mu_a) * (gray_b - mu_b)).mean())
    num = (2 * mu_a * mu_b + c1) * (2 * cov_ab + c2)
    den = (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
    if den <= 0:
        return 0.0
    return float(max(0.0, min(1.0, num / den)))


def edge_f1_proxy(gray_a, gray_b):
    edge_a = edge_map(gray_a) > 0.08
    edge_b = edge_map(gray_b) > 0.08
    tp = float(np.logical_and(edge_a, edge_b).sum())
    fp = float(np.logical_and(edge_a, ~edge_b).sum())
    fn = float(np.logical_and(~edge_a, edge_b).sum())
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def tag_structural_proxy_counts(counts_a, counts_b):
    ca = dict(counts_a)
    cb = dict(counts_b)
    keys = sorted(set(ca) | set(cb))
    if not keys:
        return 1.0
    diff = sum(abs(ca.get(k, 0) - cb.get(k, 0)) for k in keys)
    total = sum(ca.get(k, 0) + cb.get(k, 0) for k in keys)
    if total == 0:
        return 1.0
    return float(np.exp(-6.0 * diff / total))


def pdf_proxy_score(svg_pred, svg_ref):
    if not HAS_RASTER_METRIC:
        return {}
    try:
        arr_pred = raster_rgba(svg_pred)
        gray_pred = grayscale(arr_pred)
        pred_tag_counts = svg_tag_counts(svg_pred)
        ref_proxy = cached_reference_proxy(svg_ref)
    except Exception:
        return {}

    arr_ref = ref_proxy["arr_ref"]
    gray_ref = ref_proxy["gray_ref"]
    ssim_val = ssim_proxy(gray_pred, gray_ref)
    edge_val = edge_f1_proxy(gray_pred, gray_ref)
    visual_val = 0.7 * ssim_val + 0.3 * edge_val
    structural_val = tag_structural_proxy_counts(pred_tag_counts, ref_proxy["tag_counts"])
    compactness_val = float(np.exp(-abs(np.log((len(svg_pred) + 50) / (ref_proxy["svg_len"] + 50)))))
    pdf_proxy = float(
        (visual_val ** 0.85)
        * (structural_val ** 0.12)
        * (compactness_val ** 0.03)
    )
    render_mae = float(np.abs(arr_pred - arr_ref).mean())
    return {
        "visual_proxy": visual_val,
        "ssim_proxy": ssim_val,
        "edge_f1_proxy": edge_val,
        "structural_proxy": structural_val,
        "compactness_proxy": compactness_val,
        "render_mae": render_mae,
        "pdf_proxy_score": pdf_proxy,
    }


def evaluate_model_on_holdout(checkpoint_path, frame, num_gpus=NUM_GPUS):
    if not torch.cuda.is_available():
        raise RuntimeError("Checkpoint proxy evaluation requires CUDA.")

    available_gpus = torch.cuda.device_count()
    num_gpus = max(1, min(int(num_gpus), available_gpus))

    gpu_models = []
    for gpu_id in range(num_gpus):
        with torch.cuda.device(gpu_id):
            m, tok = FastLanguageModel.from_pretrained(
                model_name=checkpoint_path,
                max_seq_length=CONFIG["max_seq_length"],
                dtype=torch.bfloat16,
                load_in_4bit=False,
                device_map={"": gpu_id},
                local_files_only=HF_LOCAL_FILES_ONLY,
            )
            m = FastLanguageModel.for_inference(m)
            m.eval()
            m.config.use_cache = True
        gpu_models.append((m, tok))

    frame_rows = list(frame.itertuples())
    chunks = [frame_rows[i::num_gpus] for i in range(num_gpus)]

    def _run_chunk(gpu_id, chunk_rows):
        if not chunk_rows:
            return []
        m, tok = gpu_models[gpu_id]
        device = f"cuda:{gpu_id}"
        torch.cuda.set_device(gpu_id)
        generated_rows = []
        for row_batch in batched_rows(chunk_rows, CONFIG["checkpoint_proxy_batch_size"]):
            texts = [
                tok.apply_chat_template(
                    format_chat_messages(row.prompt),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for row in row_batch
            ]
            orig_padding_side = tok.padding_side
            tok.padding_side = "left"
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            try:
                inputs = tok(texts, return_tensors="pt", padding=True).to(device)
                with torch.inference_mode():
                    output_ids = m.generate(
                        **inputs,
                        max_new_tokens=CONFIG["max_new_tokens_eval"],
                        do_sample=False,
                        use_cache=False,
                    )
                prompt_len = inputs["input_ids"].shape[1]
                for i, row in enumerate(row_batch):
                    raw_text = tok.decode(output_ids[i][prompt_len:], skip_special_tokens=True)
                    generated_rows.append((row, raw_text))
            finally:
                tok.padding_side = orig_padding_side
            if torch.cuda.is_available():
                with torch.cuda.device(gpu_id):
                    torch.cuda.empty_cache()

        rows = []
        for row, raw_text in generated_rows:
            svg = normalize_svg(extract_svg(raw_text))
            valid = is_valid_svg(svg)
            metrics = pdf_proxy_score(svg, row.svg_norm) if valid else {}
            rows.append(
                {
                    "id": row.id,
                    "prompt": row.prompt,
                    "valid": valid,
                    "svg_chars": len(svg),
                    "raw_preview": raw_text[:240],
                    **metrics,
                }
            )
        return rows

    all_rows = run_gpu_workers_concurrently(chunks, _run_chunk)

    for m, _ in gpu_models:
        del m
    torch.cuda.empty_cache()

    valid_pdf_scores = [r["pdf_proxy_score"] for r in all_rows if r.get("pdf_proxy_score") is not None]
    render_maes = [r["render_mae"] for r in all_rows if r.get("render_mae") is not None]
    invalid_count = sum(not row["valid"] for row in all_rows)
    return {
        "rows": all_rows,
        "mean_pdf_proxy_score": (sum(valid_pdf_scores) / len(valid_pdf_scores)) if valid_pdf_scores else None,
        "mean_render_mae": (sum(render_maes) / len(render_maes)) if render_maes else None,
        "invalid_count": invalid_count,
    }


def list_checkpoint_dirs(include_legacy=False):
    checkpoint_by_name = {}
    search_dirs = [CONFIG["drive_checkpoint_dir"], CONFIG["output_dir"]]
    if include_legacy or TRAINING_STAGE == "main":
        search_dirs.extend(
            [
                f"{LOCAL_ROOT}/checkpoints/",
                SAVE_DIR,
            ]
        )
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
        for entry in os.listdir(search_dir):
            path = os.path.join(search_dir, entry)
            if os.path.isdir(path) and entry.startswith("checkpoint-"):
                checkpoint_by_name[entry] = path
    return [
        checkpoint_by_name[name]
        for name in sorted(checkpoint_by_name, key=lambda value: int(value.split("-")[-1]))
    ]


def resolve_checkpoint_path(checkpoint_ref, checkpoint_dirs):
    if not checkpoint_ref:
        return None
    if os.path.isdir(checkpoint_ref):
        return checkpoint_ref

    checkpoint_name = os.path.basename(str(checkpoint_ref).rstrip(os.sep))
    for path in checkpoint_dirs:
        if os.path.basename(path) == checkpoint_name:
            return path

    search_dirs = [CONFIG["output_dir"], CONFIG["drive_checkpoint_dir"]]
    if TRAINING_STAGE == "main":
        search_dirs.extend([SAVE_DIR, f"{LOCAL_ROOT}/checkpoints/"])
    for search_dir in search_dirs:
        candidate = os.path.join(search_dir, checkpoint_name)
        if os.path.isdir(candidate):
            return candidate

    return None


def load_json_if_exists(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def batched_rows(rows, batch_size):
    for start in range(0, len(rows), batch_size):
        yield rows[start:start + batch_size]


def run_gpu_workers_concurrently(chunks, worker_fn):
    outputs = [[] for _ in range(len(chunks))]
    with ThreadPoolExecutor(max_workers=max(1, len(chunks))) as executor:
        future_to_gpu = {
            executor.submit(worker_fn, gpu_id, chunk_rows): gpu_id
            for gpu_id, chunk_rows in enumerate(chunks)
            if chunk_rows
        }
        for future, gpu_id in future_to_gpu.items():
            outputs[gpu_id] = future.result()

    rows = []
    for gpu_rows in outputs:
        rows.extend(gpu_rows)
    return rows


def infer_trainer_best_checkpoint(checkpoint_dirs, include_legacy_state=False):
    state_paths = [
        os.path.join(CONFIG["output_dir"], "trainer_state.json"),
        os.path.join(CONFIG["drive_checkpoint_dir"], "trainer_state.json"),
    ]
    if include_legacy_state:
        state_paths.append(os.path.join(SAVE_DIR, "trainer_state.json"))
    state_paths.extend(
        os.path.join(checkpoint_path, "trainer_state.json")
        for checkpoint_path in reversed(checkpoint_dirs)
    )

    for state_path in state_paths:
        state = load_json_if_exists(state_path)
        best_checkpoint = resolve_checkpoint_path(
            (state or {}).get("best_model_checkpoint"),
            checkpoint_dirs,
        )
        if best_checkpoint:
            return best_checkpoint

    return checkpoint_dirs[-1] if checkpoint_dirs else None


def infer_long_context_start_source():
    adapter_candidates = [
        SAVE_DIR,
        MAIN_STAGE_FINAL_ADAPTER_DIR,
    ]
    for candidate in adapter_candidates:
        if os.path.exists(os.path.join(candidate, "adapter_config.json")):
            return candidate

    manifest_paths = [
        os.path.join(SAVE_DIR, "kaggle_artifacts", "run_summary.json"),
        os.path.join(SAVE_DIR, "kaggle_artifacts", "kaggle_inference_manifest.json"),
    ]
    for manifest_path in manifest_paths:
        selected_checkpoint = (load_json_if_exists(manifest_path) or {}).get("selected_checkpoint_by_pdf_proxy")
        resolved = resolve_checkpoint_path(selected_checkpoint, [])
        if resolved:
            return resolved

    legacy_checkpoint_dirs = []
    for search_dir in [f"{LOCAL_ROOT}/checkpoints/", SAVE_DIR]:
        if not os.path.exists(search_dir):
            continue
        for entry in os.listdir(search_dir):
            path = os.path.join(search_dir, entry)
            if os.path.isdir(path) and entry.startswith("checkpoint-"):
                legacy_checkpoint_dirs.append(path)
    legacy_checkpoint_dirs = sorted(
        set(legacy_checkpoint_dirs),
        key=lambda value: int(os.path.basename(value).split("-")[-1]),
    )
    return infer_trainer_best_checkpoint(legacy_checkpoint_dirs, include_legacy_state=True)


def load_training_adapter_weights(model, adapter_source):
    from peft import load_peft_weights, set_peft_model_state_dict

    print(f"Initializing trainable adapter weights from: {adapter_source}")
    adapter_weights = load_peft_weights(adapter_source, device="cpu")
    load_result = set_peft_model_state_dict(
        model,
        adapter_weights,
        adapter_name="default",
    )
    missing_keys = getattr(load_result, "missing_keys", None) or []
    unexpected_keys = getattr(load_result, "unexpected_keys", None) or []
    if missing_keys:
        print(f"Adapter load missing keys: {len(missing_keys)}")
    if unexpected_keys:
        print(f"Adapter load unexpected keys: {len(unexpected_keys)}")


if TRAINING_STAGE == "long_context":
    LONG_CONTEXT_START_SOURCE = infer_long_context_start_source()
    if not LONG_CONTEXT_START_SOURCE:
        raise RuntimeError(
            "long_context requires a completed main-stage adapter or selected checkpoint. "
            "Run main post-train selection first."
        )
    print(f"Long-context initialization source: {LONG_CONTEXT_START_SOURCE}")


def copy_final_adapter_to_drive():
    for fname in os.listdir(CONFIG["final_adapter_dir"]):
        src_path = os.path.join(CONFIG["final_adapter_dir"], fname)
        dst_path = os.path.join(SAVE_DIR, fname)
        if os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)


def run_post_train_selection_and_audit(eval_frame, trainer_best_checkpoint=None, ddp_active=False):
    if SKIP_HOLDOUT_SELECTION:
        print("--- Saving adapter + merge prep (holdout selection skipped) ---")
    else:
        print("--- Selecting checkpoint + saving adapter + running holdout audit ---")
    os.makedirs(CONFIG["inference_artifact_dir"], exist_ok=True)

    fallback_templates = {
        "default": (
            "<svg xmlns='http://www.w3.org/2000/svg' width='256' height='256' viewBox='0 0 256 256'>"
            "<rect width='256' height='256' fill='white'/>"
            "<rect x='40' y='40' width='176' height='176' rx='24' fill='black'/>"
            "</svg>"
        ),
        "circle": (
            "<svg xmlns='http://www.w3.org/2000/svg' width='256' height='256' viewBox='0 0 256 256'>"
            "<rect width='256' height='256' fill='white'/>"
            "<circle cx='128' cy='128' r='72' fill='black'/>"
            "</svg>"
        ),
        "square": (
            "<svg xmlns='http://www.w3.org/2000/svg' width='256' height='256' viewBox='0 0 256 256'>"
            "<rect width='256' height='256' fill='white'/>"
            "<rect x='56' y='56' width='144' height='144' fill='black'/>"
            "</svg>"
        ),
        "lines": (
            "<svg xmlns='http://www.w3.org/2000/svg' width='256' height='256' viewBox='0 0 256 256'>"
            "<rect width='256' height='256' fill='white'/>"
            "<rect x='36' y='58' width='184' height='12' fill='black'/>"
            "<rect x='36' y='94' width='156' height='12' fill='black'/>"
            "<rect x='36' y='130' width='204' height='12' fill='black'/>"
            "<rect x='36' y='166' width='140' height='12' fill='black'/>"
            "</svg>"
        ),
        "star": (
            "<svg xmlns='http://www.w3.org/2000/svg' width='256' height='256' viewBox='0 0 256 256'>"
            "<rect width='256' height='256' fill='white'/>"
            "<polygon points='128,36 150,96 214,96 162,134 182,196 128,158 74,196 94,134 42,96 106,96' fill='black'/>"
            "</svg>"
        ),
    }

    checkpoint_eval_size = min(CONFIG["checkpoint_proxy_eval_samples"], len(eval_frame))
    checkpoint_eval_positions = np.linspace(0, len(eval_frame) - 1, num=checkpoint_eval_size, dtype=int)
    checkpoint_eval_positions = np.unique(checkpoint_eval_positions)
    checkpoint_eval_df = eval_frame.iloc[checkpoint_eval_positions].copy().reset_index(drop=True)

    checkpoint_dirs = list_checkpoint_dirs()
    selected_checkpoint = resolve_checkpoint_path(trainer_best_checkpoint, checkpoint_dirs)
    if selected_checkpoint is None:
        selected_checkpoint = infer_trainer_best_checkpoint(checkpoint_dirs)

    checkpoint_proxy_rows = []
    if len(checkpoint_dirs) > CONFIG["max_checkpoints_to_eval"] and not SKIP_HOLDOUT_SELECTION:
        print(
            f"Skipping first {len(checkpoint_dirs) - CONFIG['max_checkpoints_to_eval']} checkpoints "
            f"(evaluating last {CONFIG['max_checkpoints_to_eval']} only)"
        )
        checkpoint_dirs = checkpoint_dirs[-CONFIG["max_checkpoints_to_eval"]:]

    if SKIP_HOLDOUT_SELECTION:
        if selected_checkpoint:
            print(
                "Skipping checkpoint proxy selection and holdout audit. "
                f"Using trainer best checkpoint: {os.path.basename(selected_checkpoint)}"
            )
        else:
            print("Skipping checkpoint proxy selection and holdout audit.")
    elif HAS_RASTER_METRIC and checkpoint_dirs:
        print(f"Evaluating {len(checkpoint_dirs)} checkpoint(s) on {len(checkpoint_eval_df)} holdout prompts")
        if "model" in globals():
            try:
                del globals()["model"]
            except Exception:
                pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for checkpoint_path in checkpoint_dirs:
            checkpoint_name = os.path.basename(checkpoint_path)
            eval_gpus = 1 if ddp_active else NUM_GPUS
            print(f"  -> {checkpoint_name} (loading on {eval_gpus} GPUs)")
            try:
                proxy_result = evaluate_model_on_holdout(
                    checkpoint_path,
                    checkpoint_eval_df,
                    num_gpus=eval_gpus,
                )
                checkpoint_proxy_rows.append(
                    {
                        "checkpoint": checkpoint_path,
                        "checkpoint_name": checkpoint_name,
                        "mean_pdf_proxy_score": proxy_result["mean_pdf_proxy_score"],
                        "mean_render_mae": proxy_result["mean_render_mae"],
                        "invalid_count": proxy_result["invalid_count"],
                    }
                )
                print(
                    f"     pdf_proxy={proxy_result['mean_pdf_proxy_score']!s} | "
                    f"invalid={proxy_result['invalid_count']} | "
                    f"render_mae={proxy_result['mean_render_mae']!s}"
                )
            except Exception as e:
                checkpoint_proxy_rows.append(
                    {
                        "checkpoint": checkpoint_path,
                        "checkpoint_name": checkpoint_name,
                        "mean_pdf_proxy_score": None,
                        "mean_render_mae": None,
                        "invalid_count": None,
                        "error": str(e),
                    }
                )
                print(f"     failed: {e}")

        scored_rows = [row for row in checkpoint_proxy_rows if row.get("mean_pdf_proxy_score") is not None]
        if scored_rows:
            best_row = max(
                scored_rows,
                key=lambda row: (
                    row["mean_pdf_proxy_score"],
                    -(row["invalid_count"] or 0),
                    -(row["mean_render_mae"] if row["mean_render_mae"] is not None else 1e9),
                ),
            )
            selected_checkpoint = best_row["checkpoint"]
            print(f"Selected checkpoint by pdf_proxy_score: {best_row['checkpoint_name']}")
        else:
            print("No checkpoint proxy scores available; falling back to trainer best checkpoint.")

    if not selected_checkpoint:
        raise RuntimeError("No checkpoint available for post-train selection.")

    print(f"Loading selected checkpoint: {selected_checkpoint}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=selected_checkpoint,
        max_seq_length=CONFIG["max_seq_length"],
        dtype=torch.bfloat16,
        load_in_4bit=False,
        device_map={"": 0},
        local_files_only=HF_LOCAL_FILES_ONLY,
    )
    model = FastLanguageModel.for_inference(model)
    model.eval()
    model.config.use_cache = True

    model.save_pretrained(CONFIG["final_adapter_dir"])
    tokenizer.save_pretrained(CONFIG["final_adapter_dir"])
    with open(os.path.join(CONFIG["final_adapter_dir"], "training_config.json"), "w") as f:
        json.dump(CONFIG, f, indent=2, default=str)
    copy_final_adapter_to_drive()

    manifest = {
        "system_prompt": SYSTEM_PROMPT,
        "base_model": CONFIG["model_name"],
        "train_model_name_or_path": CONFIG["train_model_name_or_path"],
        "training_stage": CONFIG["training_stage"],
        "max_seq_length": CONFIG["max_seq_length"],
        "package_versions": PACKAGE_VERSIONS,
        "skip_holdout_selection": SKIP_HOLDOUT_SELECTION,
        "selected_checkpoint_by_pdf_proxy": selected_checkpoint,
        "checkpoint_proxy_eval_samples": CONFIG["checkpoint_proxy_eval_samples"],
        "generation": {
            "num_candidates": CONFIG["kaggle_num_candidates"],
            "max_new_tokens": CONFIG["max_new_tokens_eval"],
            "temperature": CONFIG["kaggle_temperature"],
            "top_p": CONFIG["kaggle_top_p"],
            "top_k": CONFIG["kaggle_top_k"],
            "repetition_penalty": CONFIG["kaggle_repetition_penalty"],
        },
        "adaptive_generation": {
            "extra_candidates": CONFIG["kaggle_adaptive_extra_candidates"],
            "min_greedy_score_keep": CONFIG["kaggle_min_greedy_score_keep"],
            "enable_clip_rerank": CONFIG["kaggle_enable_clip_rerank"],
        },
        "render_rerank": {
            "enabled": True,
            "coverage_target_simple": 0.14,
            "coverage_target_complex": 0.22,
            "edge_density_target_simple": 0.020,
            "edge_density_target_complex": 0.045,
            "center_weight": 2.5,
            "border_penalty_weight": 5.0,
            "render_weight": 18.0,
        },
        "kaggle_profiles": {
            "fast": {
                "num_candidates": 3,
                "adaptive_extra_candidates": 2,
                "temperature": 0.65,
                "top_p": 0.90,
                "top_k": 40,
                "repetition_penalty": 1.03,
                "min_greedy_score_keep": 1004.0,
                "enable_clip_rerank": False,
                "clip_rerank_top_candidates": 0,
                "clip_weight": 0.0,
            },
            "balanced": {
                "num_candidates": CONFIG["kaggle_num_candidates"],
                "adaptive_extra_candidates": CONFIG["kaggle_adaptive_extra_candidates"],
                "temperature": CONFIG["kaggle_temperature"],
                "top_p": CONFIG["kaggle_top_p"],
                "top_k": CONFIG["kaggle_top_k"],
                "repetition_penalty": CONFIG["kaggle_repetition_penalty"],
                "min_greedy_score_keep": CONFIG["kaggle_min_greedy_score_keep"],
                "enable_clip_rerank": False,
                "clip_rerank_top_candidates": 3,
                "clip_weight": 35.0,
            },
            "high_score": {
                "num_candidates": 8,
                "adaptive_extra_candidates": 7,
                "temperature": 0.78,
                "top_p": 0.94,
                "top_k": 64,
                "repetition_penalty": 1.06,
                "min_greedy_score_keep": 1011.0,
                "enable_clip_rerank": True,
                "clip_rerank_top_candidates": 3,
                "clip_weight": 40.0,
            },
        },
        "reranker": {
            "prefer_valid_svg": True,
            "prefer_balanced_length": [160, 9000],
            "max_paths": 256,
        },
        "fallback_templates": fallback_templates,
    }
    manifest_path = os.path.join(CONFIG["inference_artifact_dir"], "kaggle_inference_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    package_versions_path = os.path.join(CONFIG["inference_artifact_dir"], "package_versions.json")
    with open(package_versions_path, "w") as f:
        json.dump(PACKAGE_VERSIONS, f, indent=2)

    eval_export = eval_frame[["id", "prompt", "svg_norm"]].rename(columns={"svg_norm": "reference_svg"})
    eval_export_path = os.path.join(CONFIG["inference_artifact_dir"], "holdout_eval_prompts.csv")
    eval_export.to_csv(eval_export_path, index=False)

    fallback_svg_path = os.path.join(CONFIG["inference_artifact_dir"], "fallback.svg")
    with open(fallback_svg_path, "w") as f:
        f.write(fallback_templates["default"])

    fallback_templates_path = os.path.join(CONFIG["inference_artifact_dir"], "fallback_templates.json")
    with open(fallback_templates_path, "w") as f:
        json.dump(fallback_templates, f, indent=2)

    checkpoint_proxy_path = os.path.join(CONFIG["inference_artifact_dir"], "checkpoint_proxy_scores.csv")
    pd.DataFrame(checkpoint_proxy_rows).to_csv(checkpoint_proxy_path, index=False)

    audit_rows = []
    invalid_count = 0
    pdf_scores = []
    render_maes = []
    audit_elapsed_min = 0.0
    audit_path = os.path.join(SAVE_DIR, "eval_generation_audit.json")
    audit_csv_path = os.path.join(CONFIG["inference_artifact_dir"], "eval_generation_audit.csv")

    if not SKIP_HOLDOUT_SELECTION:
        audit_wall_start = time.time()
        audit_df = eval_frame.sample(
            n=min(CONFIG["generation_eval_samples"], len(eval_frame)),
            random_state=SEED,
        ).reset_index(drop=True)

        audit_extra_gpus = 0 if ddp_active else max(0, NUM_GPUS - 1)
        print(f"Loading {audit_extra_gpus} additional model copies for parallel holdout audit...")
        audit_extra_models = []
        for gpu_id in range(1, 1 + audit_extra_gpus):
            with torch.cuda.device(gpu_id):
                am, at = FastLanguageModel.from_pretrained(
                    model_name=selected_checkpoint,
                    max_seq_length=CONFIG["max_seq_length"],
                    dtype=torch.bfloat16,
                    load_in_4bit=False,
                    device_map={"": gpu_id},
                    local_files_only=HF_LOCAL_FILES_ONLY,
                )
                am = FastLanguageModel.for_inference(am)
                am.eval()
                am.config.use_cache = True
            audit_extra_models.append((am, at))

        audit_gpu_models = [(model, tokenizer)] + audit_extra_models

        def _audit_worker(gpu_id, chunk_rows):
            if not chunk_rows:
                return []
            m, tok = audit_gpu_models[gpu_id]
            device = f"cuda:{gpu_id}"
            torch.cuda.set_device(gpu_id)
            generated_rows = []
            for row_batch in batched_rows(chunk_rows, CONFIG["generation_audit_batch_size"]):
                texts = [
                    tok.apply_chat_template(
                        format_chat_messages(row.prompt),
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for row in row_batch
                ]
                orig_padding_side = tok.padding_side
                tok.padding_side = "left"
                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token
                try:
                    inputs = tok(texts, return_tensors="pt", padding=True).to(device)
                    with torch.inference_mode():
                        output_ids = m.generate(
                            **inputs,
                            max_new_tokens=CONFIG["max_new_tokens_eval"],
                            do_sample=False,
                            use_cache=False,
                        )
                    prompt_len = inputs["input_ids"].shape[1]
                    for i, row in enumerate(row_batch):
                        raw_text = tok.decode(output_ids[i][prompt_len:], skip_special_tokens=True)
                        generated_rows.append((row, raw_text))
                finally:
                    tok.padding_side = orig_padding_side
                if torch.cuda.is_available():
                    with torch.cuda.device(gpu_id):
                        torch.cuda.empty_cache()

            rows = []
            for row, raw_text in generated_rows:
                svg = normalize_svg(extract_svg(raw_text))
                valid = is_valid_svg(svg)
                metrics = pdf_proxy_score(svg, row.svg_norm) if valid else {}
                audit_row = {
                    "id": row.id,
                    "prompt": row.prompt,
                    "valid": valid,
                    "svg_chars": len(svg),
                    "raw_preview": raw_text[:400],
                }
                audit_row.update(metrics)
                rows.append(audit_row)
            return rows

        audit_rows_list = list(audit_df.itertuples())
        audit_chunks = [audit_rows_list[i::len(audit_gpu_models)] for i in range(len(audit_gpu_models))]
        audit_rows = run_gpu_workers_concurrently(audit_chunks, _audit_worker)

        for am, _ in audit_extra_models:
            del am
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        invalid_count = sum(not row["valid"] for row in audit_rows)
        with open(audit_path, "w") as f:
            json.dump(audit_rows, f, indent=2)
        pd.DataFrame(audit_rows).to_csv(audit_csv_path, index=False)

        pdf_scores = [row["pdf_proxy_score"] for row in audit_rows if row.get("pdf_proxy_score") is not None]
        render_maes = [row["render_mae"] for row in audit_rows if row.get("render_mae") is not None]
        audit_elapsed_min = (time.time() - audit_wall_start) / 60
    else:
        print("Holdout generation audit skipped.")
    run_summary = {
        "seed": SEED,
        "training_stage": CONFIG["training_stage"],
        "skip_holdout_selection": SKIP_HOLDOUT_SELECTION,
        "selected_checkpoint_by_pdf_proxy": selected_checkpoint,
        "holdout_audit_rows": len(audit_rows),
        "invalid_generations": invalid_count,
        "mean_pdf_proxy_score": (sum(pdf_scores) / len(pdf_scores)) if pdf_scores else None,
        "mean_render_mae": (sum(render_maes) / len(render_maes)) if render_maes else None,
        "audit_runtime_minutes": audit_elapsed_min,
        "package_versions": PACKAGE_VERSIONS,
    }
    run_summary_path = os.path.join(CONFIG["inference_artifact_dir"], "run_summary.json")
    with open(run_summary_path, "w") as f:
        json.dump(run_summary, f, indent=2)

    print(f"Adapter saved to Drive: {SAVE_DIR}")
    print(f"Selected checkpoint: {selected_checkpoint}")
    print(f"Inference manifest: {manifest_path}")
    print(f"Checkpoint proxy scores: {checkpoint_proxy_path}")
    print(f"Holdout export: {eval_export_path}")
    print(f"Fallback SVG: {fallback_svg_path}")
    print(f"Fallback templates: {fallback_templates_path}")
    print(f"Package versions: {package_versions_path}")
    print(f"Run summary: {run_summary_path}")
    if SKIP_HOLDOUT_SELECTION:
        print("Holdout audit rows: skipped")
        print("Invalid SVG generations: skipped")
        print("Holdout audit runtime (minutes): skipped")
    else:
        print(f"Holdout audit rows: {len(audit_rows)}")
        print(f"Invalid SVG generations: {invalid_count}")
        print(f"Holdout audit runtime (minutes): {audit_elapsed_min:.2f}")
        if pdf_scores:
            print(f"Mean PDF proxy score: {sum(pdf_scores) / len(pdf_scores):.4f}")
        if render_maes:
            print(f"Mean render MAE: {sum(render_maes) / len(render_maes):.4f}")
        print(f"Audit log: {audit_path}")
        print(f"Audit CSV: {audit_csv_path}")
        for row in audit_rows[:5]:
            status = "OK" if row["valid"] else "BAD"
            proxy = row.get("pdf_proxy_score")
            proxy_text = f" | pdf_proxy={proxy:.4f}" if proxy is not None else ""
            print(f"[{status}] {row['id']} | svg_chars={row['svg_chars']}{proxy_text} | {row['prompt'][:90]}")

    try:
        del model
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return selected_checkpoint


def merge_saved_adapter():
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    def normalize_merged_checkpoint_dir(path):
        if not os.path.isdir(path):
            return

        expected_paths = (
            os.path.join(path, "model.safetensors"),
            os.path.join(path, "model.safetensors.index.json"),
            os.path.join(path, "pytorch_model.bin"),
            os.path.join(path, "pytorch_model.bin.index.json"),
        )
        if any(os.path.exists(candidate) for candidate in expected_paths):
            return

        safetensor_files = sorted(
            name
            for name in os.listdir(path)
            if name.endswith(".safetensors") and os.path.isfile(os.path.join(path, name))
        )
        if len(safetensor_files) != 1:
            return

        source_name = safetensor_files[0]
        if source_name == "model.safetensors":
            return

        source_path = os.path.join(path, source_name)
        target_path = os.path.join(path, "model.safetensors")
        try:
            os.replace(source_path, target_path)
            print(f"Normalized merged weights: {source_name} -> model.safetensors")
        except FileNotFoundError:
            if not os.path.exists(target_path):
                print(f"Warning: merged checkpoint disappeared while normalizing {path}")
        except OSError as exc:
            print(f"Warning: failed to normalize merged checkpoint under {path}: {exc}")

    adapter_path = SAVE_DIR if os.path.exists(os.path.join(SAVE_DIR, "adapter_config.json")) else CONFIG["final_adapter_dir"]
    print(f"Loading base model for merge: {CONFIG['model_name']}")
    base_model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_name"],
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    print(f"Loading adapter from: {adapter_path}")
    merged = PeftModel.from_pretrained(base_model, adapter_path)

    print("Merging...")
    merged = merged.merge_and_unload()

    os.makedirs(MERGE_OUTPUT, exist_ok=True)
    merged.save_pretrained(MERGE_OUTPUT)
    merged_tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    merged_tokenizer.save_pretrained(MERGE_OUTPUT)
    normalize_merged_checkpoint_dir(MERGE_OUTPUT)

    total_size = sum(
        os.path.getsize(os.path.join(MERGE_OUTPUT, f))
        for f in os.listdir(MERGE_OUTPUT)
        if os.path.isfile(os.path.join(MERGE_OUTPUT, f))
    )
    print(f"Merged model: {MERGE_OUTPUT}")
    print(f"Size: {total_size / 1e9:.1f} GB")
    print("Upload this folder to Kaggle as a dataset for inference.")


if RUN_ONLY_POST_TRAIN_SELECTION:
    print("--- Standalone post-train selection mode ---")
    standalone_best_checkpoint = infer_trainer_best_checkpoint(list_checkpoint_dirs())
    if standalone_best_checkpoint:
        print(f"Trainer best checkpoint hint: {standalone_best_checkpoint}")
    else:
        print("Trainer best checkpoint hint unavailable; will rank available checkpoints directly.")
    run_post_train_selection_and_audit(
        eval_df,
        trainer_best_checkpoint=standalone_best_checkpoint,
        ddp_active=False,
    )
    merge_saved_adapter()
    raise SystemExit(0)


# Model loading.
print(f"--- Loading {CONFIG['train_model_name_or_path']} via Unsloth ---")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=CONFIG["train_model_name_or_path"],
    max_seq_length=CONFIG["max_seq_length"],
    load_in_4bit=True,
    dtype=torch.bfloat16,
    local_files_only=HF_LOCAL_FILES_ONLY,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = FastLanguageModel.get_peft_model(
    model,
    r=CONFIG["lora_r"],
    lora_alpha=CONFIG["lora_alpha"],
    lora_dropout=CONFIG["lora_dropout"],
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=SEED,
)
if TRAINING_STAGE == "long_context":
    load_training_adapter_weights(model, LONG_CONTEXT_START_SOURCE)
    print("Long-context stage loaded the selected main-stage adapter weights.")
model.config.use_cache = False
model.print_trainable_parameters()

for _i in range(torch.cuda.device_count()):
    _alloc = torch.cuda.memory_allocated(_i) / 1e9
    _total = torch.cuda.get_device_properties(_i).total_memory / 1e9
    print(f"GPU {_i} memory after load: {_alloc:.1f} / {_total:.1f} GB")


# Batch-size benchmark is skipped in script mode.

# Dataset preparation with local caching.
print("--- Preparing dataset ---")


def format_sft_text(example):
    messages = format_chat_messages(example["prompt"], example["svg_norm"])
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": text, "text_length": len(text)}


def frame_to_dataset(frame):
    base = Dataset.from_pandas(frame[["prompt", "svg_norm"]], preserve_index=False)
    return base.map(format_sft_text, remove_columns=base.column_names)


_dataset_ddp_active = int(os.environ.get("WORLD_SIZE", "1")) > 1
if os.path.exists(DATASET_CACHE):
    print(f"Loading cached dataset from {DATASET_CACHE} ...")
elif IS_MAIN_PROCESS or not _dataset_ddp_active:
    dataset_dict = DatasetDict(
        {
            "train": frame_to_dataset(train_df),
            "eval": frame_to_dataset(eval_df),
        }
    )
    dataset_dict.save_to_disk(DATASET_CACHE)
    with open(DATASET_CACHE_READY, "w") as f:
        f.write("ready\n")
    print(f"Saved dataset cache to {DATASET_CACHE}")
else:
    print(f"Waiting for rank 0 to build dataset cache at {DATASET_CACHE} ...")
    wait_start = time.time()
    while not os.path.exists(DATASET_CACHE_READY):
        if os.path.exists(DATASET_CACHE):
            # Backward-compatible fallback for caches created before the ready marker existed.
            break
        if time.time() - wait_start > 1800:
            raise TimeoutError(f"Timed out waiting for dataset cache: {DATASET_CACHE}")
        time.sleep(2)

if _dist.is_available() and _dist.is_initialized():
    _dist.barrier()

dataset_dict = load_from_disk(DATASET_CACHE)

train_dataset = dataset_dict["train"]
eval_dataset = dataset_dict["eval"]
print(f"Train dataset: {len(train_dataset)} samples")
print(f"Eval dataset:  {len(eval_dataset)} samples")
print("Sample formatted text preview:")
print(train_dataset[0]["text"][:500])


# Training loop with checkpointing and eval loss.
print("--- Training ---")


def build_sft_config():
    params = inspect.signature(SFTConfig).parameters
    sft_kwargs = {
        "output_dir": CONFIG["output_dir"],
        "num_train_epochs": CONFIG["num_train_epochs"],
        "per_device_train_batch_size": CONFIG["per_device_train_batch_size"],
        "gradient_accumulation_steps": CONFIG["gradient_accumulation_steps"],
        "learning_rate": CONFIG["learning_rate"],
        "lr_scheduler_type": CONFIG["lr_scheduler_type"],
        # warmup_ratio deprecated in TRL v5.2 → use warmup_steps.
        # We set a placeholder here and overwrite it after steps_per_epoch is known.
        "warmup_steps": 100,
        "weight_decay": CONFIG["weight_decay"],
        "bf16": True,
        "fp16": False,
        "logging_steps": CONFIG["logging_steps"],
        # epoch-by-epoch mode saves at every epoch; full run saves every 1000 steps
        "save_strategy": "epoch" if EPOCH_TO_TRAIN is not None else "steps",
        **({} if EPOCH_TO_TRAIN is not None else {"save_steps": CHECKPOINT_STEPS}),
        "save_total_limit": CONFIG["save_total_limit"],
        "dataloader_num_workers": 4,
        "dataloader_pin_memory": True,
        "dataloader_prefetch_factor": 2,
        "report_to": "none",
        "seed": SEED,
        "optim": "adamw_8bit",
        "max_grad_norm": 1.0,
        "ddp_find_unused_parameters": False,
        "max_length": CONFIG["max_seq_length"],
        "dataset_text_field": "text",
        "packing": False,  # must be False when using DataCollatorForCompletionOnlyLM
    }

    # eval_strategy MUST match save_strategy when load_best_model_at_end=True.
    _strat = "epoch" if EPOCH_TO_TRAIN is not None else "steps"
    if "eval_strategy" in params:
        sft_kwargs["eval_strategy"] = _strat
        if _strat == "steps":
            sft_kwargs["eval_steps"] = CHECKPOINT_STEPS
    elif "evaluation_strategy" in params:
        sft_kwargs["evaluation_strategy"] = _strat
        if _strat == "steps":
            sft_kwargs["eval_steps"] = CHECKPOINT_STEPS

    if "load_best_model_at_end" in params:
        sft_kwargs["load_best_model_at_end"] = True
    if "metric_for_best_model" in params:
        sft_kwargs["metric_for_best_model"] = "eval_loss"
    if "greater_is_better" in params:
        sft_kwargs["greater_is_better"] = False
    if "group_by_length" in params:
        sft_kwargs["group_by_length"] = CONFIG["group_by_length"]
    if "length_column_name" in params:
        sft_kwargs["length_column_name"] = "text_length"
    if "tf32" in params:
        sft_kwargs["tf32"] = True
    if "save_safetensors" in params:
        sft_kwargs["save_safetensors"] = True

    # Use DataCollatorForCompletionOnlyLM instead of assistant_only_loss /
    # completion_only_loss — works with flat text datasets and avoids the
    # "dataset is not conversational" ValueError from Unsloth's SFTTrainer.
    # packing must be False for the collator to correctly identify response boundaries.
    loss_mode = "completion_collator"

    # Fix eos_token: TRL/Unsloth auto-detects <EOS_TOKEN> from the chat template
    # placeholder, which doesn't exist in Qwen2's vocab → ValueError.
    if "eos_token" in params:
        im_end = "<|im_end|>"
        sft_kwargs["eos_token"] = (
            im_end
            if tokenizer.convert_tokens_to_ids(im_end) is not None
            else tokenizer.eos_token
        )

    return SFTConfig(**sft_kwargs), loss_mode


class DriveCheckpointCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        if int(os.environ.get("LOCAL_RANK", 0)) != 0:
            return control
        latest = f"checkpoint-{state.global_step}"
        src = os.path.join(args.output_dir, latest)
        dst = os.path.join(CONFIG["drive_checkpoint_dir"], latest)
        if not os.path.isdir(src):
            return control
        try:
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print(f"  >> Copied {latest} to {CONFIG['drive_checkpoint_dir']}")
        except Exception as e:
            print(f"  >> Drive checkpoint copy failed: {e}")
        return control


def _find_subsequence(sequence, subsequence):
    if not subsequence or len(subsequence) > len(sequence):
        return -1
    limit = len(sequence) - len(subsequence) + 1
    for start in range(limit):
        if sequence[start:start + len(subsequence)] == subsequence:
            return start
    return -1


class AssistantOnlyDataCollator:
    def __init__(self, tokenizer, response_template_ids, max_length, label_pad_token_id=-100):
        self.tokenizer = tokenizer
        self.response_template_ids = list(response_template_ids)
        self.max_length = max_length
        self.label_pad_token_id = label_pad_token_id

    def _tokenize_or_pad(self, features):
        if not features:
            raise ValueError("AssistantOnlyDataCollator received an empty batch.")

        first = features[0]
        if "input_ids" in first:
            pad_features = []
            for feature in features:
                item = {"input_ids": feature["input_ids"]}
                if "attention_mask" in feature:
                    item["attention_mask"] = feature["attention_mask"]
                pad_features.append(item)
            return self.tokenizer.pad(pad_features, padding=True, return_tensors="pt")

        if "text" in first:
            texts = [feature["text"] for feature in features]
            return self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=False,
                return_tensors="pt",
            )

        raise ValueError(f"Unsupported batch format for AssistantOnlyDataCollator: {list(first.keys())}")

    def __call__(self, features):
        batch = self._tokenize_or_pad(features)
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        labels = input_ids.clone()

        if attention_mask is not None:
            labels = labels.masked_fill(attention_mask == 0, self.label_pad_token_id)

        for row_idx in range(input_ids.shape[0]):
            seq_len = int(attention_mask[row_idx].sum().item()) if attention_mask is not None else input_ids.shape[1]
            tokens = input_ids[row_idx, :seq_len].tolist()
            start_idx = _find_subsequence(tokens, self.response_template_ids)
            if start_idx < 0:
                preview = self.tokenizer.decode(tokens[: min(seq_len, 160)], skip_special_tokens=False)
                raise RuntimeError(
                    "Assistant response marker not found in batch item. "
                    f"Expected token IDs {self.response_template_ids}. Preview: {preview!r}"
                )
            response_start = start_idx + len(self.response_template_ids)
            labels[row_idx, :response_start] = self.label_pad_token_id

        batch["labels"] = labels
        return batch


sft_config, _requested_loss_mode = build_sft_config()

# Epoch-by-epoch override
if EPOCH_TO_TRAIN is not None:
    if EPOCH_TO_TRAIN < 1 or EPOCH_TO_TRAIN > CONFIG["num_train_epochs"]:
        raise ValueError(
            f"EPOCH_TO_TRAIN={EPOCH_TO_TRAIN} is out of range "
            f"[1, {CONFIG['num_train_epochs']}]."
        )
    sft_config.num_train_epochs = EPOCH_TO_TRAIN
    print(f"[epoch-by-epoch] Targeting epoch {EPOCH_TO_TRAIN} of {CONFIG['num_train_epochs']} total")

train_wall_start = time.time()

# Build completion-only collator — masks loss on everything except assistant responses.
# Prefer TRL's native implementation when available, otherwise fall back to a
# local collator so masked loss still works on newer/older TRL builds.
_data_collator = None
_train_on_responses_fn = None
loss_mode = "full_sequence_loss"

_response_template = "<|im_start|>assistant\n"
_response_ids = tokenizer.encode(_response_template, add_special_tokens=False)
if not _response_ids:
    raise RuntimeError(f"Response template {repr(_response_template)} produced no token IDs.")
print(f"Response template token IDs: {_response_ids}  "
      f"({tokenizer.convert_ids_to_tokens(_response_ids)})")

try:
    from trl import DataCollatorForCompletionOnlyLM
    _data_collator = DataCollatorForCompletionOnlyLM(
        response_template=_response_ids,
        tokenizer=tokenizer,
        mlm=False,
    )
    print("Using DataCollatorForCompletionOnlyLM")
    loss_mode = "trl_completion_collator"
except ImportError:
    try:
        from trl import train_on_responses_only as _train_on_responses_fn
        print("DataCollatorForCompletionOnlyLM not available — will use train_on_responses_only (TRL 0.12+)")
        loss_mode = "trl_train_on_responses_only"
    except ImportError:
        _data_collator = AssistantOnlyDataCollator(
            tokenizer=tokenizer,
            response_template_ids=_response_ids,
            max_length=CONFIG["max_seq_length"],
        )
        loss_mode = "custom_completion_collator"
        print("Using custom AssistantOnlyDataCollator fallback.")

print(f"Loss mode: {loss_mode}")
if loss_mode == "full_sequence_loss":
    msg = "Installed TRL build does not expose assistant-only/completion-only loss."
    if CONFIG.get("require_masked_loss", False):
        raise RuntimeError(msg + " Upgrade TRL/Unsloth before training for best quality.")
    print("Warning: " + msg)
else:
    print("Loss mode: assistant-response masking is enabled.")

_trainer_kwargs = dict(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
)
if _data_collator is not None:
    _trainer_kwargs["data_collator"] = _data_collator

trainer = SFTTrainer(**_trainer_kwargs)

if _data_collator is None and _train_on_responses_fn is not None:
    trainer = _train_on_responses_fn(
        trainer,
        instruction_template="<|im_start|>user\n",
        response_template=_response_template,
    )
    print("train_on_responses_only applied.")
trainer.add_callback(DriveCheckpointCallback())

steps_per_epoch = len(trainer.get_train_dataloader())
total_steps = steps_per_epoch * sft_config.num_train_epochs  # use actual epoch count (respects EPOCH_TO_TRAIN override)
# Now that we know total steps, recalculate warmup_steps from the original ratio
actual_warmup_steps = max(1, int(total_steps * CONFIG["warmup_ratio"]))
trainer.args.warmup_steps = actual_warmup_steps
print(f"Steps/epoch: {steps_per_epoch}")
print(f"Approx total steps: {total_steps}")
print(f"Warmup steps: {actual_warmup_steps} ({CONFIG['warmup_ratio']*100:.0f}% of total)")
print(f"Training batch size: {CONFIG['per_device_train_batch_size']}")
print(f"Checkpoints: local={CONFIG['output_dir']} | drive={CONFIG['drive_checkpoint_dir']}")

# When EPOCH_TO_TRAIN == 1, always start fresh regardless of existing checkpoints.
# When EPOCH_TO_TRAIN > 1, resume from the most-recent epoch checkpoint.
# Long-context also resumes by default when stage-2 checkpoints already exist.
force_fresh = (EPOCH_TO_TRAIN == 1) or (
    TRAINING_STAGE == "long_context" and not ALLOW_LONG_CONTEXT_RESUME
)
resume_path = None
if not force_fresh:
    for search_dir in [CONFIG["output_dir"], CONFIG["drive_checkpoint_dir"], SAVE_DIR]:
        if not os.path.exists(search_dir):
            continue
        ckpts = [
            d for d in os.listdir(search_dir)
            if os.path.isdir(os.path.join(search_dir, d)) and d.startswith("checkpoint-")
        ]
        if not ckpts:
            continue
        latest = sorted(ckpts, key=lambda x: int(x.split("-")[-1]))[-1]
        found_path = os.path.join(search_dir, latest)
        if search_dir == SAVE_DIR:
            local_path = os.path.join(CONFIG["output_dir"], latest)
            if not os.path.exists(local_path):
                print(f"Copying Drive checkpoint to local: {latest}")
                shutil.copytree(found_path, local_path, dirs_exist_ok=True)
            resume_path = local_path
        else:
            resume_path = found_path
        break

if resume_path:
    print(f"RESUMING from: {resume_path}")
    train_result = trainer.train(resume_from_checkpoint=resume_path)
else:
    if TRAINING_STAGE == "long_context":
        print("Starting fresh long_context trainer state from the selected main-stage adapter.")
    elif force_fresh:
        print("Starting fresh (EPOCH_TO_TRAIN=1)")
    else:
        print("Starting fresh")
    train_result = trainer.train()

train_elapsed_hours = (time.time() - train_wall_start) / 3600
print("Training metrics:")
print(train_result.metrics)
print(f"Training runtime (hours): {train_elapsed_hours:.2f}")
print(f"Best checkpoint: {trainer.state.best_model_checkpoint}")

# Save a labelled adapter snapshot so the next epoch can identify its starting point
if EPOCH_TO_TRAIN is not None:
    epoch_adapter_dir = os.path.join(SAVE_DIR, f"epoch_{EPOCH_TO_TRAIN}_adapter")
    os.makedirs(epoch_adapter_dir, exist_ok=True)
    model.save_pretrained(epoch_adapter_dir)
    tokenizer.save_pretrained(epoch_adapter_dir)
    print(f"[epoch-by-epoch] Adapter snapshot saved → {epoch_adapter_dir}")
    print(f"  To continue: set EPOCH_TO_TRAIN = {EPOCH_TO_TRAIN + 1} and re-run this cell.")


# Best-checkpoint selection, adapter save, and holdout audit.
# Only rank 0 runs inference/evaluation in DDP mode
_ddp_active = _dist.is_available() and _dist.is_initialized()

if DEFER_POST_TRAIN_SELECTION:
    if IS_MAIN_PROCESS:
        print("Post-train selection deferred to standalone launcher step.")
else:
    if not IS_MAIN_PROCESS and _ddp_active:
        _dist.barrier()

    if IS_MAIN_PROCESS:
        run_post_train_selection_and_audit(
            eval_df,
            trainer_best_checkpoint=trainer.state.best_model_checkpoint,
            ddp_active=_ddp_active,
        )

    if _ddp_active:
        _dist.barrier()

    if IS_MAIN_PROCESS:
        merge_saved_adapter()
