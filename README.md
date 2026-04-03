# Text-to-SVG Generation — Kaggle Competition

NYU Deep Learning — Spring 2026  
Team: Errol Elbasan, Meghana Gudamsetty

## Model

Fine-tuned **Qwen2.5-Coder-3B-Instruct** with **LoRA** (4-bit quantization via Unsloth) for SVG code generation from text prompts. The model takes a natural-language description as input and outputs a valid SVG string.

## Model Weights

[Link to model weights — Google Drive / HuggingFace] *(update this)*

## Repository Structure

```
├── train_qwen2p5coder3b.py     # LoRA fine-tuning script
├── inference_qwen2p5coder3b.py # Inference + submission CSV generation
├── requirements.txt            # Package dependencies
└── README.md
```

## Environment Setup

```bash
pip install -r requirements.txt
```

Or the script will auto-install dependencies when run in Google Colab.

Key packages: `unsloth`, `transformers`, `trl`, `peft`, `bitsandbytes`, `cairosvg`, `accelerate`

## Training

The training script fine-tunes Qwen2.5-Coder-3B-Instruct using LoRA with 4-bit quantization via Unsloth.

**Default usage (single GPU):**
```bash
python train_qwen2p5coder3b.py
```

**Multi-GPU (torchrun):**
```bash
torchrun --nproc_per_node=<N> train_qwen2p5coder3b.py
```

**Key hyperparameters (configurable via environment variables):**

| Variable | Default | Description |
|---|---|---|
| `MAIN_LR` | `1e-4` | Learning rate for main training stage |
| `PER_DEVICE_BATCH_SIZE` | `8` | Batch size per GPU |
| `GRADIENT_ACCUMULATION_STEPS` | `4` | Gradient accumulation steps |
| `CHECKPOINT_STEPS` | `500` | Save checkpoint every N steps |
| `TRAINING_STAGE` | `main` | `main` or `long_context` |

**Data:** Place your training CSV at `{ROOT}/Data/train.csv` with `id`, `prompt`, and `svg` columns.

**Epoch-by-epoch training** (for Colab where sessions time out):
```bash
EPOCH_TO_TRAIN=1 python train_qwen2p5coder3b.py  # Train epoch 1
EPOCH_TO_TRAIN=2 python train_qwen2p5coder3b.py  # Resume and train epoch 2
```

Adapters are saved to `{ROOT}/lora-adapter/` and synced to Google Drive automatically.

## Inference

The inference script loads the fine-tuned model and generates SVGs for each test prompt, with candidate re-ranking and fallback handling.

```bash
python inference_qwen2p5coder3b.py
```

**Key environment variables:**

| Variable | Default | Description |
|---|---|---|
| `ADAPTER_PATH` | `{ROOT}/lora-adapter` | Path to LoRA adapter |
| `MODEL_PATH` | `{ROOT}/model-merged` | Path to merged model |
| `TEST_CSV_PATH` | `{ROOT}/Data/test.csv` | Test data |
| `SUBMISSION_CSV_OUT` | `{ROOT}/submission.csv` | Output CSV path |

The script outputs a CSV with `id` and `svg` columns, ready for Kaggle submission.

## Reproducibility

- Random seed is fixed to `42` in all code (Python, NumPy, PyTorch, CUDA)
- Model: `unsloth/qwen2.5-coder-3b-instruct-bnb-4bit`
- Quantization: 4-bit with `bf16` compute
- Inference uses greedy decoding + temperature sampling with candidate re-ranking

## AI Tooling Disclosure

LLM assistants (Claude) were used for coding assistance and debugging during development.
