# CSCI316 Project 2 — Transfer Learning for Low-Resource Language Understanding

**Group:** big_boyz  
**Task:** Sentiment Analysis in Gulf Arabic Code-Switched Text  
**Base Model:** [Jais-1.3B](https://huggingface.co/inceptionai/jais-family-1p3b) (Arabic-focused LLM)  
**Course:** CSCI316 Big Data Mining & Applications — University of Wollongong in Dubai

## Overview

This project explores how transfer learning can adapt English-centric LLMs to understand Gulf Arabic sentiment in code-switched text (Arabic mixed with English). We implement and compare two transfer learning strategies:

1. **Full Fine-Tuning** — All 1.3B parameters updated using DeepSpeed ZeRO-2 with CPU offloading
2. **LoRA (Low-Rank Adaptation)** — Only ~0.5% of parameters trained via low-rank adapters

We evaluate beyond standard metrics with a custom **Dialectal Sentiment Fidelity Score (DSFS)** that tests cultural understanding of Gulf Arabic expressions, code-switching patterns, and sarcasm.

## Dataset

~166K samples from multiple sources:
- Arabic Sentiment Twitter Corpus (~58K tweets)
- LABR: Large Arabic Book Reviews (~63K, 1-5 stars → 3 classes)
- HARD: Hotel Arabic Reviews Dataset
- 60 synthetic Gulf Arabic code-switched examples

**Labels:** Negative (0), Neutral (1), Positive (2)

## Project Structure

```
big_boyz_repo/
├── configs/
│   ├── config.py                    # All hyperparameters and paths
│   └── ds_zero2_cpu_offload.json    # DeepSpeed ZeRO-2 config
├── scripts/
│   ├── utils.py                     # Preprocessing, metrics, seeds
│   ├── dataset.py                   # PyTorch Dataset class
│   ├── prepare_data.py              # Step 1: Download & preprocess data
│   ├── train_full_finetune.py       # Step 2: Full fine-tuning (DeepSpeed)
│   ├── train_lora.py                # Step 3: LoRA fine-tuning
│   ├── evaluate.py                  # Step 4: Comparison & cultural eval
│   └── serve.py                     # Flask inference API
├── data/                            # Generated train/val/test CSVs
├── checkpoints/                     # Saved model weights
├── results/                         # Metrics, plots, comparisons
├── Dockerfile                       # Containerized inference
├── requirements.txt
└── README.md
```

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | GTX 1080 Ti (11 GB) | RTX 3090/4090 (24 GB) |
| CPU | i5-10400 / Ryzen 5 3600 | i5-12400 / Ryzen 5 5600X |
| RAM | 32 GB | 64 GB |
| Storage | 30 GB SSD | 50 GB NVMe SSD |
| CUDA | 11.8+ | 11.8 |

## Setup

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate          # Linux/Mac
venv\Scripts\activate             # Windows

# Install PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install project dependencies
pip install -r requirements.txt
```

### 2. Authenticate with HuggingFace

```bash
huggingface-cli login
```

### 3. Verify GPU

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

## Running the Pipeline

Execute scripts in order from the project root:

```bash
# Step 1: Download datasets, preprocess, create splits
python scripts/prepare_data.py

# Step 2: Full fine-tuning with DeepSpeed ZeRO-2 + CPU offloading
deepspeed scripts/train_full_finetune.py

# Step 3: LoRA fine-tuning
python scripts/train_lora.py

# Step 4: Compare strategies, cultural evaluation, qualitative analysis
python scripts/evaluate.py
```

> **Note:** If DeepSpeed is not installed, Step 2 falls back to standard HuggingFace Trainer automatically.

## Docker Deployment

Deploy the inference API without training — the LoRA adapter downloads automatically from [HuggingFace Hub](https://huggingface.co/ziyanhashim/jais-lora-gulf-arabic-sentiment).

**Prerequisites:**
- A HuggingFace account with access to the gated [Jais-1.3B](https://huggingface.co/inceptionai/jais-family-1p3b) base model
- A HuggingFace access token (`HF_TOKEN`) — generate one at https://huggingface.co/settings/tokens

```bash
# Build image
docker build -t big_boyz_sentiment .

# Run container (first run downloads ~2.7 GB of model weights)
docker run -p 5000:5000 -e HF_TOKEN=hf_your_token_here big_boyz_sentiment

# Test prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "هالمطعم وايد حلو the food is amazing"}'
```

> **Note:** The first startup takes several minutes while model weights download. Subsequent runs reuse cached weights if the container is restarted (not rebuilt).

## Transfer Learning Strategies

### Strategy 1: Full Fine-Tuning (DeepSpeed ZeRO-2)
- Updates all 1.3B parameters
- Uses ZeRO Stage 2 to offload optimizer states (AdamW) to CPU RAM
- Enables gradient checkpointing to reduce activation memory
- GPU holds weights + gradients + activations (~13-14 GB)
- CPU holds optimizer states (~10 GB in system RAM)

### Strategy 2: LoRA (Low-Rank Adaptation)
- Injects low-rank matrices into attention layers (q, k, v, o projections)
- Trains only ~7M parameters (~0.5% of total)
- Rank=16, Alpha=32, Dropout=0.1
- Significantly faster and more memory efficient

## Cultural Evaluation: DSFS

The **Dialectal Sentiment Fidelity Score** evaluates model understanding across three tiers:

1. **Gulf Expressions** (weight=1.0): Gulf-specific phrases like "وايد حلو" (very nice)
2. **Code-Switching** (weight=0.8): Mixed Arabic-English sentences
3. **Culturally Ambiguous** (weight=1.2): Sarcasm and context-dependent phrases

## References

- Jais: [Sengupta et al., 2023](https://arxiv.org/abs/2308.16149)
- LoRA: [Hu et al., 2021](https://arxiv.org/abs/2106.09685)
- DeepSpeed ZeRO: [Rajbhandari et al., 2020](https://arxiv.org/abs/1910.02054)

## License

Academic use only — CSCI316 Winter Session 2026, UOWD.
