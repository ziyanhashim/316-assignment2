# CSCI316 Project 2 — Transfer Learning for Low-Resource Language Understanding

**Group:** big_boyz  
**Task:** Sentiment Analysis in Gulf Arabic Code-Switched Text  
**Course:** CSCI316 Big Data Mining & Applications — University of Wollongong in Dubai

## Overview

This project explores how transfer learning can adapt LLMs to understand Gulf Arabic sentiment in code-switched text (Arabic mixed with English). We implement and compare two transfer learning strategies using **PyTorch/HuggingFace** and provide a cross-framework comparison with **TensorFlow/Keras**.

### Models Used

| Framework | Model | Parameters | Link |
|-----------|-------|------------|------|
| PyTorch | [Jais-1.3B](https://huggingface.co/inceptionai/jais-family-1p3b) | 1.3B | Arabic-focused LLM by Inception/MBZUAI |
| TensorFlow | [XLM-RoBERTa-base](https://huggingface.co/xlm-roberta-base) | 278M | Multilingual (100 languages incl. Arabic) |

### Transfer Learning Strategies (PyTorch)

1. **Progressive Layer Unfreezing** — Unfreeze last 6/24 transformer layers + classification head (~22% of parameters)
2. **LoRA (Low-Rank Adaptation)** — Inject low-rank adapters into all attention layers (~0.96% of parameters)

### Key Results

| Strategy | Accuracy | F1 (Macro) | Trainable Params |
|----------|----------|------------|------------------|
| Layer Unfreezing (Jais) | 71.05% | 63.45% | 302M (21.86%) |
| **LoRA (Jais)** | **89.20%** | **86.06%** | **13.4M (0.96%)** |
| TF/Keras (XLM-RoBERTa) | 67.93% | 62.34% | — |

## Dataset

~166K samples from multiple sources:
- **Arabic Sentiment Twitter Corpus** (~58K tweets) — binary pos/neg
- **LABR**: Large Arabic Book Reviews (~63K, 1-5 stars → 3 classes)
- **HARD**: Hotel Arabic Reviews Dataset
- **60 synthetic** Gulf Arabic code-switched examples

**Labels:** Negative (0), Neutral (1), Positive (2)

## Project Structure

```
316-assignment2/
├── configs/
│   ├── config.py                    # Hyperparameters and paths
│   ├── accelerate_config.yaml       # Accelerate configuration
│   └── ds_zero2_cpu_offload.json    # DeepSpeed ZeRO-2 config
├── scripts/
│   ├── utils.py                     # Preprocessing, metrics, seeds
│   ├── dataset.py                   # PyTorch Dataset class
│   ├── prepare_data.py              # Step 1: Download & preprocess data
│   ├── train_full_finetune.py       # Step 2: Progressive layer unfreezing
│   ├── train_lora.py                # Step 3: LoRA fine-tuning
│   ├── train_tensorflow.py          # Step 4: TensorFlow/Keras implementation
│   ├── evaluate.py                  # Step 5: Comparison & cultural eval
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

**Tested on:** Intel i9-14900K, 64 GB DDR5, NVIDIA RTX 4070 Ti Super (16 GB), Windows 11

## Setup

### 1. Create Virtual Environment

```bash
# Linux/Mac
python -m venv venv
source venv/bin/activate

# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Windows (Command Prompt)
python -m venv venv
venv\Scripts\activate
```

### 2. Install PyTorch (CUDA 11.8)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install Dependencies

```bash
pip install transformers==4.44.0 datasets peft accelerate sentencepiece protobuf scikit-learn pandas numpy matplotlib seaborn emoji flask huggingface-hub tqdm
```

For TensorFlow (CPU — TF dropped Windows GPU support after 2.10):
```bash
pip install tensorflow
```

### 4. Authenticate with HuggingFace

Jais-1.3B requires authentication:
```bash
python -c "from huggingface_hub import login; login()"
```
Paste your token from https://huggingface.co/settings/tokens

### 5. Verify GPU

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

## Running the Pipeline

Execute scripts in order from the project root:

```bash
# Step 1: Download datasets, preprocess, create train/val/test splits
python scripts/prepare_data.py

# Step 2: Progressive layer unfreezing (Jais-1.3B, last 6/24 layers)
python scripts/train_full_finetune.py

# Step 3: LoRA fine-tuning (Jais-1.3B, rank=16 adapters on all layers)
python scripts/train_lora.py

# Step 4: TensorFlow/Keras implementation (XLM-RoBERTa-base)
python scripts/train_tensorflow.py

# Step 5: Compare strategies, cultural evaluation, qualitative analysis
python scripts/evaluate.py
```

## Docker Deployment

Deploy the inference API without training — the LoRA adapter downloads automatically from [HuggingFace Hub](https://huggingface.co/ziyanhashim/jais-lora-gulf-arabic-sentiment).

**Prerequisites:**
- A HuggingFace account with access to the gated [Jais-1.3B](https://huggingface.co/inceptionai/jais-family-1p3b) base model
- A HuggingFace access token (`HF_TOKEN`) — generate one at https://huggingface.co/settings/tokens

### Build and Run

```bash
docker build -t big_boyz_sentiment .

# First run downloads ~2.7 GB of model weights
docker run -p 5000:5000 -e HF_TOKEN=your_token_here big_boyz_sentiment
```

### Test the API

**Command Prompt (Windows):**
```bash
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d "{\"text\": \"this restaurant is amazing\"}"
```

**PowerShell (Windows):**
```powershell
Invoke-WebRequest -Uri http://localhost:5000/predict -Method POST -ContentType "application/json" -Body '{"text": "this restaurant is amazing"}'
```

**Linux/Mac:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "هالمطعم وايد حلو the food is amazing"}'
```

**Health check:** Visit http://localhost:5000/health in your browser.

> **Note:** The first startup takes several minutes while model weights download. Subsequent runs reuse cached weights if the container is restarted (not rebuilt).

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Predict sentiment for Arabic/code-switched text |
| `/health` | GET | Check API status |

## Transfer Learning Strategies

### Strategy 1: Progressive Layer Unfreezing
- Freezes the first 18/24 transformer layers (general linguistic features)
- Unfreezes last 6 layers + final LayerNorm + classification head
- Trains 302M parameters (~22% of total)
- Uses fp32 model with fp16 mixed-precision training

### Strategy 2: LoRA (Low-Rank Adaptation)
- Injects trainable low-rank matrices into attention layers (`c_attn`, `c_proj`, `c_fc`, `c_fc2`)
- Freezes all original weights, trains only adapter parameters
- Trains 13.4M parameters (~0.96% of total)
- Rank=16, Alpha=32, Dropout=0.1

### TensorFlow/Keras (Cross-Framework)
- Uses XLM-RoBERTa-base (278M params) — Jais lacks TF support due to custom PyTorch code
- Unfreezes last 2/12 encoder blocks + classification head
- Runs on CPU (TensorFlow dropped native Windows GPU support after v2.10)

## Cultural Evaluation: DSFS

The **Dialectal Sentiment Fidelity Score** evaluates model understanding across three tiers using 25 hand-crafted test examples:

| Category | Weight | Accuracy | Examples |
|----------|--------|----------|----------|
| Gulf Expressions | 1.0 | 70% (7/10) | "وايد حلو", "الله يعطيك العافيه" |
| Code-Switching | 0.8 | 70% (7/10) | "This is amazing والله ما توقعت" |
| Culturally Ambiguous | 1.2 | 80% (4/5) | Sarcasm, context-dependent phrases |

**Overall DSFS Score: 72.5%**

## References

- Jais: [Sengupta et al., 2023](https://arxiv.org/abs/2308.16149) — *Jais and Jais-chat: Arabic-Centric Foundation and Instruction-Tuned Open Generative Large Language Models*
- LoRA: [Hu et al., 2021](https://arxiv.org/abs/2106.09685) — *LoRA: Low-Rank Adaptation of Large Language Models*
- DeepSpeed ZeRO: [Rajbhandari et al., 2020](https://arxiv.org/abs/1910.02054) — *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*
- XLM-RoBERTa: [Conneau et al., 2020](https://arxiv.org/abs/1911.02116) — *Unsupervised Cross-lingual Representation Learning at Scale*

## License

Academic use only — CSCI316 Winter Session 2026, UOWD.
