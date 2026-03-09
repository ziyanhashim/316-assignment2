"""
CSCI316 Project 2 — Configuration Constants
Group: big_boyz
"""
import os

# ============================================================
# Paths — Local PC setup (no Google Drive / Colab)
# ============================================================
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "checkpoints")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

for d in [CHECKPOINT_DIR, DATA_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# Model
# ============================================================
MODEL_NAME = "inceptionai/jais-family-1p3b"
NUM_LABELS = 3
LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}
LABEL2ID = {"Negative": 0, "Neutral": 1, "Positive": 2}
MAX_LENGTH = 128

# ============================================================
# Training — DeepSpeed ZeRO-2 + CPU Offloading
# ============================================================
BATCH_SIZE = 16           # Larger batch — DeepSpeed offloads optimizer to CPU
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch = 16 * 2 = 32

# DeepSpeed config path
DEEPSPEED_CONFIG = os.path.join(PROJECT_DIR, "configs", "ds_zero2_cpu_offload.json")

# ============================================================
# LoRA
# ============================================================
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1

# ============================================================
# Device
# ============================================================
def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEVICE = None  # Call get_device() when torch is available
SEED = 42
