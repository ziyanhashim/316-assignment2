"""
CSCI316 Project 2 — Strategy 2: LoRA (PEFT) Fine-Tuning
Low-Rank Adaptation — trains ~0.5% of parameters.

Run:  python scripts/train_lora.py
  OR: deepspeed scripts/train_lora.py
"""
import sys
import os
import json
import time
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from sklearn.metrics import classification_report, confusion_matrix

from configs.config import *
from scripts.utils import set_seed, compute_metrics
from scripts.dataset import ArabicSentimentDataset

set_seed(SEED)

# Compatibility patch
import transformers.pytorch_utils as _pu
if not hasattr(_pu, "find_pruneable_heads_and_indices"):
    try:
        from transformers.utils import find_pruneable_heads_and_indices
        _pu.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
    except ImportError:
        pass


def main():
    print("=" * 60)
    print("STRATEGY 2: LoRA (PEFT) Fine-Tuning")
    print("=" * 60)

    # ----------------------------------------------------------
    # 1. Load tokenizer and data
    # ----------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    train_dataset = ArabicSentimentDataset(train_df, tokenizer, MAX_LENGTH)
    val_dataset = ArabicSentimentDataset(val_df, tokenizer, MAX_LENGTH)
    test_dataset = ArabicSentimentDataset(test_df, tokenizer, MAX_LENGTH)

    # ----------------------------------------------------------
    # 2. Load model + apply LoRA
    # ----------------------------------------------------------
    print(f"\nLoading {MODEL_NAME} for LoRA...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["c_attn", "c_proj", "c_fc", "c_fc2"],
        bias="none",
        modules_to_save=["score"],
    )

    model = get_peft_model(model, lora_config)

    lora_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_total = sum(p.numel() for p in model.parameters())

    print(f"\nLoRA Configuration:")
    print(f"  Rank (r):     {LORA_RANK}")
    print(f"  Alpha:        {LORA_ALPHA}")
    print(f"  Dropout:      {LORA_DROPOUT}")
    print(f"  Target:       q_proj, v_proj, k_proj, o_proj")
    print(f"  Trainable:    {lora_trainable:,} ({lora_trainable / lora_total * 100:.4f}%)")
    model.print_trainable_parameters()

    # ----------------------------------------------------------
    # 3. Train
    # ----------------------------------------------------------
    lora_dir = os.path.join(CHECKPOINT_DIR, "lora_finetune")
    os.makedirs(lora_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=lora_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE * 5,  # LoRA benefits from higher LR
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        fp16=True,
        gradient_checkpointing=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_dir=os.path.join(lora_dir, "logs"),
        logging_steps=50,
        report_to="none",
        seed=SEED,
        data_seed=SEED,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # Resume
    checkpoints = glob.glob(os.path.join(lora_dir, "checkpoint-*"))
    resume_from = max(checkpoints, key=os.path.getctime) if checkpoints else None
    if resume_from:
        print(f"Resuming from {resume_from}")

    print(f"\n{'=' * 60}")
    print(f"STARTING LoRA FINE-TUNING")
    print(f"Training {lora_trainable:,} parameters ({lora_trainable / lora_total * 100:.4f}%)")
    print(f"{'=' * 60}")

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    train_result = trainer.train(resume_from_checkpoint=resume_from)
    ft_time = time.time() - start_time
    ft_memory = torch.cuda.max_memory_allocated() / 1e9

    print(f"\nCompleted in {ft_time:.1f}s ({ft_time / 60:.1f} min)")
    print(f"Training loss: {train_result.training_loss:.4f}")
    print(f"Peak GPU memory: {ft_memory:.2f} GB")

    # ----------------------------------------------------------
    # 4. Evaluate
    # ----------------------------------------------------------
    print("\nEvaluating on test set...")
    eval_results = trainer.evaluate(test_dataset)
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    labels = predictions.label_ids

    print(f"\n{'=' * 60}")
    print("LoRA — TEST RESULTS")
    print(f"{'=' * 60}")
    for k, v in eval_results.items():
        if "eval_" in k:
            print(f"  {k.replace('eval_', ''):>15}: {v:.4f}")

    print(f"\n{classification_report(labels, preds, target_names=[LABEL_MAP[i] for i in range(NUM_LABELS)])}")

    # Save model
    model_path = os.path.join(CHECKPOINT_DIR, "jais_lora_best")
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
                xticklabels=[LABEL_MAP[i] for i in range(NUM_LABELS)],
                yticklabels=[LABEL_MAP[i] for i in range(NUM_LABELS)], ax=ax)
    ax.set_title("Confusion Matrix — LoRA", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix_lora.png"), dpi=150)

    # Save results
    results = {
        "accuracy": eval_results.get("eval_accuracy", 0),
        "f1": eval_results.get("eval_f1", 0),
        "precision": eval_results.get("eval_precision", 0),
        "recall": eval_results.get("eval_recall", 0),
        "training_time_seconds": ft_time,
        "peak_gpu_memory_gb": ft_memory,
        "total_params": lora_total,
        "trainable_params": lora_trainable,
        "trainable_percent": lora_trainable / lora_total * 100,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "strategy": "LoRA (PEFT)",
    }
    with open(os.path.join(RESULTS_DIR, "lora_ft_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n✓ LoRA fine-tuning complete!")


if __name__ == "__main__":
    main()
