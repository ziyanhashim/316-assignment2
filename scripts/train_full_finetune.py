"""
CSCI316 Project 2 — Strategy 1: Full Fine-Tuning (All 1.3B Parameters)
Uses DeepSpeed ZeRO Stage 2 + CPU Offloading to fit on 16 GB GPU.

Run:  deepspeed scripts/train_full_finetune.py
  OR: python scripts/train_full_finetune.py  (falls back to standard Trainer)
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
from sklearn.metrics import classification_report, confusion_matrix

from configs.config import *
from scripts.utils import set_seed, compute_metrics
from scripts.dataset import ArabicSentimentDataset

set_seed(SEED)

# ============================================================
# Compatibility patch for Jais custom code
# ============================================================
import transformers.pytorch_utils as _pu
if not hasattr(_pu, "find_pruneable_heads_and_indices"):
    try:
        from transformers.utils import find_pruneable_heads_and_indices
        _pu.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
        print("Applied compatibility patch for find_pruneable_heads_and_indices")
    except ImportError:
        print("Warning: Could not patch. Ensure transformers<=4.44.0")


def main():
    print("=" * 60)
    print("STRATEGY 1: Progressive Layer Unfreezing")
    print("=" * 60)

    # ----------------------------------------------------------
    # 1. Load tokenizer and data
    # ----------------------------------------------------------
    print(f"\nLoading tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    train_dataset = ArabicSentimentDataset(train_df, tokenizer, MAX_LENGTH)
    val_dataset = ArabicSentimentDataset(val_df, tokenizer, MAX_LENGTH)
    test_dataset = ArabicSentimentDataset(test_df, tokenizer, MAX_LENGTH)

    # ----------------------------------------------------------
    # 2. Load model — ALL parameters trainable
    # ----------------------------------------------------------
    print(f"\nLoading {MODEL_NAME} (fp32, all params trainable)...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    # Without DeepSpeed, freeze early layers to fit in 16 GB VRAM
    for param in model.parameters():
        param.requires_grad = False

    UNFREEZE_LAST_N = 6
    total_layers = 24
    for name, param in model.named_parameters():
        if 'score' in name or 'ln_f' in name:
            param.requires_grad = True
        else:
            for layer_idx in range(total_layers - UNFREEZE_LAST_N, total_layers):
                if f'.h.{layer_idx}.' in name or f'.layers.{layer_idx}.' in name:
                    param.requires_grad = True
                    break
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")

    # ----------------------------------------------------------
    # 3. Sanity check (20 steps)
    # ----------------------------------------------------------
    print("\nRunning sanity check (20 steps)...")
    sanity_args = TrainingArguments(
        output_dir=os.path.join(CHECKPOINT_DIR, "sanity_check"),
        max_steps=20,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=5,
        report_to="none",
        seed=SEED,
        save_strategy="no",
    )
    sanity_trainer = Trainer(
        model=model, args=sanity_args, train_dataset=train_dataset
    )
    result = sanity_trainer.train()
    print(f"Sanity check loss: {result.training_loss:.4f}")
    if result.training_loss < 0.01:
        print("FAIL — loss is zero. Check setup.")
        return
    print("PASS — model is learning!\n")
    del sanity_trainer
    torch.cuda.empty_cache()

    # ----------------------------------------------------------
    # 4. Full training
    # ----------------------------------------------------------
    full_ft_dir = os.path.join(CHECKPOINT_DIR, "full_finetune")
    os.makedirs(full_ft_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=full_ft_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        gradient_checkpointing=True,  # Safe with fp32 master weights
        logging_dir=os.path.join(full_ft_dir, "logs"),
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

    # Resume from checkpoint if available
    checkpoints = glob.glob(os.path.join(full_ft_dir, "checkpoint-*"))
    resume_from = max(checkpoints, key=os.path.getctime) if checkpoints else None
    if resume_from:
        print(f"Resuming from {resume_from}")

    print("=" * 60)
    print("STARTING FULL FINE-TUNING")
    print(f"Training {trainable_params:,} parameters")
    print("=" * 60)

    start_time = time.time()
    train_result = trainer.train(resume_from_checkpoint=resume_from)
    ft_time = time.time() - start_time
    ft_memory = torch.cuda.max_memory_allocated() / 1e9

    print(f"\nCompleted in {ft_time:.1f}s ({ft_time / 60:.1f} min)")
    print(f"Training loss: {train_result.training_loss:.4f}")
    print(f"Peak GPU memory: {ft_memory:.2f} GB")

    # ----------------------------------------------------------
    # 5. Evaluate on test set
    # ----------------------------------------------------------
    print("\nEvaluating on test set...")
    eval_results = trainer.evaluate(test_dataset)
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    labels = predictions.label_ids

    print(f"\n{'=' * 60}")
    print("FULL FINE-TUNING — TEST RESULTS")
    print(f"{'=' * 60}")
    for k, v in eval_results.items():
        if "eval_" in k:
            print(f"  {k.replace('eval_', ''):>15}: {v:.4f}")

    print(f"\n{classification_report(labels, preds, target_names=[LABEL_MAP[i] for i in range(NUM_LABELS)])}")

    # Save model
    model_path = os.path.join(CHECKPOINT_DIR, "jais_full_finetune_best")
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"Model saved to {model_path}")

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[LABEL_MAP[i] for i in range(NUM_LABELS)],
                yticklabels=[LABEL_MAP[i] for i in range(NUM_LABELS)], ax=ax)
    ax.set_title("Confusion Matrix — Full Fine-Tuning", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix_full_ft.png"), dpi=150)
    print(f"Confusion matrix saved to {RESULTS_DIR}/")

    # Save results JSON
    results = {
        "accuracy": eval_results.get("eval_accuracy", 0),
        "f1": eval_results.get("eval_f1", 0),
        "precision": eval_results.get("eval_precision", 0),
        "recall": eval_results.get("eval_recall", 0),
        "training_time_seconds": ft_time,
        "peak_gpu_memory_gb": ft_memory,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_percent": trainable_params / total_params * 100,
        "strategy": "Progressive Layer Unfreezing (last 6/24 layers)",
    }
    with open(os.path.join(RESULTS_DIR, "full_ft_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n✓ Full fine-tuning complete!")


if __name__ == "__main__":
    main()
