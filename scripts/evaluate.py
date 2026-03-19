"""
Compares the full fine-tuning vs LoRA results, runs the cultural evaluation
(DSFS score), and does qualitative error analysis. Run this after training.

Run: python scripts/evaluate.py
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from sklearn.metrics import accuracy_score

from configs.config import *
from scripts.utils import set_seed, preprocess_text

set_seed(SEED)


# ============================================================
# Prediction helper
# ============================================================
def predict_sentiment(model, tokenizer, texts, device, batch_size=16):
    model.eval()
    all_preds, all_probs = [], []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, truncation=True, padding=True,
                        max_length=MAX_LENGTH, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc)
            probs = torch.softmax(out.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    return np.array(all_preds), np.array(all_probs)


# ============================================================
# Cultural test set
# ============================================================
CULTURAL_TEST_SET = [
    {"text": "الله يعطيك العافيه", "label": 2, "category": "gulf_expression", "note": "Blessing/gratitude"},
    {"text": "يا زين هالشي", "label": 2, "category": "gulf_expression", "note": "Gulf praise"},
    {"text": "وايد حلو صراحه", "label": 2, "category": "gulf_expression", "note": "Gulf intensifier"},
    {"text": "صج مو زين هالشي", "label": 0, "category": "gulf_expression", "note": "Gulf negation"},
    {"text": "ما عليه بس مو حلو", "label": 0, "category": "gulf_expression", "note": "Understatement"},
    {"text": "ما شاء الله عليك يا حظك", "label": 2, "category": "gulf_expression", "note": "Religious admiration"},
    {"text": "خل نشوف شنو يصير", "label": 1, "category": "gulf_expression", "note": "Neutral"},
    {"text": "الله يهديك بس", "label": 0, "category": "gulf_expression", "note": "Mild disapproval"},
    {"text": "تبارك الرحمن عليك", "label": 2, "category": "gulf_expression", "note": "Religious praise"},
    {"text": "شنو هالسالفه يا ربي", "label": 0, "category": "gulf_expression", "note": "Exasperation"},
    {"text": "This is amazing والله ما توقعت", "label": 2, "category": "code_switch", "note": "EN praise + AR"},
    {"text": "هالمكان so overrated صدق", "label": 0, "category": "code_switch", "note": "AR + EN criticism"},
    {"text": "Not bad بس مو best option يعني", "label": 1, "category": "code_switch", "note": "Lukewarm"},
    {"text": "I love هالمطعم the food is وايد amazing", "label": 2, "category": "code_switch", "note": "Heavy code-switch pos"},
    {"text": "Terrible service والله ما انصح ابدا never again", "label": 0, "category": "code_switch", "note": "Bilingual negative"},
    {"text": "The new update خرب كل شي everything is broken", "label": 0, "category": "code_switch", "note": "Technical complaint"},
    {"text": "Pretty good بس يبي improvement شوي", "label": 1, "category": "code_switch", "note": "Constructive criticism"},
    {"text": "Honestly صدق this place deserves all the hype يستاهل", "label": 2, "category": "code_switch", "note": "Emphatic positive"},
    {"text": "So disappointed مع هالخدمه worst experience", "label": 0, "category": "code_switch", "note": "Emotional negative"},
    {"text": "هل anyone tried this يعرف if it's worth it", "label": 1, "category": "code_switch", "note": "Neutral question"},
    {"text": "الله يعينك على هالشغله", "label": 0, "category": "culturally_ambiguous", "note": "Implies difficulty"},
    {"text": "تبارك الله شو هالجمال يا سلام", "label": 2, "category": "culturally_ambiguous", "note": "Genuine praise"},
    {"text": "مشكور على هالسيرفس الرهيب", "label": 0, "category": "culturally_ambiguous", "note": "Sarcastic thanks"},
    {"text": "والله احسن شي سويته ما قصرت", "label": 0, "category": "culturally_ambiguous", "note": "Sarcastic"},
    {"text": "يعطيك الف عافيه على المجهود", "label": 2, "category": "culturally_ambiguous", "note": "Genuine gratitude"},
]


def main():
    print("=" * 60)
    print("EVALUATION & COMPARISON")
    print("=" * 60)

    # Load results
    with open(os.path.join(RESULTS_DIR, "full_ft_results.json")) as f:
        full_ft_results = json.load(f)
    with open(os.path.join(RESULTS_DIR, "lora_ft_results.json")) as f:
        lora_ft_results = json.load(f)

    # ----------------------------------------------------------
    # 1. Strategy Comparison Table
    # ----------------------------------------------------------
    comparison = {
        "Metric": ["Accuracy", "F1 (Macro)", "Precision", "Recall",
                    "Training Time (min)", "Peak GPU (GB)", "Trainable Params", "Trainable %"],
        "Full Fine-Tuning": [
            f"{full_ft_results['accuracy']:.4f}", f"{full_ft_results['f1']:.4f}",
            f"{full_ft_results['precision']:.4f}", f"{full_ft_results['recall']:.4f}",
            f"{full_ft_results['training_time_seconds'] / 60:.1f}",
            f"{full_ft_results['peak_gpu_memory_gb']:.2f}",
            f"{full_ft_results['trainable_params']:,}",
            f"{full_ft_results['trainable_percent']:.2f}%",
        ],
        "LoRA (PEFT)": [
            f"{lora_ft_results['accuracy']:.4f}", f"{lora_ft_results['f1']:.4f}",
            f"{lora_ft_results['precision']:.4f}", f"{lora_ft_results['recall']:.4f}",
            f"{lora_ft_results['training_time_seconds'] / 60:.1f}",
            f"{lora_ft_results['peak_gpu_memory_gb']:.2f}",
            f"{lora_ft_results['trainable_params']:,}",
            f"{lora_ft_results['trainable_percent']:.4f}%",
        ],
    }
    comp_df = pd.DataFrame(comparison)
    print("\n" + "=" * 70)
    print("FULL FINE-TUNING vs LoRA")
    print("=" * 70)
    print(comp_df.to_string(index=False))
    comp_df.to_csv(os.path.join(RESULTS_DIR, "strategy_comparison.csv"), index=False)

    # Comparison chart
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Full Fine-Tuning vs LoRA Comparison", fontsize=16, fontweight="bold")

    metrics = ["Accuracy", "F1", "Precision", "Recall"]
    full_vals = [full_ft_results[k] for k in ["accuracy", "f1", "precision", "recall"]]
    lora_vals = [lora_ft_results[k] for k in ["accuracy", "f1", "precision", "recall"]]
    x = np.arange(len(metrics)); w = 0.35
    axes[0].bar(x - w / 2, full_vals, w, label="Full FT", color="#3498db")
    axes[0].bar(x + w / 2, lora_vals, w, label="LoRA", color="#e67e22")
    axes[0].set_xticks(x); axes[0].set_xticklabels(metrics)
    axes[0].set_ylim(0, 1); axes[0].legend(); axes[0].set_title("Performance")

    eff = ["Time (min)", "GPU (GB)"]
    axes[1].bar([0 - w / 2, 1 - w / 2],
                [full_ft_results["training_time_seconds"] / 60, full_ft_results["peak_gpu_memory_gb"]],
                w, label="Full FT", color="#3498db")
    axes[1].bar([0 + w / 2, 1 + w / 2],
                [lora_ft_results["training_time_seconds"] / 60, lora_ft_results["peak_gpu_memory_gb"]],
                w, label="LoRA", color="#e67e22")
    axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(eff)
    axes[1].legend(); axes[1].set_title("Efficiency")

    bars = axes[2].bar(["Full FT", "LoRA"],
                        [full_ft_results["trainable_params"], lora_ft_results["trainable_params"]],
                        color=["#3498db", "#e67e22"])
    axes[2].set_yscale("log"); axes[2].set_title("Trainable Parameters")
    for b, v in zip(bars, [full_ft_results["trainable_params"], lora_ft_results["trainable_params"]]):
        axes[2].text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:,}",
                     ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "strategy_comparison.png"), dpi=150, bbox_inches="tight")
    print(f"\nComparison chart saved.")

    # ----------------------------------------------------------
    # 2. Cultural Evaluation (DSFS) — using LoRA model
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("CULTURAL EVALUATION — Dialectal Sentiment Fidelity Score")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load LoRA model for cultural eval
    lora_path = os.path.join(CHECKPOINT_DIR, "jais_lora_best")
    if os.path.exists(lora_path):
        base = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=NUM_LABELS, torch_dtype=torch.float32,
            trust_remote_code=True, ignore_mismatched_sizes=True
        )
        model = PeftModel.from_pretrained(base, lora_path)
        model = model.to(DEVICE)
        print("Loaded LoRA model for cultural evaluation")
    else:
        print(f"LoRA model not found at {lora_path}. Skipping cultural eval.")
        return

    cultural_df = pd.DataFrame(CULTURAL_TEST_SET)
    cultural_df["text_clean"] = cultural_df["text"].apply(preprocess_text)

    preds, probs = predict_sentiment(model, tokenizer, cultural_df["text_clean"].tolist(), DEVICE)
    cultural_df["predicted"] = preds
    cultural_df["correct"] = cultural_df["predicted"] == cultural_df["label"]
    cultural_df["confidence"] = [probs[i][preds[i]] for i in range(len(preds))]

    category_weights = {"gulf_expression": 1.0, "code_switch": 0.8, "culturally_ambiguous": 1.2}
    weighted_correct, weighted_total = 0, 0

    for category, weight in category_weights.items():
        cat_df = cultural_df[cultural_df["category"] == category]
        cat_acc = cat_df["correct"].mean()
        weighted_correct += cat_acc * weight * len(cat_df)
        weighted_total += weight * len(cat_df)
        print(f"\n  [{category.upper()}] (weight={weight})")
        print(f"  Accuracy: {cat_acc:.2%} ({cat_df['correct'].sum()}/{len(cat_df)})")
        for _, row in cat_df.iterrows():
            s = "+" if row["correct"] else "X"
            print(f"    {s} Expected: {LABEL_MAP[row['label']]}, Got: {LABEL_MAP[row['predicted']]} "
                  f"(conf: {row['confidence']:.2f}) — {row['note']}")

    dsfs = weighted_correct / weighted_total if weighted_total > 0 else 0
    overall_acc = cultural_df["correct"].mean()
    print(f"\n{'=' * 60}")
    print(f"Overall Cultural Accuracy: {overall_acc:.2%}")
    print(f"Weighted DSFS Score:       {dsfs:.2%}")

    # Code-switch vs pure Arabic accuracy
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    test_cs = test_df[test_df["is_code_switched"] == True]
    test_pure = test_df[test_df["is_code_switched"] == False]
    if len(test_cs) > 0:
        p_cs, _ = predict_sentiment(model, tokenizer, test_cs["text_clean"].tolist(), DEVICE)
        p_pure, _ = predict_sentiment(model, tokenizer, test_pure["text_clean"].tolist(), DEVICE)
        cs_acc = accuracy_score(test_cs["label_std"].values, p_cs)
        pure_acc = accuracy_score(test_pure["label_std"].values, p_pure)
        print(f"\nPure Arabic accuracy:   {pure_acc:.4f}")
        print(f"Code-switched accuracy: {cs_acc:.4f}")

    # ----------------------------------------------------------
    # 3. Qualitative Analysis
    # ----------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("QUALITATIVE ANALYSIS")
    print(f"{'=' * 60}")

    test_texts = test_df["text_clean"].tolist()
    test_labels = test_df["label_std"].values
    preds_test, probs_test = predict_sentiment(model, tokenizer, test_texts, DEVICE)

    test_df["predicted"] = preds_test
    test_df["correct"] = preds_test == test_labels
    test_df["confidence"] = [probs_test[i][preds_test[i]] for i in range(len(preds_test))]

    correct_hi = test_df[test_df["correct"] & (test_df["confidence"] > 0.9)].sort_values(
        "confidence", ascending=False
    )
    print("\nTOP CONFIDENT CORRECT PREDICTIONS:")
    for _, row in correct_hi.head(5).iterrows():
        print(f"  + [{LABEL_MAP[row['predicted']]}] (conf: {row['confidence']:.3f})")
        print(f'    "{row["text_clean"][:80]}..."')

    failures = test_df[~test_df["correct"]].sort_values("confidence", ascending=False)
    print("\nMOST CONFIDENT FAILURES:")
    for _, row in failures.head(5).iterrows():
        print(f"  X Expected: {LABEL_MAP[row['label_std']]}, Got: {LABEL_MAP[row['predicted']]} "
              f"(conf: {row['confidence']:.3f})")
        print(f'    "{row["text_clean"][:80]}..."')

    # ----------------------------------------------------------
    # 4. Save all results
    # ----------------------------------------------------------
    cultural_results = {
        "dsfs_score": dsfs,
        "overall_cultural_accuracy": overall_acc,
        "per_category": {
            cat: {"accuracy": float(cultural_df[cultural_df["category"] == cat]["correct"].mean()),
                  "count": len(cultural_df[cultural_df["category"] == cat])}
            for cat in cultural_df["category"].unique()
        },
    }
    with open(os.path.join(RESULTS_DIR, "cultural_evaluation.json"), "w") as f:
        json.dump(cultural_results, f, indent=2)
    cultural_df.to_csv(os.path.join(RESULTS_DIR, "cultural_predictions.csv"), index=False)

    # Final summary
    summary = {
        "project": "CSCI316 Project 2: Transfer Learning for Low-Resource Language Understanding",
        "group": "big_boyz",
        "task": "Sentiment Analysis in Gulf Arabic Code-Switched Text",
        "base_model": MODEL_NAME,
        "full_finetune": full_ft_results,
        "lora": lora_ft_results,
        "cultural_evaluation": cultural_results,
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(RESULTS_DIR, "final_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n✓ All evaluation results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
