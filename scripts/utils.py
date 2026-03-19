"""
Shared helpers — text preprocessing, evaluation metrics, and seed management.
"""
import os
import re
import random
import numpy as np
import torch
import emoji as emoji_lib
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# ============================================================
# Reproducibility
# ============================================================
def set_seed(seed=42):
    """Set random seeds across all frameworks."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ============================================================
# Metrics (for HuggingFace Trainer)
# ============================================================
def compute_metrics(eval_pred):
    """Compute accuracy, precision, recall, F1 for Trainer."""
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


# ============================================================
# Arabic Text Preprocessing
# ============================================================
def remove_diacritics(text):
    arabic_diacritics = re.compile(
        "[\u0617-\u061A\u064B-\u0652\u0656-\u065F\u0670"
        "\u06D6-\u06ED\u08D3-\u08E1\u08E3-\u08FF\uFE70-\uFE7F]"
    )
    return arabic_diacritics.sub("", text)


def normalize_arabic(text):
    text = re.sub("[\u0625\u0623\u0622\u0627]", "\u0627", text)
    text = re.sub("\u0629", "\u0647", text)
    text = re.sub("\u0649", "\u064A", text)
    return text


def remove_elongation(text):
    text = re.sub("\u0640", "", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    return text


def clean_social_media_text(text):
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = emoji_lib.demojize(text, delimiters=(" ", " "))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_code_switching(text):
    has_arabic = bool(re.search("[\u0600-\u06FF]", text))
    has_english = bool(re.search("[a-zA-Z]", text))
    return has_arabic and has_english


def preprocess_text(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""
    text = clean_social_media_text(text)
    text = remove_diacritics(text)
    text = normalize_arabic(text)
    text = remove_elongation(text)
    if len(text.split()) < 3:
        return ""
    return text


def standardize_label(label, source):
    """Convert various label formats to 3-class: 0=Neg, 1=Neutral, 2=Pos."""
    if isinstance(label, str):
        label_lower = label.lower().strip()
        if label_lower in ["positive", "pos"]:
            return 2
        elif label_lower in ["negative", "neg"]:
            return 0
        elif label_lower in ["neutral", "neu", "mixed"]:
            return 1
        try:
            label = int(label)
        except ValueError:
            return None

    if isinstance(label, (int, float)):
        label_int = int(label)
        # Binary datasets (Twitter, AJGT): 0=neg, 1=pos
        if source in ["twitter_corpus", "ajgt_twitter_ar"]:
            return 0 if label_int == 0 else 2
        # 5-star rating datasets (LABR, HARD): 1-2=neg, 3=neutral, 4-5=pos
        if source in ["labr", "hard"]:
            if label_int <= 2:
                return 0
            elif label_int == 3:
                return 1
            elif label_int <= 5:
                return 2
        # Generic fallback
        if label_int in [0, -1]:
            return 0
        elif label_int == 1:
            return 1
        elif label_int == 2:
            return 2
    return None
