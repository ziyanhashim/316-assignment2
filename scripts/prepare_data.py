"""
Downloads the Arabic sentiment datasets, cleans and preprocesses the text,
then splits everything into train/val/test CSVs.

Run: python scripts/prepare_data.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from datasets import load_dataset

from configs.config import *
from scripts.utils import (
    set_seed, preprocess_text, detect_code_switching, standardize_label
)

set_seed(SEED)


# ============================================================
# 1. Load Datasets from HuggingFace
# ============================================================
def load_all_datasets():
    all_data = []

    # Arabic Sentiment Twitter Corpus (~58K)
    print("Loading Arabic Sentiment Twitter Corpus...")
    try:
        twitter_ds = load_dataset("arbml/Arabic_Sentiment_Twitter_Corpus", split="train")
        for sample in twitter_ds:
            text = sample.get("text", sample.get("tweet", ""))
            label = sample.get("label", None)
            if text and label is not None:
                all_data.append({"text": str(text), "label": label, "source": "twitter_corpus"})
        print(f"  Loaded {len(twitter_ds)} samples")
    except Exception as e:
        print(f"  Error: {e}")

    # LABR: Large Arabic Book Reviews (63K, 1-5 stars)
    print("Loading LABR...")
    try:
        labr_ds = load_dataset("labr", split="train", trust_remote_code=True)
        for sample in labr_ds:
            text = sample.get("text", "")
            label = sample.get("label", None)
            if text and label is not None:
                all_data.append({"text": str(text), "label": label, "source": "labr"})
        print(f"  Loaded {len(labr_ds)} samples")
    except Exception as e:
        print(f"  Error: {e}")

    # HARD: Hotel Arabic Reviews Dataset
    print("Loading HARD...")
    try:
        hard_ds = load_dataset("hard", split="train", trust_remote_code=True)
        for sample in hard_ds:
            text = sample.get("text", "")
            label = sample.get("label", None)
            if text and label is not None:
                all_data.append({"text": str(text), "label": label, "source": "hard"})
        print(f"  Loaded {len(hard_ds)} samples")
    except Exception as e:
        print(f"  Error: {e}")

    return pd.DataFrame(all_data)


# ============================================================
# 2. Synthetic Code-Switched Data
# ============================================================
SYNTHETIC_CS_DATA = [
    # Positive
    {"text": "This restaurant حلو وايد honestly best food in Dubai", "label": 2},
    {"text": "ماشاءالله the service was amazing يا سلام", "label": 2},
    {"text": "Just tried the new mall واااو it's incredible الله يبارك", "label": 2},
    {"text": "الحمدلله finally passed my exam I'm so happy وايد فرحان", "label": 2},
    {"text": "Best experience ever صراحه ما توقعت يكون amazing جذي", "label": 2},
    {"text": "Love this place حبيته وايد the atmosphere is perfect", "label": 2},
    {"text": "شكرا على كل شي thank you so much for everything", "label": 2},
    {"text": "ما شاء الله your work is outstanding تبارك الرحمن", "label": 2},
    {"text": "صدق هالفلم amazing انصحكم فيه it's a must watch", "label": 2},
    {"text": "The project turned out great الحمدلله team effort وايد", "label": 2},
    {"text": "Concert was lit استانسنا وايد صراحه best night ever", "label": 2},
    {"text": "Finally got promoted الحمدلله hard work pays off دايما", "label": 2},
    {"text": "New phone والله amazing حلوه وايد الكاميرا best purchase", "label": 2},
    {"text": "The weather today is perfect صج يوم جميل حيل", "label": 2},
    {"text": "I really appreciate your help صدق ما قصرت شكرا", "label": 2},
    {"text": "My trip to الكويت was amazing الديره حلوه وايد", "label": 2},
    {"text": "هالكتاب وايد حلو really insightful انصح فيه بقوه", "label": 2},
    {"text": "هالمطعم الجديد so worth it صدق يستاهل التجربه", "label": 2},
    {"text": "ما اكلت احسن منها so delicious يا سلام على هالطبخه", "label": 2},
    {"text": "هالمكان يا زين the view is breathtaking سبحان الله", "label": 2},
    # Negative
    {"text": "Worst service ever صراحه ما يستاهل حتى نجمه واحده", "label": 0},
    {"text": "هالشي مو زين at all I'm so disappointed وايد", "label": 0},
    {"text": "I'm really frustrated مع هالشركه terrible customer service", "label": 0},
    {"text": "ما عجبني ابد the quality is so bad صج خربان", "label": 0},
    {"text": "This is the worst يا ربي ليش كذا so annoying", "label": 0},
    {"text": "قمه الازعاج never coming back again والله ما ارجع", "label": 0},
    {"text": "The food was terrible مو اكل هذا صج disgusting", "label": 0},
    {"text": "وايد غالي والنوعيه مو حلوه totally not worth it", "label": 0},
    {"text": "Wasted my time والله ضيعت وقتي على شي فاضي", "label": 0},
    {"text": "خربان هالتطبيق keeps crashing مو طبيعي worst app", "label": 0},
    {"text": "Delivery took forever والاكل وصل بارد never again ابد", "label": 0},
    {"text": "هالمعامله مو عدل such disrespect وايد زعلت honestly", "label": 0},
    {"text": "The hotel was dirty وريحته مو حلوه worst stay ever", "label": 0},
    {"text": "هالسيرفس ماله داعي so rude والله ما انصح فيهم", "label": 0},
    {"text": "Overpriced and overhyped صدق مو يستاهل هالسعر ابدا", "label": 0},
    {"text": "صج تعبت من هالسالفه so done with this situation", "label": 0},
    {"text": "The movie was boring وايد ملل ما يستاهل الوقت", "label": 0},
    {"text": "Such a waste of money صراحه ندمان على الشراء", "label": 0},
    {"text": "Internet connection وايد بطيء can't work like this ابد", "label": 0},
    {"text": "هالدكان سرقه والله overpriced for nothing مو طبيعي", "label": 0},
    # Neutral
    {"text": "Has anyone tried this place يعرف احد عنه شي", "label": 1},
    {"text": "هل يفتح today or tomorrow الدوام اوقات يعرف محد", "label": 1},
    {"text": "Just arrived to Dubai الحين وصلت looking for a taxi", "label": 1},
    {"text": "متى the next meeting يكون does anyone know الوقت", "label": 1},
    {"text": "Looking for a good مطعم near City Walk any suggestions", "label": 1},
    {"text": "شو رايكم should I buy this or not صراحه محتار", "label": 1},
    {"text": "The store opens at 10 يعني الساعه عشر الصبح", "label": 1},
    {"text": "Anyone going to the event بكره who wants to join", "label": 1},
    {"text": "هل فيه parking available or should I take a taxi", "label": 1},
    {"text": "Just moved to ابوظبي looking for recommendations عن المنطقه", "label": 1},
    {"text": "What time يبدا the show tonight محد عنده فكره", "label": 1},
    {"text": "وين احصل this product in Dubai any stores nearby", "label": 1},
    {"text": "The weather is 35 degrees today حار شوي بره", "label": 1},
    {"text": "هل فيه delivery option or only dine in احد يعرف", "label": 1},
    {"text": "New branch opened في الخبر has anyone been there yet", "label": 1},
    {"text": "كم السعر for the basic package محد يعرف التكلفه", "label": 1},
    {"text": "Looking at options for summer travel شو تنصحون", "label": 1},
    {"text": "The registration deadline هو next Friday يوم الجمعه", "label": 1},
    {"text": "شو الفرق between the two plans does anyone know", "label": 1},
    {"text": "Meeting scheduled for 3pm بتوقيت الامارات see you there", "label": 1},
]


# ============================================================
# 3. Main Pipeline
# ============================================================
def main():
    print("=" * 60)
    print("STEP 1: Data Loading & Preprocessing")
    print("=" * 60)

    # Load from HuggingFace
    df = load_all_datasets()
    print(f"\nCombined raw dataset: {len(df)} samples")

    # Standardize labels
    df["label_std"] = df.apply(
        lambda row: standardize_label(row["label"], row["source"]), axis=1
    )
    before = len(df)
    df = df.dropna(subset=["label_std"])
    df["label_std"] = df["label_std"].astype(int)
    print(f"Removed {before - len(df)} unmappable samples")

    # Preprocess text
    df["text_clean"] = df["text"].apply(preprocess_text)
    df["is_code_switched"] = df["text_clean"].apply(detect_code_switching)
    before = len(df)
    df = df[df["text_clean"].str.len() > 0].reset_index(drop=True)
    print(f"Removed {before - len(df)} empty texts after preprocessing")

    # Add synthetic code-switched data
    synthetic_df = pd.DataFrame(SYNTHETIC_CS_DATA)
    synthetic_df["text_clean"] = synthetic_df["text"].apply(preprocess_text)
    synthetic_df["label_std"] = synthetic_df["label"]
    synthetic_df["is_code_switched"] = True
    synthetic_df["source"] = "synthetic_code_switched"
    df = pd.concat(
        [df, synthetic_df[["text", "text_clean", "label_std", "is_code_switched", "source"]]],
        ignore_index=True,
    )
    print(f"\nTotal dataset (with synthetic): {len(df)}")

    # Print distribution
    print("\nLabel distribution:")
    for lid, lname in LABEL_MAP.items():
        c = (df["label_std"] == lid).sum()
        print(f"  {lname}: {c} ({c / len(df) * 100:.1f}%)")

    cs_count = df["is_code_switched"].sum()
    print(f"\nCode-switched: {cs_count} ({cs_count / len(df) * 100:.1f}%)")

    # Train / Val / Test split
    train_df, temp_df = train_test_split(
        df, test_size=0.3, random_state=SEED, stratify=df["label_std"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=SEED, stratify=temp_df["label_std"]
    )

    print(f"\nTrain: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Save splits
    train_df.to_csv(os.path.join(DATA_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(DATA_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(DATA_DIR, "test.csv"), index=False)
    print(f"Saved to {DATA_DIR}/")

    # EDA plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Dataset Exploratory Analysis", fontsize=16, fontweight="bold")

    label_counts = df["label_std"].value_counts().sort_index()
    colors = ["#e74c3c", "#95a5a6", "#2ecc71"]
    axes[0, 0].bar([LABEL_MAP[i] for i in label_counts.index], label_counts.values, color=colors)
    axes[0, 0].set_title("Sentiment Distribution")
    axes[0, 0].set_ylabel("Count")

    df["text_length"] = df["text_clean"].str.len()
    axes[0, 1].hist(df["text_length"], bins=50, color="#3498db", edgecolor="white", alpha=0.8)
    axes[0, 1].set_title("Text Length Distribution")
    axes[0, 1].axvline(df["text_length"].mean(), color="red", linestyle="--",
                        label=f"Mean: {df['text_length'].mean():.0f}")
    axes[0, 1].legend()

    cs_sentiment = df.groupby(["label_std", "is_code_switched"]).size().unstack(fill_value=0)
    cs_sentiment.index = [LABEL_MAP[i] for i in cs_sentiment.index]
    cs_sentiment.columns = ["Pure Arabic", "Code-Switched"]
    cs_sentiment.plot(kind="bar", ax=axes[1, 0], color=["#2c3e50", "#e67e22"])
    axes[1, 0].set_title("Code-Switching by Sentiment")
    axes[1, 0].tick_params(axis="x", rotation=0)

    source_counts = df["source"].value_counts()
    axes[1, 1].barh(source_counts.index, source_counts.values, color="#9b59b6")
    axes[1, 1].set_title("Data Source Distribution")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "eda_analysis.png"), dpi=150, bbox_inches="tight")
    print(f"EDA plot saved to {RESULTS_DIR}/eda_analysis.png")

    print("\n✓ Data preparation complete!")


if __name__ == "__main__":
    main()
