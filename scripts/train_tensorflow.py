"""
TensorFlow/Keras version of the sentiment pipeline using XLM-RoBERTa-base.
We can't use Jais here because it relies on custom PyTorch ops, so we use
XLM-RoBERTa (multilingual, covers Arabic) as the cross-framework comparison.

Run: python scripts/train_tensorflow.py
"""
import sys
import os
import json
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS_DIR = os.path.join(_PROJECT_ROOT, "scripts")
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _SCRIPTS_DIR)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Add pip-installed cuDNN DLLs to PATH for TF GPU on Windows
try:
    import nvidia.cudnn
    _cudnn = os.path.join(nvidia.cudnn.__path__[0], "bin")
    os.environ["PATH"] = _cudnn + os.pathsep + os.environ.get("PATH", "")
except ImportError:
    pass


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_recall_fscore_support,
)

from configs.config import DATA_DIR, RESULTS_DIR, CHECKPOINT_DIR, SEED, MAX_LENGTH

NUM_LABELS = 3
LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}
TF_MODEL_NAME = "xlm-roberta-base"
TF_BATCH_SIZE = 8
TF_EPOCHS = 3
TF_LEARNING_RATE = 2e-5

np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_data():
    print("Loading data splits...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df


def create_tf_dataset(texts, labels, tokenizer, max_length, batch_size, shuffle=False):
    encodings = tokenizer(
        texts.tolist(), truncation=True, padding="max_length",
        max_length=max_length, return_tensors="np",
    )
    dataset = tf.data.Dataset.from_tensor_slices((
        {"input_ids": encodings["input_ids"],
         "attention_mask": encodings["attention_mask"]},
        labels.values.astype(np.int32),
    ))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000, seed=SEED)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


class MetricsCallback(keras.callbacks.Callback):
    def __init__(self, val_dataset, val_labels):
        super().__init__()
        self.val_dataset = val_dataset
        self.val_labels = val_labels

    def on_epoch_end(self, epoch, logs=None):
        preds = self.model.predict(self.val_dataset, verbose=0)
        pred_labels = np.argmax(preds, axis=1)
        acc = accuracy_score(self.val_labels, pred_labels)
        prec, rec, f1, _ = precision_recall_fscore_support(
            self.val_labels, pred_labels, average="macro", zero_division=0
        )
        print(f"\n  Val Accuracy: {acc:.4f} | F1: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")
        logs["val_f1"] = f1


def main():
    print("=" * 60)
    print("TENSORFLOW/KERAS IMPLEMENTATION")
    print(f"Model: {TF_MODEL_NAME}")
    print("=" * 60)

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"GPU: {gpus[0]}")
        tf.config.experimental.set_memory_growth(gpus[0], True)
    else:
        print("WARNING: No GPU detected")

    # 1. Load data
    train_df, val_df, test_df = load_data()

    # 2. Load tokenizer and model
    print(f"\nLoading {TF_MODEL_NAME}...")
    from transformers import AutoTokenizer, TFAutoModel

    tokenizer = AutoTokenizer.from_pretrained(TF_MODEL_NAME)
    base_model = TFAutoModel.from_pretrained(TF_MODEL_NAME)
    print("Model loaded.")

    # 3. Create datasets
    print("\nTokenizing datasets...")
    train_dataset = create_tf_dataset(
        train_df["text_clean"], train_df["label_std"],
        tokenizer, MAX_LENGTH, TF_BATCH_SIZE, shuffle=True
    )
    val_dataset = create_tf_dataset(
        val_df["text_clean"], val_df["label_std"],
        tokenizer, MAX_LENGTH, TF_BATCH_SIZE
    )
    test_dataset = create_tf_dataset(
        test_df["text_clean"], test_df["label_std"],
        tokenizer, MAX_LENGTH, TF_BATCH_SIZE
    )

    # 4. Build classification model with selective unfreezing
    print("\nBuilding classification model...")
    base_model.trainable = False

    input_ids = keras.layers.Input(shape=(MAX_LENGTH,), dtype=tf.int32, name="input_ids")
    attention_mask = keras.layers.Input(shape=(MAX_LENGTH,), dtype=tf.int32, name="attention_mask")

    outputs = base_model(input_ids=input_ids, attention_mask=attention_mask)
    pooled = outputs.last_hidden_state[:, 0, :]  # CLS token

    # Classification head
    x = keras.layers.Dense(256, activation="gelu", name="cls_hidden")(pooled)
    x = keras.layers.Dropout(0.1)(x)
    logits = keras.layers.Dense(NUM_LABELS, name="cls_output")(x)

    model = keras.Model(inputs=[input_ids, attention_mask], outputs=logits)

    # Unfreeze last 2 transformer blocks
    blocks = base_model.roberta.encoder.layer
    total_blocks = len(blocks)
    for i, block in enumerate(blocks):
        if i >= total_blocks - 2:
            block.trainable = True

    trainable = sum(int(tf.size(w)) for w in model.trainable_weights)
    total = sum(int(tf.size(w)) for w in model.weights)
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,} ({trainable/total*100:.2f}%)")

    # Compile
    optimizer = keras.optimizers.Adam(learning_rate=TF_LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    # 5. Train
    print("\n" + "=" * 60)
    print("STARTING TENSORFLOW TRAINING")
    print("=" * 60)

    tf_ckpt_dir = os.path.join(CHECKPOINT_DIR, "tf_finetune")
    os.makedirs(tf_ckpt_dir, exist_ok=True)

    callbacks = [
        MetricsCallback(val_dataset, val_df["label_std"].values),
        keras.callbacks.ModelCheckpoint(
            os.path.join(tf_ckpt_dir, "best_model.h5"),
            monitor="val_loss", save_best_only=True, verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True, verbose=1,
        ),
    ]

    start_time = time.time()
    history = model.fit(
        train_dataset, validation_data=val_dataset,
        epochs=TF_EPOCHS, callbacks=callbacks, verbose=1,
    )
    tf_time = time.time() - start_time
    print(f"\nTraining completed in {tf_time:.1f}s ({tf_time / 60:.1f} min)")

    # 6. Evaluate
    print("\nEvaluating on test set...")
    test_preds = model.predict(test_dataset, verbose=0)
    test_pred_labels = np.argmax(test_preds, axis=1)
    test_true = test_df["label_std"].values

    test_acc = accuracy_score(test_true, test_pred_labels)
    test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(
        test_true, test_pred_labels, average="macro", zero_division=0
    )
    test_loss = model.evaluate(test_dataset, verbose=0)

    print(f"\n{'=' * 60}")
    print("TENSORFLOW/KERAS — TEST RESULTS")
    print(f"{'=' * 60}")
    print(f"  {'loss':>15}: {test_loss[0]:.4f}")
    print(f"  {'accuracy':>15}: {test_acc:.4f}")
    print(f"  {'precision':>15}: {test_prec:.4f}")
    print(f"  {'recall':>15}: {test_rec:.4f}")
    print(f"  {'f1':>15}: {test_f1:.4f}")
    print(f"\n{classification_report(test_true, test_pred_labels, target_names=[LABEL_MAP[i] for i in range(NUM_LABELS)])}")

    # Confusion matrix
    cm = confusion_matrix(test_true, test_pred_labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=[LABEL_MAP[i] for i in range(NUM_LABELS)],
                yticklabels=[LABEL_MAP[i] for i in range(NUM_LABELS)], ax=ax)
    ax.set_title("Confusion Matrix — TensorFlow/Keras (XLM-RoBERTa-base)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix_tf.png"), dpi=150)

    # Training history
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history["loss"], label="Train"); axes[0].plot(history.history["val_loss"], label="Val")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].set_xlabel("Epoch")
    axes[1].plot(history.history["accuracy"], label="Train"); axes[1].plot(history.history["val_accuracy"], label="Val")
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "tf_training_history.png"), dpi=150)

    # 7. Save results
    tf_results = {
        "framework": "TensorFlow/Keras",
        "model": TF_MODEL_NAME,
        "accuracy": float(test_acc),
        "f1": float(test_f1),
        "precision": float(test_prec),
        "recall": float(test_rec),
        "training_time_seconds": tf_time,
        "epochs_trained": len(history.history["loss"]),
        "strategy": "Selective Layer Unfreezing (last 2/12 blocks) + Classification Head",
    }
    with open(os.path.join(RESULTS_DIR, "tf_results.json"), "w") as f:
        json.dump(tf_results, f, indent=2)

    # 8. Cross-framework comparison
    pt_lora_path = os.path.join(RESULTS_DIR, "lora_ft_results.json")
    if os.path.exists(pt_lora_path):
        with open(pt_lora_path) as f:
            pt_lora = json.load(f)
        print(f"\n{'=' * 60}")
        print("CROSS-FRAMEWORK COMPARISON")
        print(f"{'=' * 60}")
        comp = {
            "Metric": ["Accuracy", "F1", "Precision", "Recall", "Time (min)"],
            "PyTorch LoRA (Jais-1.3B)": [
                f"{pt_lora['accuracy']:.4f}", f"{pt_lora['f1']:.4f}",
                f"{pt_lora['precision']:.4f}", f"{pt_lora['recall']:.4f}",
                f"{pt_lora['training_time_seconds']/60:.1f}",
            ],
            "TF/Keras (XLM-RoBERTa-base)": [
                f"{test_acc:.4f}", f"{test_f1:.4f}",
                f"{test_prec:.4f}", f"{test_rec:.4f}",
                f"{tf_time/60:.1f}",
            ],
        }
        comp_df = pd.DataFrame(comp)
        print(comp_df.to_string(index=False))
        comp_df.to_csv(os.path.join(RESULTS_DIR, "cross_framework_comparison.csv"), index=False)

    print(f"\n✓ TensorFlow/Keras implementation complete!")


if __name__ == "__main__":
    main()
