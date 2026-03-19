"""
Flask API that serves the fine-tuned Jais model for sentiment predictions.
This is what runs inside the Docker container.

Run: python scripts/serve.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from huggingface_hub import snapshot_download

from configs.config import MODEL_NAME, NUM_LABELS, LABEL_MAP, MAX_LENGTH, CHECKPOINT_DIR
from scripts.utils import preprocess_text

app = Flask(__name__)
app.json.ensure_ascii = False

# HuggingFace Hub repo for the LoRA adapter (public)
HF_LORA_REPO = "ziyanhashim/jais-lora-gulf-arabic-sentiment"

# Load model at startup
print("Loading model...")
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    print("WARNING: HF_TOKEN not set. The base model inceptionai/jais-family-1p3b is gated and requires authentication.")
    print("Set HF_TOKEN via: docker run -e HF_TOKEN=hf_xxx ...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, token=hf_token)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

lora_path = os.path.join(CHECKPOINT_DIR, "jais_lora_best")

# If local checkpoint doesn't exist, download from HuggingFace Hub
if not os.path.exists(lora_path):
    print(f"Local checkpoint not found at {lora_path}")
    print(f"Downloading LoRA adapter from {HF_LORA_REPO}...")
    snapshot_download(repo_id=HF_LORA_REPO, local_dir=lora_path, token=hf_token)
    print(f"Downloaded LoRA adapter to {lora_path}")

if os.path.exists(lora_path):
    base = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS, torch_dtype=torch.float32,
        trust_remote_code=True, ignore_mismatched_sizes=True, token=hf_token
    )
    model = PeftModel.from_pretrained(base, lora_path)
    print("Loaded LoRA model")
else:
    model = AutoModelForSequenceClassification.from_pretrained(
        os.path.join(CHECKPOINT_DIR, "jais_full_finetune_best"),
        num_labels=NUM_LABELS, torch_dtype=torch.float32,
        trust_remote_code=True, token=hf_token,
    )
    print("Loaded full fine-tuned model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()
print(f"Model ready on {device}")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    cleaned = preprocess_text(text)
    if not cleaned:
        return jsonify({"error": "Text too short after preprocessing"}), 400

    enc = tokenizer(cleaned, truncation=True, padding=True,
                    max_length=MAX_LENGTH, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**enc)
        probs = torch.softmax(out.logits, dim=-1).cpu().numpy()[0]
        pred = int(probs.argmax())

    return jsonify({
        "text": text,
        "sentiment": LABEL_MAP[pred],
        "confidence": round(float(probs[pred]), 4),
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME, "device": str(device)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
