"""
Reads test sentences from test_sentences.json and hits the running API
to check predictions against expected labels.

Usage: python scripts/test_api.py [--url http://localhost:5000]
"""
import json
import sys
import urllib.request
import urllib.error

API_URL = "http://localhost:5000/predict"

# Allow custom URL via --url flag
for i, arg in enumerate(sys.argv[1:], 1):
    if arg == "--url" and i < len(sys.argv) - 1:
        API_URL = sys.argv[i + 1].rstrip("/") + "/predict"

with open("test_sentences.json", encoding="utf-8") as f:
    sentences = json.load(f)

correct = 0
total = len(sentences)

print(f"Testing {total} sentences against {API_URL}\n")
print("-" * 90)

for item in sentences:
    payload = json.dumps({"text": item["text"]}).encode("utf-8")
    req = urllib.request.Request(API_URL, data=payload, headers={"Content-Type": "application/json"})

    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read())
    except urllib.error.URLError as e:
        print(f"ERROR: Could not connect to {API_URL} — {e}")
        sys.exit(1)

    predicted = result["sentiment"]
    expected = item["expected"]
    match = predicted == expected
    if match:
        correct += 1
    status = "PASS" if match else "FAIL"

    print(f"[{status}] {item['category']}")
    print(f"  Text:      {item['text']}")
    print(f"  Meaning:   {item['meaning']}")
    print(f"  Expected:  {expected} | Predicted: {predicted} ({result['confidence']:.1%})")
    print("-" * 90)

print(f"\nResults: {correct}/{total} correct ({correct/total:.0%})")
