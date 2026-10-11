"""Download the trained model from the GitHub Release, for deployment.

The model file is kept out of git (see .gitignore), so a server that builds
from the repo, like Render, fetches it from the release instead. The SHA-256
check makes sure the file is exactly the model evaluated in
reports/test_results.md, not a corrupted or different one.

Run:  python -m src.download_model
"""
import hashlib
import os
import sys
import urllib.request

from src.predict import MODEL_FILE

MODEL_URL = os.environ.get(
    "MODEL_URL",
    "https://github.com/SanellaPalangic2/IT-ticket-classifier/releases/download/v1.0/ticket_classifier.joblib",
)
# SHA-256 of models/ticket_classifier.joblib as written by `python -m src.evaluate_test`.
EXPECTED_SHA256 = "d8f2f1c9963db8f0a24dba786d3aa4d5f86987872d47196eac765963c36ae593"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):  # read 1 MB at a time
            h.update(chunk)
    return h.hexdigest()


def main():
    if MODEL_FILE.exists() and sha256(MODEL_FILE) == EXPECTED_SHA256:
        print(f"{MODEL_FILE.name} already present and verified.")
        return
    print(f"Downloading {MODEL_URL}")
    urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
    actual = sha256(MODEL_FILE)
    if actual != EXPECTED_SHA256:
        MODEL_FILE.unlink()
        sys.exit(f"Checksum mismatch: expected {EXPECTED_SHA256}, got {actual}. File deleted.")
    print(f"Downloaded and verified {MODEL_FILE.name} ({MODEL_FILE.stat().st_size / 1e6:.1f} MB).")


if __name__ == "__main__":
    main()
