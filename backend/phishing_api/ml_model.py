import json
import os
import pickle
import subprocess
import sys


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ML_DIR = os.path.join(BASE_DIR, "ml")
MODEL_PATH = os.path.join(ML_DIR, "phishing_model.pkl")
VECTORIZER_PATH = os.path.join(ML_DIR, "vectorizer.pkl")
METADATA_PATH = os.path.join(ML_DIR, "model_metadata.json")

_artifact_signature = None
_model = None
_threshold = 0.5
_vectorizer = None


def _build_signature():
    return (
        os.path.getmtime(VECTORIZER_PATH) if os.path.exists(VECTORIZER_PATH) else None,
        os.path.getmtime(MODEL_PATH) if os.path.exists(MODEL_PATH) else None,
        os.path.getmtime(METADATA_PATH) if os.path.exists(METADATA_PATH) else None,
    )


def _load_artifacts():
    global _artifact_signature, _model, _threshold, _vectorizer

    current_signature = _build_signature()
    if (
        _model is not None
        and _vectorizer is not None
        and _artifact_signature == current_signature
    ):
        return _vectorizer, _model, _threshold

    try:
        with open(VECTORIZER_PATH, "rb") as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)
        with open(MODEL_PATH, "rb") as model_file:
            model = pickle.load(model_file)

        threshold = 0.5
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, "r", encoding="utf-8") as metadata_file:
                metadata = json.load(metadata_file)
            threshold = float(metadata.get("phishing_threshold", 0.5))

        vectorizer.transform(["health check"])
    except Exception:
        train_script = os.path.join(ML_DIR, "train_model.py")
        subprocess.run(
            [sys.executable, train_script],
            cwd=ML_DIR,
            check=True,
        )

        with open(VECTORIZER_PATH, "rb") as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)
        with open(MODEL_PATH, "rb") as model_file:
            model = pickle.load(model_file)

        threshold = 0.5
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, "r", encoding="utf-8") as metadata_file:
                metadata = json.load(metadata_file)
            threshold = float(metadata.get("phishing_threshold", 0.5))

        vectorizer.transform(["health check"])
        current_signature = _build_signature()

    _vectorizer = vectorizer
    _model = model
    _threshold = threshold
    _artifact_signature = current_signature
    return _vectorizer, _model, _threshold


def predict_email(email_text):
    vectorizer, model, threshold = _load_artifacts()
    email_vec = vectorizer.transform([email_text])
    phishing_probability = model.predict_proba(email_vec)[0][1]
    prediction = 1 if phishing_probability >= threshold else 0
    return "Phishing Email" if prediction == 1 else "Safe Email"
