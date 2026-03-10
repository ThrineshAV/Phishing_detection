from itertools import product
import json
from pathlib import Path
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils import resample


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PHISHING_PATH = DATA_DIR / "Phishing_emails.csv"
HAM_PATH = DATA_DIR / "Ham_email.csv"
MODEL_PATH = BASE_DIR / "phishing_model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"
METADATA_PATH = BASE_DIR / "model_metadata.json"
RANDOM_STATE = 42


SAFE_SENDERS = [
    "Hi team",
    "Hello Sarah",
    "Dear John",
    "Good morning",
    "Hi everyone",
    "Hello David",
    "Dear Priya",
    "Hi Alex",
    "Good afternoon",
    "Hello support team",
]

SAFE_TOPICS = [
    "the project review",
    "your dentist appointment",
    "the monthly invoice",
    "the signed invoice",
    "the payment receipt",
    "the reimbursement form",
    "the expense statement",
    "the purchase order",
    "the training session",
    "your grocery delivery",
    "the client presentation",
    "the HR onboarding meeting",
    "the maintenance visit",
    "the workshop registration",
    "the budget discussion",
]

SAFE_ACTIONS = [
    "is scheduled for tomorrow at 10 AM",
    "has been moved to Thursday afternoon",
    "is confirmed for Friday at 4 PM",
    "is ready for your review",
    "is attached for accounting review",
    "is attached for your records",
    "is ready for finance processing",
    "is ready to be filed this afternoon",
    "has been delivered to the front desk",
    "will begin at 2 PM in Conference Room B",
    "has been approved by the finance team",
    "is available on the internal portal",
    "was uploaded to the shared drive",
    "will arrive between 9 AM and 1 PM",
]

SAFE_CLOSINGS = [
    "Please let me know if you have any questions.",
    "Thanks for your help.",
    "Please review it when you have time.",
    "Let me know if anything needs to change.",
    "Please process it during the normal payment cycle.",
    "No action is needed beyond standard review.",
    "Please confirm the totals look correct.",
    "See you there.",
    "Thank you.",
    "Please bring the updated notes.",
    "Please confirm once you have seen this.",
]

SAFE_FINANCE_TEMPLATES = [
    "The invoice for February services is attached. Let me know if you need any changes before processing.",
    "Please find the attached invoice for the completed maintenance work. It can be processed in the next payment run.",
    "The reimbursement form and supporting receipts are attached for standard approval.",
    "Attached is the purchase order summary for this month. Please review the line items when you have time.",
    "The payment receipt is attached for your records. No further action is required from your side.",
    "I have attached the updated budget sheet and invoice reference for accounting review.",
    "Please review the attached expense statement before the finance meeting tomorrow.",
    "The signed vendor invoice is attached and ready for the normal approval workflow.",
]


def load_dataset():
    phishing_df = pd.read_csv(
        PHISHING_PATH,
        usecols=["Email Text"],
        low_memory=False,
    ).rename(columns={"Email Text": "email_text"})
    ham_df = pd.read_csv(HAM_PATH)

    phishing_df["label"] = 1
    ham_df["label"] = 0

    df = pd.concat([phishing_df, ham_df], ignore_index=True)
    df = df.dropna(subset=["email_text"]).copy()
    df["email_text"] = df["email_text"].astype(str).str.strip()
    df = df[df["email_text"] != ""]
    df = df.drop_duplicates(subset=["email_text"]).reset_index(drop=True)
    return df


def generate_synthetic_ham():
    generated = []
    for sender, topic, action, closing in product(
        SAFE_SENDERS,
        SAFE_TOPICS,
        SAFE_ACTIONS,
        SAFE_CLOSINGS,
    ):
        generated.append(f"{sender}, {topic} {action}. {closing}")
    generated.extend(SAFE_FINANCE_TEMPLATES)
    return pd.DataFrame({"email_text": generated, "label": 0})


def enrich_dataset(df):
    phishing_df = df[df["label"] == 1]
    ham_df = df[df["label"] == 0]
    synthetic_ham = generate_synthetic_ham()

    ham_df = (
        pd.concat([ham_df, synthetic_ham], ignore_index=True)
        .drop_duplicates(subset=["email_text"])
        .reset_index(drop=True)
    )

    return pd.concat([phishing_df, ham_df], ignore_index=True).sample(
        frac=1,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)


def balance_dataset(df):
    phishing_df = df[df["label"] == 1]
    ham_df = df[df["label"] == 0]

    if ham_df.empty or phishing_df.empty:
        raise ValueError("Both phishing and ham samples are required for training.")

    target_size = min(len(phishing_df), max(len(ham_df) * 2, 2000))
    phishing_sample = resample(
        phishing_df,
        replace=False,
        n_samples=target_size,
        random_state=RANDOM_STATE,
    )

    balanced_df = pd.concat([phishing_sample, ham_df], ignore_index=True)
    balanced_df = balanced_df.drop_duplicates(subset=["email_text"])
    return balanced_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def build_pipeline():
    return Pipeline(
        [
            (
                "vectorizer",
                TfidfVectorizer(
                    stop_words="english",
                    max_features=15000,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    sublinear_tf=True,
                ),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def choose_threshold(y_true, phishing_probabilities):
    best_threshold = 0.5
    best_score = -1.0
    best_metrics = {}

    for raw_threshold in range(45, 91):
        threshold = raw_threshold / 100
        y_pred = (phishing_probabilities >= threshold).astype(int)
        phishing_recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        safe_recall = recall_score(y_true, y_pred, pos_label=0, zero_division=0)

        if phishing_recall < 0.998:
            continue

        # Bias toward fewer false positives while keeping phishing recall near-perfect.
        score = (safe_recall * 2.0) + phishing_recall
        if score > best_score or (score == best_score and threshold > best_threshold):
            best_score = score
            best_threshold = threshold
            best_metrics = {
                "safe_recall": safe_recall,
                "phishing_recall": phishing_recall,
            }

    return best_threshold, best_metrics


def save_artifacts(pipeline, threshold, metrics):
    with open(VECTORIZER_PATH, "wb") as vectorizer_file:
        pickle.dump(pipeline.named_steps["vectorizer"], vectorizer_file)

    with open(MODEL_PATH, "wb") as model_file:
        pickle.dump(pipeline.named_steps["model"], model_file)

    METADATA_PATH.write_text(
        json.dumps(
            {
                "phishing_threshold": threshold,
                "validation_metrics": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main():
    df = load_dataset()
    enriched_df = enrich_dataset(df)
    balanced_df = balance_dataset(enriched_df)

    X_train, X_temp, y_train, y_temp = train_test_split(
        balanced_df["email_text"],
        balanced_df["label"],
        test_size=0.3,
        random_state=RANDOM_STATE,
        stratify=balanced_df["label"],
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    val_phishing_prob = pipeline.predict_proba(X_val)[:, 1]
    threshold, validation_metrics = choose_threshold(y_val, val_phishing_prob)

    test_phishing_prob = pipeline.predict_proba(X_test)[:, 1]
    test_predictions = (test_phishing_prob >= threshold).astype(int)

    save_artifacts(pipeline, threshold, validation_metrics)

    print(f"Original dataset size: {len(df)}")
    print(df["label"].value_counts().sort_index().rename(index={0: "ham", 1: "phishing"}))
    print(f"Enriched dataset size: {len(enriched_df)}")
    print(
        enriched_df["label"]
        .value_counts()
        .sort_index()
        .rename(index={0: "ham", 1: "phishing"})
    )
    print(f"Balanced training dataset size: {len(balanced_df)}")
    print(
        balanced_df["label"]
        .value_counts()
        .sort_index()
        .rename(index={0: "ham", 1: "phishing"})
    )
    print(f"Selected phishing threshold: {threshold:.2f}")
    print("Validation recall summary:")
    print(validation_metrics)
    print("Confusion matrix:")
    print(confusion_matrix(y_test, test_predictions))
    print("Classification report:")
    print(classification_report(y_test, test_predictions, target_names=["Safe", "Phishing"]))


if __name__ == "__main__":
    main()
