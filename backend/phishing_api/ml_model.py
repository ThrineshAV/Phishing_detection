import os
import pickle

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_PATH = os.path.join(BASE_DIR, "ml", "phishing_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "ml", "vectorizer.pkl")

# Load vectorizer
with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

# Load trained model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


def predict_email(email_text):

    email_vec = vectorizer.transform([email_text])

    prediction = model.predict(email_vec)[0]

    if prediction == 1:
        return "Phishing Email"
    else:
        return "Safe Email"