import pickle
import re

# Load model and vectorizer at startup
model = pickle.load(open("ml/phishing_model.pkl", "rb"))
vectorizer = pickle.load(open("ml/vectorizer.pkl", "rb"))

def clean_text(text):
    text = re.sub(r"http\S+", "link", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text.lower()

def predict_email(text):
    clean = clean_text(text)
    vector = vectorizer.transform([clean])
    prediction = model.predict(vector)[0]
    return "Phishing ⚠️" if prediction == 1 else "Legitimate ✅"
