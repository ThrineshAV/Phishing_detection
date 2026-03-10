import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
df1 = pd.read_csv("data/phishing_emails.csv", usecols=["Email Text"])
df2 = pd.read_csv("data/Ham_email.csv")

# Rename column
df1.rename(columns={"Email Text": "email_text"}, inplace=True)

df1["label"] = 1
df2["label"] = 0

# Combine datasets
df = pd.concat([df1, df2]).sample(frac=1).reset_index(drop=True)

# 🔹 Remove empty email text rows
df = df.dropna(subset=["email_text"])

X = df["email_text"]
y = df["label"]

# Vectorizer
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)

X_vec = vectorizer.fit_transform(X)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_vec, y)

# Save vectorizer
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Save model
with open("phishing_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model + Vectorizer saved successfully!")