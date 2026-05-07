# Requirements: scikit-learn, pandas, numpy
# Dataset: use your labeled email dataset (columns: "text", "label") or scikit-learn's fetch_20newsgroups/other spam corpora.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# 1) Load data (replace with your CSV or dataframe)
# Example: df = pd.read_csv("emails.csv")  # columns: "text", "label" where label is "spam" or "ham"
# For demonstration, here's a tiny synthetic example (replace in practice)
data = {
    "text": [
        "Congratulations! You've won a free trip. Claim now!",
        "Meeting at 10am tomorrow — agenda attached.",
        "Lowest prices on meds, buy now!",
        "Can you review the quarterly report?",
        "You have been selected for a $1000 gift card.",
        "Lunch plans? Let's meet at noon."
    ],
    "label": ["spam","ham","spam","ham","spam","ham"]
}
df = pd.DataFrame(data)

# 2) Prepare X, y
X = df["text"].values
y = df["label"].values
le = LabelEncoder()
y_enc = le.fit_transform(y)  # spam=1, ham=0 (order depends on labels)

# 3) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.25, random_state=42, stratify=y_enc
)

# 4) Pipeline: TF-IDF vectorizer + linear SVM (LinearSVC) — fast for high-dim text
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_df=0.9, min_df=2, ngram_range=(1,2))),
    ("svc", LinearSVC(C=1.0, random_state=42))
])

# 5) Fit model
pipeline.fit(X_train, y_train)

# 6) Predict and evaluate
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=le.classes_)

print(f"Accuracy: {acc:.4f}")
print("Confusion matrix:\n", cm)
print("Classification report:\n", report)

# 7) Example single prediction
examples = [
    "Limited time offer! Click to get rich fast.",
    "Please find the minutes from today's meeting attached."
]
preds = pipeline.predict(examples)
for text, p in zip(examples, preds):
    print(f"Text: {text[:60]}... => Predicted: {le.inverse_transform([p])[0]}")
