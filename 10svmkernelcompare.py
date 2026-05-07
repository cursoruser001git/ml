# Requirements: scikit-learn, pandas, numpy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- Load data (replace with real dataset) ---
# Example: df = pd.read_csv("emails.csv")  # columns: "text", "label"
data = {
    "text": [
        "Congratulations! You've won a free trip. Claim now!",
        "Meeting at 10am tomorrow — agenda attached.",
        "Lowest prices on meds, buy now!",
        "Can you review the quarterly report?",
        "You have been selected for a $1000 gift card.",
        "Lunch plans? Let's meet at noon.",
        "Earn cash fast, no experience needed!",
        "Project deadline moved to Friday.",
        "Exclusive offer: 50% off for subscribers.",
        "Please approve the invoice."
    ],
    "label": ["spam","ham","spam","ham","spam","ham","spam","ham","spam","ham"]
}
df = pd.DataFrame(data)

X = df["text"].values
y = df["label"].values

# Encode labels to 0/1
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_enc = le.fit_transform(y)  # e.g., ham=0, spam=1

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.3, random_state=42, stratify=y_enc
)

# Common TF-IDF transform
tfidf = TfidfVectorizer(max_df=0.9, min_df=1, ngram_range=(1,2))

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

results = []

# 1) Linear SVM via LinearSVC (fast, no probabilities)
linear_clf = LinearSVC(C=1.0, random_state=42, max_iter=5000)
linear_clf.fit(X_train_tfidf, y_train)
y_pred_lin = linear_clf.predict(X_test_tfidf)
results.append(("LinearSVC", accuracy_score(y_test, y_pred_lin), confusion_matrix(y_test, y_pred_lin),
                classification_report(y_test, y_pred_lin, target_names=le.classes_)))

# 2) SVM with linear kernel using SVC (can provide probability with probability=True but slower)
svc_linear = SVC(kernel="linear", C=1.0, probability=True, random_state=42)
svc_linear.fit(X_train_tfidf, y_train)
y_pred_svc_lin = svc_linear.predict(X_test_tfidf)
results.append(("SVC_linear", accuracy_score(y_test, y_pred_svc_lin), confusion_matrix(y_test, y_pred_svc_lin),
                classification_report(y_test, y_pred_svc_lin, target_names=le.classes_)))

# 3) SVM with RBF kernel
svc_rbf = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)
svc_rbf.fit(X_train_tfidf, y_train)
y_pred_rbf = svc_rbf.predict(X_test_tfidf)
results.append(("SVC_rbf", accuracy_score(y_test, y_pred_rbf), confusion_matrix(y_test, y_pred_rbf),
                classification_report(y_test, y_pred_rbf, target_names=le.classes_)))

# 4) Optional: hyperparameter tuning (grid search) for RBF (uncomment to run)
# param_grid = {"C":[0.1,1,10], "gamma":["scale","auto",0.01,0.1,1]}
# grid = GridSearchCV(SVC(kernel="rbf", probability=False, random_state=42), param_grid, cv=3, n_jobs=-1)
# grid.fit(X_train_tfidf, y_train)
# best = grid.best_estimator_
# y_pred_grid = best.predict(X_test_tfidf)
# results.append(("SVC_rbf_tuned", accuracy_score(y_test, y_pred_grid), confusion_matrix(y_test, y_pred_grid),
#                 classification_report(y_test, y_pred_grid, target_names=le.classes_)))

# --- Print comparison ---
for name, acc, cm, report in results:
    print(f"Model: {name}")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion matrix:")
    print(cm)
    print("Classification report:")
    print(report)
    print("-" * 50)

# --- Example: predict new samples with each model ---
examples = [
    "Limited time offer! Click to get rich fast.",
    "Please see attached agenda for tomorrow's meeting."
]
ex_tfidf = tfidf.transform(examples)
for name, model in [("LinearSVC", linear_clf), ("SVC_linear", svc_linear), ("SVC_rbf", svc_rbf)]:
    preds = model.predict(ex_tfidf)
    labels = le.inverse_transform(preds)
    print(f"{name} predictions: {labels.tolist()}")
