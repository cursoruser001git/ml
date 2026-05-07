# Requirements: scikit-learn, pandas, numpy
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# 1) Load a multifeature dataset (binary classification)
data = load_breast_cancer(as_frame=True)
X = data.data        # many numeric features
y = data.target      # 0/1 labels

# 2) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 3) Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4) Fit logistic regression (use liblinear or saga for small datasets)
model = LogisticRegression(solver="liblinear", max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# 5) Predict and evaluate
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=data.target_names)

print(f"Accuracy: {acc:.4f}")
print("Confusion matrix:\n", cm)
print("Classification report:\n", report)

# 6) Optional: inspect coefficients (feature importance direction)
coef = pd.Series(model.coef_[0], index=X.columns).sort_values(key=abs, ascending=False)
print("Top coefficients (by absolute value):\n", coef.head(10))
