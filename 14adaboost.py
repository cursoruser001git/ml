# Requirements: scikit-learn, pandas, numpy
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1) Load dataset (replace with your CSV if needed)
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target  # 0/1

# 2) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 3) Base estimator (decision stump)
base = DecisionTreeClassifier(max_depth=1, random_state=42)

# 4) AdaBoost classifier
adb = AdaBoostClassifier(
    estimator=base,
    n_estimators=100,
    learning_rate=1.0,
    random_state=42
)
adb.fit(X_train, y_train)

# 5) Predict and evaluate
y_pred = adb.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=data.target_names)

print(f"Accuracy: {acc:.4f}")
print("Confusion matrix:\n", cm)
print("Classification report:\n", report)

# 6) Optional: feature importances
importances = pd.Series(adb.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Top feature importances:\n", importances.head(10))
