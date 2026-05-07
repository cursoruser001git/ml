# Requirements: pandas, scikit-learn, numpy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# 1) Example dataset (replace with your data)
data = {
    "study_hours":   [1,2,2.5,3,3.5,4,4.5,5,6,7,1.5,2.2,3.8,5.5,6.5],
    "attendance":    [60,70,65,80,85,90,75,95,88,92,55,68,82,96,89],  # percent
    # Pass if combined score high enough (this is synthetic label)
    "pass":          [0,0,0,1,1,1,1,1,1,1,0,0,1,1,1]
}
df = pd.DataFrame(data)

# 2) Features and target
X = df[["study_hours", "attendance"]]
y = df["pass"]  # 0 = fail, 1 = pass

# 3) Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4) Optional: scale features (recommended for logistic regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5) Fit logistic regression
model = LogisticRegression(solver="liblinear", random_state=42)
model.fit(X_train_scaled, y_train)

# 6) Predict and evaluate
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:,1]

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["fail","pass"])

print(f"Accuracy: {acc:.3f}")
print("Confusion matrix:\n", cm)
print("Classification report:\n", report)

# 7) Example: predict probability for a new student
new_student = np.array([[4.0, 85.0]])          # 4 hours study, 85% attendance
new_scaled = scaler.transform(new_student)
prob_pass = model.predict_proba(new_scaled)[0,1]
pred_label = model.predict(new_scaled)[0]
print(f"Predicted probability of pass: {prob_pass:.3f}, label: {pred_label}")

# 8) Inspect coefficients (on scaled features)
coef = model.coef_[0]
intercept = model.intercept_[0]
print(f"Intercept: {intercept:.3f}")
print(f"Coefficients (study_hours, attendance): {coef[0]:.3f}, {coef[1]:.3f}")
