# Requirements: scikit-learn, pandas, numpy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Example synthetic dataset (replace with your own CSV)
data = {
    "study_hours":   [1,2,2.5,3,3.5,4,4.5,5,6,7,1.5,2.2,3.8,5.5,6.5],
    "attendance":    [60,70,65,80,85,90,75,95,88,92,55,68,82,96,89],  # percent
    "pass":          [0,0,0,1,1,1,1,1,1,1,0,0,1,1,1]                   # 0=fail,1=pass
}
df = pd.DataFrame(data)

# Features / target
X = df[["study_hours", "attendance"]]
y = df["pass"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features (recommended for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train SVM classifier (RBF kernel)
model = SVC(kernel="rbf", C=1.0, gamma="scale", probability=False, random_state=42)
model.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["fail","pass"])

print(f"Accuracy: {acc:.3f}")
print("Confusion matrix:\n", cm)
print("Classification report:\n", report)

# Example: predict single new sample
new_student = np.array([[4.0, 85.0]])           # 4 hours, 85% attendance
new_scaled = scaler.transform(new_student)
pred_label = model.predict(new_scaled)[0]
print(f"Predicted label for {new_student[0].tolist()}: {pred_label}")
