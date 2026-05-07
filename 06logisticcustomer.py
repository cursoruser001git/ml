# Requirements: pandas, scikit-learn, numpy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# 1) Example dataset (replace with your data)
data = {
    "age":    [22,25,30,35,40,45,50,23,31,38,27,48,52,29,60,33,41,55,20,46],
    "income": [30,35,45,50,60,80,90,28,48,55,36,85,95,42,100,52,63,88,22,70],  # thousands per year
    "buy":    [0,0,1,1,1,1,1,0,1,1,0,1,1,0,1,1,1,1,0,1]  # 1 = purchased
}
df = pd.DataFrame(data)

# 2) Features and target
X = df[["age", "income"]]
y = df["buy"]

# 3) Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# 4) Scale features (recommended)
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
roc_auc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["no_buy","buy"])

print(f"Accuracy: {acc:.3f}")
print(f"ROC AUC: {roc_auc:.3f}")
print("Confusion matrix:\n", cm)
print("Classification report:\n", report)

# 7) Coefficients (on scaled features) and odds ratios
coef = model.coef_[0]
intercept = model.intercept_[0]
odds_ratios = np.exp(coef)
print(f"Intercept: {intercept:.3f}")
print(f"Coefficients (age, income): {coef[0]:.3f}, {coef[1]:.3f}")
print(f"Odds ratios: age -> {odds_ratios[0]:.3f}, income -> {odds_ratios[1]:.3f}")

# 8) Example prediction for a new customer
new_customer = np.array([[37, 58]])   # age 37, income 58k
new_scaled = scaler.transform(new_customer)
prob_buy = model.predict_proba(new_scaled)[0,1]
label = model.predict(new_scaled)[0]
print(f"Predicted probability of buying: {prob_buy:.3f}, predicted label: {label}")
