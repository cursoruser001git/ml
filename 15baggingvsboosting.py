# Requirements: scikit-learn, pandas, numpy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# 1) Create sample wine dataset
cols = ["Class", "Alcohol","Malic_acid","Ash","Alcalinity_of_ash","Magnesium",
        "Total_phenols","Flavanoids","Nonflavanoid_phenols","Proanthocyanins",
        "Color_intensity","Hue","OD280/OD315","Proline"]

# Sample wine data (3 classes, 13 features)
data_sample = {
    "Class": [1,1,1,1,1,2,2,2,2,2,3,3,3,3,3],
    "Alcohol": [14.2,13.2,13.2,14.4,13.1,12.2,13.5,12.8,13.7,12.0,13.4,12.1,13.8,12.3,13.6],
    "Malic_acid": [3.8,3.6,4.1,4.4,2.2,3.2,3.7,3.8,2.5,4.1,4.7,4.1,5.0,2.3,3.5],
    "Ash": [2.3,2.7,2.5,2.9,2.3,2.4,2.6,2.7,2.8,2.8,2.8,2.7,3.2,2.8,2.6],
    "Alcalinity_of_ash": [15,21,19,25,20,20,20,19,25,22,28,30,23,23,17],
    "Magnesium": [112,65,97,118,120,92,99,86,125,99,112,110,97,97,106],
    "Total_phenols": [3.0,2.3,3.2,3.0,1.6,2.4,2.6,2.8,2.7,2.5,3.0,2.5,3.2,2.8,2.86],
    "Flavanoids": [3.2,1.8,2.8,2.8,2.3,2.1,2.3,2.3,1.9,1.8,3.0,1.7,3.0,2.7,2.68],
    "Nonflavanoid_phenols": [0.26,0.42,0.46,0.39,0.26,0.41,0.32,0.46,0.60,0.66,0.29,0.63,0.27,0.39,0.3],
    "Proanthocyanins": [3.5,1.9,3.6,1.6,1.6,4.4,1.8,3.2,1.6,2.3,2.6,1.1,2.3,2.8,2.7],
    "Color_intensity": [8.5,4.1,10.6,7.2,5.6,5.0,7.5,10.0,5.3,6.2,9.6,9.4,9.7,8.0,8.6],
    "Hue": [0.85,0.59,0.84,0.86,0.58,0.42,0.56,0.72,0.46,0.32,0.41,0.68,0.43,0.63,0.84],
    "OD280/OD315": [3.49,1.62,4.70,1.86,1.52,3.27,2.96,3.6,3.52,1.6,3.88,1.73,4.26,3.3,5.75],
    "Proline": [1480,480,1450,570,640,1235,1271,1790,649,867,1080,615,1285,830,720]
}
df = pd.DataFrame(data_sample)

X = df.drop(columns=["Class"])
y = df["Class"]

# 2) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Optional scaling for some models
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 3) Define models
base_dt = DecisionTreeClassifier(max_depth=None, random_state=42)
bag = BaggingClassifier(estimator=DecisionTreeClassifier(random_state=42),
                       n_estimators=50, random_state=42, n_jobs=-1)
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)  # bagging-like
adb = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
                         n_estimators=100, learning_rate=1.0, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

models = {"DecisionTree": base_dt, "Bagging": bag, "RandomForest": rf, "AdaBoost": adb, "GradientBoosting": gb}

# 4) Train, predict, and evaluate
results = {}
for name, model in models.items():
    # use scaled features for tree models not necessary but consistent
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    results[name] = {"accuracy": acc, "confusion_matrix": cm, "report": report}
    print(f"--- {name} ---")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion matrix:\n", cm)
    print("Classification report:\n", classification_report(y_test, y_pred))
    print()

# 5) Cross-validated comparison (mean accuracy)
from sklearn.model_selection import cross_val_score
print("Cross-validated accuracy (5-fold):")
for name, model in models.items():
    scores = cross_val_score(model, scaler.transform(X), y, cv=5, scoring="accuracy", n_jobs=-1)
    print(f"{name}: mean={scores.mean():.4f}, std={scores.std():.4f}")

# 6) Summary: results dict contains accuracy, confusion matrices, and full reports for each model.
