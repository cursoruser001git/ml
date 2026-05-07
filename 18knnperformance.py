# Requirements: scikit-learn, numpy, pandas, matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset (replace with your own X,y if desired)
data = load_iris()
X, y = data.data, data.target

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

# Distance metrics to compare (scikit-learn names)
metrics = {
    "Euclidean (p=2)": ("minkowski", 2),
    "Manhattan (p=1)": ("minkowski", 1),
    "Chebyshev": ("chebyshev", None),
    "Minkowski (p=3)": ("minkowski", 3),
    "Cosine": ("cosine", None)
}

results = {}
for name, (metric, p) in metrics.items():
    if metric == "minkowski":
        knn = KNeighborsClassifier(n_neighbors=5, metric=metric, p=p, n_jobs=-1)
    else:
        knn = KNeighborsClassifier(n_neighbors=5, metric=metric, n_jobs=-1)
    knn.fit(X_train_s, y_train)
    y_pred = knn.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    cv = cross_val_score(knn, np.vstack((X_train_s, X_test_s)), np.hstack((y_train, y_test)),
                         cv=5, scoring="accuracy", n_jobs=-1)
    results[name] = {"accuracy": acc, "cv_mean": cv.mean(), "cv_std": cv.std(),
                     "confusion_matrix": confusion_matrix(y_test, y_pred),
                     "report": classification_report(y_test, y_pred, output_dict=True)}
    print(f"--- {name} ---")
    print(f"Test accuracy: {acc:.4f}  |  CV mean (5-fold): {cv.mean():.4f} ± {cv.std():.4f}")
    print("Confusion matrix:\n", results[name]["confusion_matrix"])
    print("Classification report:\n", classification_report(y_test, y_pred))
    print()

# Summary table (simple)
summary = pd.DataFrame([
    {"metric": k, "test_acc": v["accuracy"], "cv_mean": v["cv_mean"], "cv_std": v["cv_std"]}
    for k, v in results.items()
]).sort_values(by="test_acc", ascending=False)
print("Summary:")
print(summary.to_string(index=False))

# Optional plot of test accuracies
plt.figure(figsize=(8,4))
plt.bar(summary["metric"], summary["test_acc"], yerr=summary["cv_std"], capsize=5)
plt.ylabel("Test accuracy")
plt.title("KNN performance across distance metrics (k=5)")
plt.xticks(rotation=30, ha="right")
plt.ylim(0,1)
plt.tight_layout()
plt.show()
