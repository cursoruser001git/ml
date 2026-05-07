# Requirements: scikit-learn, pandas, numpy, matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load dataset (replace with your own X,y if desired)
data = load_iris()
X, y = data.data, data.target

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

# Test different K values
k_values = list(range(1, 31))  # 1..30
test_accuracies = []
cv_means = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=2, n_jobs=-1)
    knn.fit(X_train_s, y_train)
    y_pred = knn.predict(X_test_s)
    test_accuracies.append(accuracy_score(y_test, y_pred))

    cv_scores = cross_val_score(knn, np.vstack((X_train_s, X_test_s)), np.hstack((y_train, y_test)),
                                cv=5, scoring='accuracy', n_jobs=-1)
    cv_means.append(cv_scores.mean())

# Print top results
best_k_test = k_values[int(np.argmax(test_accuracies))]
best_k_cv = k_values[int(np.argmax(cv_means))]
print(f"Best k by test accuracy: {best_k_test} (accuracy={max(test_accuracies):.4f})")
print(f"Best k by 5-fold CV:    {best_k_cv} (mean accuracy={max(cv_means):.4f})\n")

print("k\tTest Acc\tCV Mean")
for k, ta, cvm in zip(k_values, test_accuracies, cv_means):
    print(f"{k}\t{ta:.4f}\t\t{cvm:.4f}")

# Optional plot
plt.figure(figsize=(8,4))
plt.plot(k_values, test_accuracies, marker='o', label='Test accuracy')
plt.plot(k_values, cv_means, marker='s', label='5-fold CV mean')
plt.xlabel('k (n_neighbors)')
plt.ylabel('Accuracy')
plt.title('KNN: Accuracy vs k (Euclidean)')
plt.legend()
plt.grid(True)
plt.show()
