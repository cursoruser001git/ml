# Requirements: scikit-learn, pandas, numpy
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Load example dataset
data = load_iris()
X = data.data
y = data.target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Optional scaling
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Train KNN (use Euclidean distance by default, p=2)
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2, n_jobs=-1)
knn.fit(X_train_s, y_train)

# Predict and evaluate
y_pred = knn.predict(X_test_s)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))

# Optional: simple grid search for k
param_grid = {'n_neighbors': [1,3,5,7,9]}
gs = GridSearchCV(KNeighborsClassifier(metric='minkowski', p=2), param_grid, cv=5, n_jobs=-1)
gs.fit(np.vstack((X_train_s, X_test_s)), np.hstack((y_train, y_test)))
print("Best k (CV):", gs.best_params_, "Best score:", gs.best_score_)
