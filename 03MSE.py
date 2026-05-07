# Requirements: scikit-learn, pandas, numpy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

# 1) Create sample housing dataset (8 features similar to California housing)
X = pd.DataFrame({
    "MedInc": [8.3, 8.3, 7.3, 6.4, 5.5, 4.5, 3.5, 2.5, 6.0, 7.0, 5.0, 4.0, 8.0, 7.5, 6.5],
    "HouseAge": [41, 21, 52, 23, 8, 9, 28, 12, 30, 45, 18, 25, 35, 40, 22],
    "AveRooms": [6.98, 6.23, 8.26, 8.16, 5.23, 5.88, 6.94, 6.51, 6.77, 7.23, 5.54, 6.89, 7.10, 6.45, 5.87],
    "AveBedrms": [1.02, 0.97, 1.24, 1.23, 1.07, 1.04, 0.96, 1.15, 1.05, 1.10, 1.08, 0.99, 1.02, 1.05, 0.98],
    "Population": [494, 558, 565, 413, 1094, 1157, 480, 271, 707, 521, 1053, 668, 580, 420, 900],
    "AveOccup": [2.80, 2.55, 2.82, 2.60, 2.97, 3.10, 2.90, 2.40, 2.76, 2.88, 2.65, 3.05, 2.85, 2.70, 2.95],
    "Latitude": [37.88, 37.86, 37.85, 37.85, 37.84, 37.83, 37.82, 37.81, 37.80, 37.79, 37.78, 37.77, 37.76, 37.75, 37.74],
    "Longitude": [-122.23, -122.25, -122.24, -122.26, -122.25, -122.27, -122.26, -122.28, -122.27, -122.29, -122.28, -122.30, -122.29, -122.31, -122.30]
})
y = pd.Series([4.526, 3.585, 3.521, 3.413, 3.422, 2.697, 2.992, 2.414, 2.267, 2.611, 2.037, 1.867, 3.206, 2.801, 2.373])

# 2) Use a single feature (MedInc) for a simple linear regression example.
#    If you want multivariate, set X_single = X (or subset of columns).
X_single = X[["MedInc"]]

# 3) Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_single, y, test_size=0.2, random_state=42)

# 4) Fit linear regression
model = LinearRegression()
model.fit(X_train, y_train)

# 5) Predict and evaluate using MSE
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# 6) Output results
print(f"Intercept: {model.intercept_:.4f}")
print(f"Coefficient (MedInc): {model.coef_[0]:.4f}")
print(f"MSE on test set: {mse:.6f}")

# Optional: example prediction
example = np.array([[8.0]])                 # median income = 8 (typical scale in dataset)
pred = model.predict(example)[0]
print(f"Predicted median house value (100k$) for MedInc=8.0: {pred:.3f}")
