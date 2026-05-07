# Requirements: scikit-learn, numpy, matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1) Synthetic example dataset (single input feature)
rng = np.random.RandomState(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)               # feature (e.g., hours)
y = 3.0 * np.sin(X).ravel() + 0.5 * X.ravel() + rng.normal(0, 0.5, X.shape[0])
# y is continuous target (e.g., price, score)

# 2) Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3) Fit SVR models with different kernels for comparison
svr_rbf = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
svr_linear = SVR(kernel="linear", C=100, epsilon=0.1)
svr_poly = SVR(kernel="poly", C=100, degree=3, epsilon=0.1, coef0=1)

svr_rbf.fit(X_train, y_train)
svr_linear.fit(X_train, y_train)
svr_poly.fit(X_train, y_train)

# 4) Predict on test set and compute metrics
models = {"RBF": svr_rbf, "Linear": svr_linear, "Poly": svr_poly}
for name, model in models.items():
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name} SVR — MSE: {mse:.4f}, R^2: {r2:.4f}")

# 5) Analyze predictions: visualize fit on whole range
X_plot = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
plt.scatter(X, y, color="gray", alpha=0.5, label="Data")
for name, model in models.items():
    y_plot = model.predict(X_plot)
    plt.plot(X_plot, y_plot, label=f"{name} SVR")
plt.xlabel("X (feature)")
plt.ylabel("y (target)")
plt.legend()
plt.title("SVR fits (RBF, Linear, Poly)")
plt.show()

# 6) Example: inspect residuals for RBF model
y_test_pred = svr_rbf.predict(X_test)
residuals = y_test - y_test_pred
print("RBF residuals (first 10):", residuals[:10])
print("RBF predicted vs actual (first 10):")
for actual, pred in list(zip(y_test, y_test_pred))[:10]:
    print(f"actual: {actual:.3f}, pred: {pred:.3f}, error: {actual - pred:.3f}")
