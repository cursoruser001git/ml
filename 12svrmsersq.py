# Requirements: scikit-learn, numpy, matplotlib (optional)
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# --- Example synthetic dataset (single feature) ---
rng = np.random.RandomState(0)
X = np.linspace(0, 10, 200).reshape(-1, 1)
y = 2.5 * np.sin(1.5 * X).ravel() + 0.3 * X.ravel() + rng.normal(0, 0.5, X.shape[0])

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Fit SVR (RBF) with basic hyperparams ---
svr = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
svr.fit(X_train, y_train)

# --- Predict and evaluate ---
y_pred = svr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse:.4f}")
print(f"R^2: {r2:.4f}")

# --- Optional: hyperparameter tuning (uncomment to run) ---
# param_grid = {"C":[1,10,100], "gamma":[0.01,0.1,1], "epsilon":[0.01,0.1,0.2]}
# gs = GridSearchCV(SVR(kernel="rbf"), param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
# gs.fit(X_train, y_train)
# best = gs.best_estimator_
# y_pred_best = best.predict(X_test)
# print("Tuned model MSE:", mean_squared_error(y_test, y_pred_best))
# print("Tuned model R^2:", r2_score(y_test, y_pred_best))
# print("Best params:", gs.best_params_)

# --- Optional: visualize predictions vs actual ---
X_plot = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
y_plot = svr.predict(X_plot)
plt.scatter(X_test, y_test, color="black", label="test data")
plt.plot(X_plot, y_plot, color="red", label="SVR fit")
plt.legend()
plt.xlabel("Feature")
plt.ylabel("Target")
plt.title(f"SVR (MSE={mse:.4f}, R^2={r2:.4f})")
plt.show()
