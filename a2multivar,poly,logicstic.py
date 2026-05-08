from sklearn.datasets import make_regression, make_classification
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

# --- A. MULTIVARIABLE REGRESSION ---
# 10 features (variables) predicting 1 output target
X_reg, y_reg = make_regression(n_samples=100, n_features=10, noise=0.1)
multi_reg = LinearRegression()
multi_reg.fit(X_reg, y_reg)
print("Multivariable R^2 Score:", multi_reg.score(X_reg, y_reg))

# --- B. POLYNOMIAL REGRESSION ---
# We take the existing features and square them (degree=2) to allow curves
poly_converter = PolynomialFeatures(degree=2)
X_poly = poly_converter.fit_transform(X_reg)

poly_reg = LinearRegression()
poly_reg.fit(X_poly, y_reg)
print("Polynomial R^2 Score:", poly_reg.score(X_poly, y_reg))

# --- C. LOGISTIC REGRESSION ---
# Notice we switch to make_classification, because Logistic is for categories!
X_clf, y_clf = make_classification(n_samples=100, n_classes=2)
log_reg = LogisticRegression()
log_reg.fit(X_clf, y_clf)
print("Logistic Regression Accuracy:", log_reg.score(X_clf, y_clf))
