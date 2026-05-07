# Requirements: scikit-learn, numpy
import numpy as np
from sklearn.linear_model import Perceptron

# Inputs for 2-input logic
X = np.array([[0,0],[0,1],[1,0],[1,1]])

# AND
y_and = np.array([0,0,0,1])
clf_and = Perceptron(max_iter=1000, tol=1e-3, random_state=42)
clf_and.fit(X, y_and)
pred_and = clf_and.predict(X)

# OR
y_or = np.array([0,1,1,1])
clf_or = Perceptron(max_iter=1000, tol=1e-3, random_state=42)
clf_or.fit(X, y_or)
pred_or = clf_or.predict(X)

print("AND predictions:", pred_and)
print("OR  predictions:", pred_or)
print("AND weights (bias, w1, w2):", np.hstack((clf_and.intercept_, clf_and.coef_.ravel())))
print("OR  weights (bias, w1, w2):", np.hstack((clf_or.intercept_, clf_or.coef_.ravel())))
