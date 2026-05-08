from sklearn.datasets import make_blobs, make_circles
from sklearn.svm import SVC

# --- LINEAR SVM ---
# Dataset: 2 clear blobs that can be separated by a straight line
X_lin, y_lin = make_blobs(n_samples=100, centers=2, random_state=0)

linear_svm = SVC(kernel='linear') # Linear kernel draws a straight line
linear_svm.fit(X_lin, y_lin)
print("Linear SVM Accuracy:", linear_svm.score(X_lin, y_lin))

# --- NON-LINEAR SVM ---
# Dataset: A circle inside a circle (cannot be cut by a straight line)
X_nonlin, y_nonlin = make_circles(n_samples=100, factor=0.5, noise=0.05)

rbf_svm = SVC(kernel='rbf') # RBF kernel handles non-linear boundaries
rbf_svm.fit(X_nonlin, y_nonlin)
print("Non-Linear (RBF) SVM Accuracy:", rbf_svm.score(X_nonlin, y_nonlin))
