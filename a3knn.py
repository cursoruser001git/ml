from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

X, y = make_classification(n_samples=200, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 1. K-Nearest Neighbors (KNN)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# 2. Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)

# 3. Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

print("KNN Score:", knn.score(X_test, y_test))
print("Naive Bayes Score:", nb.score(X_test, y_test))
print("Decision Tree Score:", dt.score(X_test, y_test))