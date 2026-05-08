from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# 1. Dataset
X, y = make_classification(n_samples=500, random_state=1)

# 2. Bagging (Random Forest is the ultimate bagging algorithm)
bagging_model = RandomForestClassifier(n_estimators=50)
bagging_model.fit(X, y)
print("Bagging Accuracy:", bagging_model.score(X, y))

# 3. Boosting (AdaBoost)
boosting_model = AdaBoostClassifier(n_estimators=50)
boosting_model.fit(X, y)
print("Boosting Accuracy:", boosting_model.score(X, y))

# 4. Stacking (Combine Tree and Forest, judged by Logistic Regression)
estimators = [
    ('dt', DecisionTreeClassifier()),
    ('rf', RandomForestClassifier(n_estimators=10))
]
stacking_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
stacking_model.fit(X, y)
print("Stacking Accuracy:", stacking_model.score(X, y))
