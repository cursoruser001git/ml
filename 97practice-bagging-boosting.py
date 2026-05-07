import pandas as pd 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.datasets import load_wine

data = load_wine()
x = pd.DataFrame(data.data, columns= data.feature_names)
y = pd.DataFrame(data.target)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state=1)

tree = DecisionTreeClassifier(random_state=42)
bagging_model = BaggingClassifier(estimator=tree,n_estimators=50,random_state=42)

stump = DecisionTreeClassifier(max_depth=1,random_state=42)
boosting_model = AdaBoostClassifier(estimator=stump,n_estimators=50,random_state=42)

bagging_model.fit(x_train,y_train)
boosting_model.fit(x_train,y_train)

print(accuracy_score(y_test,bagging_model.predict(x_test)))
print(accuracy_score(y_test,boosting_model.predict(x_test)))

