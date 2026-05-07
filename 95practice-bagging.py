import pandas as pd
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

wine_data = load_wine()

x = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
y = wine_data.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

single_tree = DecisionTreeClassifier(random_state=0)

model  = BaggingClassifier(estimator=single_tree,n_estimators=10,random_state=2)

model.fit(x_train,y_train)

predictions = model.predict(x_test)

accuracy = accuracy_score(y_test,predictions)

print(accuracy)