import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_wine

wine_data = load_wine()
x = pd.DataFrame(wine_data.data,columns=wine_data.feature_names)
y = wine_data.target

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

stump = DecisionTreeClassifier(max_depth=1,random_state=42)

model = AdaBoostClassifier(estimator=stump,n_estimators=50,random_state=42)

model.fit(x_train,y_train)


print(accuracy_score(y_test,model.predict(x_test)))