import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = {
    'StudyHours': [2, 3, 5, 2, 1, 4, 7, 9,12,1,0,7,3],
    'Attendance': [20, 60, 80, 95, 10, 75, 85, 98,95,10,5,70,50],
    'Pass':       [0,  0,  1,  1,  0,  0,  1,  1,1,0,0,1,0] 
}
df = pd.DataFrame(data)

x = df[['StudyHours','Attendance']]
y = df[['Pass']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

model = LogisticRegression()
model.fit(x_train,y_train)

prediction = model.predict(x_test)

accuracy = accuracy_score(y_test,prediction)

print(accuracy)