import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = {
    'StudyHours': [
        # Normal Fails (Low/Low)
        1, 2, 0, 3, 2, 1, 
        # Normal Passes (The Sweet Spot)
        5, 6, 7, 8, 6, 9, 7, 8, 5, 
        # "Present but Lazy" Fails (Low Study, High Attend)
        0, 1, 2, 1, 0, 
        # "Absent Genius" Fails (High Study, Low Attend)
        10, 11, 12, 11, 10
    ],
    'Attendance': [
        # Normal Fails
        10, 20, 5, 40, 15, 30, 
        # Normal Passes
        75, 80, 85, 90, 88, 95, 70, 82, 78, 
        # Present but Lazy Fails
        85, 95, 90, 80, 100, 
        # Absent Genius Fails
        20, 15, 30, 10, 25
    ],
    'Pass': [
        0, 0, 0, 0, 0, 0,        # Normal Fails
        1, 1, 1, 1, 1, 1, 1, 1, 1, # Passes
        0, 0, 0, 0, 0,           # Lazy Fails
        0, 0, 0, 0, 0            # Absent Fails
    ]
}

df = pd.DataFrame(data)

x = df[['StudyHours','Attendance']]
y = df[['Pass']]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=99)

#linear
model_linear = SVC(kernel='linear')
model_linear.fit(x_train,y_train)

preductions_linear = model_linear.predict(x_test)

accuracy_linear = accuracy_score(y_test,preductions_linear)

print(accuracy_linear)

#RBF

model_RBF = SVC(kernel='rbf')
model_RBF.fit(x_train,y_train)

preductions_rbf = model_RBF.predict(x_test)
accuracy_rbf = accuracy_score(y_test,preductions_rbf)

print(accuracy_rbf)