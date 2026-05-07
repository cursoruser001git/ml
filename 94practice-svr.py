import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

data = {
    'StudyHours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'ExamScore':  [45, 50, 52, 65, 70, 78, 85, 90, 95, 98] 
}
df = pd.DataFrame(data)

x = df[['StudyHours']]
y = df['ExamScore']

model = SVR(kernel='linear')
model.fit(x,y)

predictions = model.predict(x)
mse = mean_squared_error(y, predictions)
r2 = r2_score(y,predictions)

print(f"Mean Squared Error: {round(mse, 2)}")
print(f"R2 is {r2}")


plt.scatter(x, y, color='red', label='Actual Data')
plt.plot(x, predictions, color='blue', label='SVR Line')
plt.title('Support Vector Regression')
plt.xlabel('Study Hours')
plt.ylabel('Exam Score')
plt.legend()
plt.show()