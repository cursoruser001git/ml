import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

data = {
    'A':[1,2,3,4,5,6,7,8,9,10,11,12,13,14],
    'B':[54,56,57,58,60,62,64,66,68,70,72,74,76,78]
}

df = pd.DataFrame(data)
x = df[['A']]
y = df[['B']] 

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.4,random_state=0)

model = LinearRegression()
model.fit(x_train,y_train)

predictions = model.predict(x_test)

mse = mean_squared_error(y_test,predictions)

plt.scatter(x_test,y_test,color = 'blue')

plt.plot(x_test,predictions,color ='red')

plt.show()