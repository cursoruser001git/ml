import pandas as pd 
from sklearn.neighbors import KNeighborsClassifier

data = {
    'Xcord':[1,2,1.5, 8,9,8.5],
    'Ycord':[1,1.5,2, 8,8.5,9],
    'Class':[0,0,0, 1,1,1]
}

df = pd.DataFrame(data)

x = df[['Xcord','Ycord']]
y = df['Class']

knn = KNeighborsClassifier(n_neighbors=3,metric='euclidean') # we also have manhattan and minkowski

knn.fit(x,y)

new_pt = [[7,7]]

pred = knn.predict(new_pt)

print(pred[0]) # prints class 1 or 2 