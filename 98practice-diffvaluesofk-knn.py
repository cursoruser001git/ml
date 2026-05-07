import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load and Split the Data
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Setup lists to remember the scores
k_values = range(1, 20, 2) # Tests odd numbers: 1, 3, 5, 7... up to 19
train_accuracies = []
test_accuracies = []

# 3. The Hyperparameter Tuning Loop
for k in k_values:
    # Initialize and train KNN with the current 'k'
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    
    # Check accuracy on the TRAINING data (Did it memorize the answers?)
    train_pred = knn.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    train_accuracies.append(train_acc)
    
    # Check accuracy on the TESTING data (Can it handle new questions?)
    test_pred = knn.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    test_accuracies.append(test_acc)

# 4. Print the best result
best_k = k_values[test_accuracies.index(max(test_accuracies))]
print(f"The best K value is {best_k} with an accuracy of {round(max(test_accuracies)*100, 2)}%")

# 5. Plot the "Elbow Curve" (Bias-Variance Tradeoff)
plt.plot(k_values, train_accuracies, label='Training Accuracy', marker='o')
plt.plot(k_values, test_accuracies, label='Testing Accuracy', marker='o')
plt.title('KNN: Accuracy vs. K Value')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Accuracy Score')
plt.xticks(k_values)
plt.legend()
plt.show()