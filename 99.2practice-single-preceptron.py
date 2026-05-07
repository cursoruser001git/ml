import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

# 1. Create the Truth Tables (The Dataset)
# Features: [Input_1, Input_2]
X = [
    [0, 0], 
    [0, 1], 
    [1, 0], 
    [1, 1]
]

# Targets (The exact answers for an AND gate and an OR gate)
y_AND = [0, 0, 0, 1] # Only True if BOTH are 1
y_OR  = [0, 1, 1, 1] # True if AT LEAST ONE is 1

# 2. Initialize the Single Perceptron
perceptron_and = Perceptron(max_iter=1000, random_state=42)
perceptron_or  = Perceptron(max_iter=1000, random_state=42)

# 3. Train the Models
perceptron_and.fit(X, y_AND)
perceptron_or.fit(X, y_OR)

# 4. Make Predictions
predictions_and = perceptron_and.predict(X)
predictions_or  = perceptron_or.predict(X)

print(f"AND Gate Actual:     {y_AND}")
print(f"AND Gate Predicted:  {predictions_and.tolist()}")
print("---")
print(f"OR Gate Actual:      {y_OR}")
print(f"OR Gate Predicted:   {predictions_or.tolist()}")