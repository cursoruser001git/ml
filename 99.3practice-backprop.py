import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# 1. Create the Truth Tables (The Dataset)
X = [
    [0, 0], 
    [0, 1], 
    [1, 0], 
    [1, 1]
]

# XOR: Output is 1 ONLY if the inputs are different
y_XOR = [0, 1, 1, 0] 

# XNOR: Output is 1 ONLY if the inputs are the same
y_XNOR = [1, 0, 0, 1]

# 2. Initialize the Multi-Layer Perceptron (Neural Network)
# hidden_layer_sizes=(4,): We are putting 4 neurons in the hidden layer
# max_iter=2000: We give it 2000 tries to learn the math (Backpropagation takes time!)
# solver='lbfgs' or 'adam': These are the optimizers that actually perform the backpropagation
model_xor = MLPClassifier(hidden_layer_sizes=(4,), activation='relu', max_iter=2000, random_state=42)
model_xnor = MLPClassifier(hidden_layer_sizes=(4,), activation='relu', max_iter=2000, random_state=42)

# 3. Train the Models (This is where Backpropagation happens!)
model_xor.fit(X, y_XOR)
model_xnor.fit(X, y_XNOR)

# 4. Make Predictions
predictions_xor = model_xor.predict(X)
predictions_xnor = model_xnor.predict(X)

print(f"XOR Actual:      {y_XOR}")
print(f"XOR Predicted:   {predictions_xor.tolist()}")
print("---")
print(f"XNOR Actual:     {y_XNOR}")
print(f"XNOR Predicted:  {predictions_xnor.tolist()}")