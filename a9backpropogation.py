import numpy as np

# ==========================================
# 1. Activation Function & Its Derivative
# ==========================================
# Sigmoid squashes numbers between 0 and 1
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# The derivative is needed for backpropagation (calculating gradients)
def sigmoid_derivative(x):
    return x * (1 - x)

# ==========================================
# 2. Training Data (The XOR Problem)
# ==========================================
# Inputs (4 examples, 2 features each)
X = np.array([[0,0], 
              [0,1], 
              [1,0], 
              [1,1]])

# Expected Outputs
y = np.array([[0], 
              [1], 
              [1], 
              [0]])

# ==========================================
# 3. Initialize Weights Randomly
# ==========================================
np.random.seed(1) # Keeps random numbers consistent for testing

# Weights connecting Input Layer (2 neurons) to Hidden Layer (2 neurons)
W1 = np.random.uniform(size=(2, 2))

# Weights connecting Hidden Layer (2 neurons) to Output Layer (1 neuron)
W2 = np.random.uniform(size=(2, 1))

learning_rate = 0.5
epochs = 10000

# ==========================================
# 4. The Backpropagation Training Loop
# ==========================================
for epoch in range(epochs):
    
    # --- STEP A: FORWARD PASS (Making a guess) ---
    # Multiply inputs by weights and apply sigmoid
    hidden_layer = sigmoid(np.dot(X, W1))
    output_layer = sigmoid(np.dot(hidden_layer, W2))
    
    # --- STEP B: CALCULATE ERROR (How wrong was the guess?) ---
    error = y - output_layer
    
    # --- STEP C: BACKWARD PASS (Backpropagation Math) ---
    # 1. How much should we change the output weights?
    # Formula: Error * Derivative of Output
    d_output = error * sigmoid_derivative(output_layer)
    
    # 2. How much did the hidden layer contribute to the error?
    error_hidden = d_output.dot(W2.T)
    
    # 3. How much should we change the hidden weights?
    d_hidden = error_hidden * sigmoid_derivative(hidden_layer)
    
    # --- STEP D: UPDATE WEIGHTS (Learning) ---
    # Adjust weights based on the calculations and learning rate
    W2 += hidden_layer.T.dot(d_output) * learning_rate
    W1 += X.T.dot(d_hidden) * learning_rate

# Print the final guesses after 10,000 rounds of learning
print("Final Predictions after Training:")
print(output_layer)