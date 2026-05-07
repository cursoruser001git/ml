import numpy as np

# ==========================================
# 1. Training Data (The AND Gate)
# ==========================================
# Inputs (4 examples, 2 features each)
X = np.array([[0, 0], 
              [0, 1], 
              [1, 0], 
              [1, 1]])

# Expected Outputs (Only 1 when both inputs are 1)
y = np.array([0, 0, 0, 1])

# ==========================================
# 2. Initialization
# ==========================================
# We need 2 weights (one for each input feature) and 1 bias
weights = np.array([0.0, 0.0]) 
bias = 0.0                     

learning_rate = 0.1
epochs = 10 # We only need a few rounds for something this simple

# ==========================================
# 3. Activation Function
# ==========================================
# The Perceptron uses a simple "Step Function"
# If the number is 0 or positive, output 1. Otherwise, output 0.
def step_function(weighted_sum):
    if weighted_sum >= 0:
        return 1
    else:
        return 0

# ==========================================
# 4. The Perceptron Learning Loop
# ==========================================
for epoch in range(epochs):
    for i in range(len(X)):
        
        # --- STEP A: Calculate the Weighted Sum ---
        # Formula: (Input1 * Weight1) + (Input2 * Weight2) + Bias
        weighted_sum = np.dot(X[i], weights) + bias
        
        # --- STEP B: Make a Prediction ---
        y_pred = step_function(weighted_sum)
        
        # --- STEP C: Calculate the Error ---
        error = y[i] - y_pred
        
        # --- STEP D: Update Weights and Bias ---
        # Perceptron Learning Rule: Weight = Weight + (Learning_Rate * Error * Input)
        weights[0] += learning_rate * error * X[i][0]
        weights[1] += learning_rate * error * X[i][1]
        bias += learning_rate * error

# ==========================================
# 5. Test the Trained Perceptron
# ==========================================
print("Trained Weights:", weights)
print("Trained Bias:", bias)

print("\nPredictions after training:")
for i in range(len(X)):
    final_sum = np.dot(X[i], weights) + bias
    print(f"Input: {X[i]} -> Prediction: {step_function(final_sum)}")