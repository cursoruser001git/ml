# Requirements: tensorflow (or tensorflow-cpu), numpy
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# Data
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
y_xor = np.array([0,1,1,0], dtype=np.float32)
y_xnor = 1 - y_xor

# XOR model (library-only: Keras handles backprop)
model_xor = keras.Sequential([
    layers.Dense(2, activation='tanh', input_shape=(2,)),
    layers.Dense(1, activation='sigmoid')
])
model_xor.compile(optimizer=keras.optimizers.SGD(learning_rate=0.1),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
model_xor.fit(X, y_xor, epochs=1000, batch_size=4, verbose=0)
probs_xor = model_xor.predict(X).flatten()
preds_xor = (probs_xor > 0.5).astype(int)
acc_xor = (preds_xor == y_xor).mean()

# XNOR model (library-only)
model_xnor = keras.Sequential([
    layers.Dense(2, activation='tanh', input_shape=(2,)),
    layers.Dense(1, activation='sigmoid')
])
model_xnor.compile(optimizer=keras.optimizers.SGD(learning_rate=0.1),
                   loss='binary_crossentropy',
                   metrics=['accuracy'])
model_xnor.fit(X, y_xnor, epochs=1000, batch_size=4, verbose=0)
probs_xnor = model_xnor.predict(X).flatten()
preds_xnor = (probs_xnor > 0.5).astype(int)
acc_xnor = (preds_xnor == y_xnor).mean()

# Output
print("XOR preds:", preds_xor, "probs:", np.round(probs_xor,3), "acc:", acc_xor)
print("XNOR preds:", preds_xnor, "probs:", np.round(probs_xnor,3), "acc:", acc_xnor)

# Show weights (library model attributes)
print("\nXOR model weights:")
for i, w in enumerate(model_xor.get_weights()):
    print(f"weight_{i}:\n", np.round(w,3))

print("\nXNOR model weights:")
for i, w in enumerate(model_xnor.get_weights()):
    print(f"weight_{i}:\n", np.round(w,3))
