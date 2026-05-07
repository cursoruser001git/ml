import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, SimpleRNN, LSTM

# ==========================================
# 1. LOAD AND PREPARE DATASET (Do this once)
# ==========================================
# Load the MNIST dataset (handwritten digits 0-9)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values from 0-255 down to 0-1 for easier training
x_train = x_train / 255.0
x_test = x_test / 255.0

# CNN needs a color channel dimension: Shape becomes (28, 28, 1)
x_train_cnn = x_train.reshape(-1, 28, 28, 1)

# RNN and LSTM need sequences: Shape stays (28, 28) -> 28 timesteps of 28 pixels

# ==========================================
# 2. Convolutional Neural Network (CNN)
# ==========================================
cnn_model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(10, activation='softmax')
])
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ==========================================
# 3. Recurrent Neural Network (RNN)
# ==========================================
rnn_model = Sequential([
    SimpleRNN(64, input_shape=(28, 28)), # Reads 28 rows, 28 pixels at a time
    Dense(10, activation='softmax')
])
rnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ==========================================
# 4. Long Short-Term Memory (LSTM)
# ==========================================
lstm_model = Sequential([
    LSTM(64, input_shape=(28, 28)),
    Dense(10, activation='softmax')
])
lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# To train any of them, you would use:
# cnn_model.fit(x_train_cnn, y_train, epochs=3)