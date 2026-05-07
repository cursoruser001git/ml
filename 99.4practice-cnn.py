import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# 1. Load the Dataset (MNIST: Handwritten Digits)
# Keras handles the train_test_split for us automatically here!
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# 2. Preprocess the Data
# CNNs expect a 4D array: (Number of Images, Height, Width, Color Channels)
# MNIST is grayscale, so Color Channels = 1. (If it were RGB, it would be 3).
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# Normalize pixel values to be between 0 and 1 (they start as 0 to 255)
train_images = train_images / 255.0
test_images = test_images / 255.0

# 3. Build the CNN Model (The Architecture)
model = models.Sequential()

# --- PART 1: FEATURE EXTRACTION (The "Convolutional" Base) ---
# Layer 1: The "Eyes". 32 filters, each 3x3 pixels.
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# Layer 2: The "Shrinker". Reduces image size to save memory and prevent overfitting.
model.add(layers.MaxPooling2D((2, 2)))
# Layer 3: More eyes to find deeper patterns (like loops and lines).
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# --- PART 2: CLASSIFICATION (The "Brain") ---
# Flatten takes our 2D image maps and unrolls them into a flat 1D line of numbers
model.add(layers.Flatten())
# A standard Hidden Layer with 64 neurons
model.add(layers.Dense(64, activation='relu'))
# The Output Layer: 10 neurons (one for each digit 0-9). Softmax turns answers into percentages.
model.add(layers.Dense(10, activation='softmax'))

# 4. Compile the Model (Setup the Backpropagation)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. Train the Model! 
# epochs=5 means the model will study the entire dataset 5 times.
print("Starting training...")
history = model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

# 6. Evaluate the final score
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"\nFinal Test Accuracy: {round(test_acc * 100, 2)}%")