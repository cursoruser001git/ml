# Requirements: tensorflow (or tensorflow-cpu), numpy
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1) Create sample image dataset (synthetic CIFAR-10 like images: 32x32x3)
num_classes = 10
np.random.seed(42)

# Generate synthetic training data (100 samples, 32x32x3 images)
x_train = np.random.randint(0, 256, (100, 32, 32, 3), dtype=np.uint8)
y_train = np.random.randint(0, num_classes, (100, 1))

# Generate synthetic test data (30 samples)
x_test = np.random.randint(0, 256, (30, 32, 32, 3), dtype=np.uint8)
y_test = np.random.randint(0, num_classes, (30, 1))

# 2) Preprocess (library helpers)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
y_train = to_categorical(y_train.ravel(), num_classes)
y_test = to_categorical(y_test.ravel(), num_classes)

# Use validation split for small dataset
x_val = x_test[:10]
y_val = y_test[:10]

# 3) Build a simple CNN using Keras (minimal custom code)
model = models.Sequential([
    layers.Input(shape=(32,32,3)),
    layers.Conv2D(32, (3,3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3,3), activation="relu", padding="same"),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3,3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3,3), activation="relu", padding="same"),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(optimizer=keras.optimizers.Adam(),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

model.summary()

# 4) Data augmentation (library)
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(x_train)

# 5) Train (Keras handles backprop)
batch_size = 32
epochs = 15
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=batch_size),
    steps_per_epoch=max(1, len(x_train) // batch_size),
    epochs=epochs,
    validation_data=(x_val, y_val),
    verbose=2
)

# 6) Evaluate on test data
test_subset = x_test
test_labels = y_test
loss, acc = model.evaluate(test_subset, test_labels, verbose=0)
print(f"Test subset loss: {loss:.4f}, accuracy: {acc:.4f}")

# 7) Save model (optional)
model.save("simple_cnn_cifar10.h5")
