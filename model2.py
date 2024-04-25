import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Filter only face images (label 0 in CIFAR-10 dataset)
train_faces = train_images[train_labels.flatten() == 0]
test_faces = test_images[test_labels.flatten() == 0]

# Normalize pixel values to range [0, 1]
train_faces = train_faces.astype('float32') / 255.0
test_faces = test_faces.astype('float32') / 255.0

# Split the data into training and validation sets
train_faces, val_faces = train_test_split(train_faces, test_size=0.1, random_state=42)

# Define the CNN architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification: face or not face
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_faces, np.ones(train_faces.shape[0]), epochs=10, batch_size=64, validation_data=(val_faces, np.ones(val_faces.shape[0])))

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_faces, np.ones(test_faces.shape[0]))
print('Test accuracy:', test_acc)
