import numpy as np
import tensorflow as tf
#from tensorflow.keras.applications.vggface import preprocess_input
from keras_vggface.utils import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Define Feature Extraction Model
def create_feature_extraction_model(input_shape=(224, 224, 3)):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128)  # Output layer for feature vector
    ])
    return model

# Define function to preprocess an image
def preprocessFace(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to create batch dimension
    return preprocess_input(img_array)

# Define function to extract features from an image
#def featureExtraction(img_paths, model):
#    preprocessed_imgs = np.concatenate([preprocessFace(img_path) for img_path in img_paths], axis=0)
#    return model.predict(preprocessed_imgs)

def featureExtraction(img_path, model):
    preprocessed_img = preprocessFace(img_path)
    return model.predict(preprocessed_img)

# Example usage

# Create feature extraction model
feature_extraction_model = create_feature_extraction_model()

# Define function to get image paths
def get_image_paths(name, idx):
    return f'Faces/{name}/{name}_{idx}.jpg'

names = ['Akshay Kumar', 'Alexandra Daddario', 'Alia Bhatt', 'Amitabh Bachchan', 'Andy Samberg', 'Anushka Sharma',
        'Billie Eilish', 'Brad Pitt', 'Camila Cabello', 'Charlize Theron', 'Claire Holt', 'Courtney Cox',
        'Dwayne Johnson', 'Elizabeth Olsen', 'Ellen Degeneres', 'Henry Cavill', 'Hrithik Roshan', 'Hugh Jackman',
        'Jessica Alba', 'Kashyap', 'Lisa Kudrow', 'Anne Hathaway', 'Arnold Schwarzenegger', 'Ben Afflek', 'Keanu Reeves',
        'Jerry Seinfeld', 'Kate Beckinsale', 'Lauren Cohan', 'Simon Pegg', 'Will Smith', 'Margot Robbie', 'Marmik', 'Natalie Portman', 'Priyanka Chopra',
        'Robert Downey Jr', 'Roger Federer', 'Tom Cruise', 'Vijay Deverakonda', 'Virat Kohli', 'Zac Efron']

# Initialize lists to store features and labels
X_train = []
y_train = []

# Iterate through each name
for name in names:
    # Iterate through each image index (up to 7)
    for idx in range(1, 50):
        img_path = get_image_paths(name, idx)
        try:
            # Extract features from the image
            features = featureExtraction(img_path, feature_extraction_model)
            # Append features to X_train
            X_train.append(features)
            # Append label to y_train (you may need to encode the label)
            y_train.append(name)
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")

# Convert lists to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

print("X_train:", X_train)
print("y_train:", y_train)
# Print shapes of X_train and y_train to verify
print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Split the data into training and validation sets
X_train, X_val, y_train_encoded, y_val_encoded = train_test_split(X_train, y_train_encoded, test_size=0.2, random_state=42)

# Define and train a Support Vector Machine (SVM) classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train_encoded)

# Predict labels for the validation set
y_pred = svm_classifier.predict(X_val)

# Calculate accuracy
accuracy = accuracy_score(y_val_encoded, y_pred)
print("Validation Accuracy:", accuracy)

#img_path = 'Faces/Dwayne Johnson/Dwayne Johnson_0.jpg'

# Extract features from the image file
#feature_vector = featureExtraction(img_path, feature_extraction_model)

# Print the feature vector
#print("Feature vector:", feature_vector)






'''
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# Define Face Detection Model
def create_face_detection_model(input_shape=(224, 224, 3)):
    model = tf.keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(4)  # 4 outputs for bounding box coordinates (x, y, width, height)
    ])
    return model

def create_feature_extraction_model(input_shape=(224, 224, 3)):
    model = tf.keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(128)  # Output layer for feature vector
    ])
    return model

def train_face_detection_model(X_train, y_train):
    face_detection_model = create_face_detection_model()
    face_detection_model.compile(optimizer='adam', loss='mse')
    face_detection_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    return face_detection_model


face_detection_model = train_face_detection_model(X_train, y_train)

def get_image_paths(name, idx):
    #return f'Faces/{name}/{name}_{idx}.jpg'
    return f'Faces/{name}/{name}_{idx}.jpg'

# Assuming you have defined the create_feature_extraction_model function
feature_extraction_model = create_feature_extraction_model()

# Function to preprocess an image
def preprocessFace(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to create batch dimension
    return preprocess_input(img_array)

# Function to extract features from an image
def featureExtraction(img_path):
    preprocessed_img = preprocessFace(img_path)
    return feature_extraction_model.predict(preprocessed_img)

# Assuming you have defined the get_image_paths function
names = ['Akshay Kumar', 'Alexandra Daddario', 'Alia Bhatt', 'Amitabh Bachchan', 'Andy Samberg', 'Anushka Sharma',
        'Billie Eilish', 'Brad Pitt', 'Camila Cabello', 'Charlize Theron', 'Claire Holt', 'Courtney Cox',
        'Dwayne Johnson', 'Elizabeth Olsen', 'Ellen Degeneres', 'Henry Cavill', 'Hrithik Roshan', 'Hugh Jackman',
        'Jessica Alba', 'Kashyap', 'Lisa Kudrow', 'Anne Hathaway', 'Arnold Schwarzenegger', 'Ben Afflek', 'Keanu Reeves',
        'Jerry Seinfeld', 'Kate Beckinsale', 'Lauren Cohan', 'Simon Pegg', 'Will Smith', 'Margot Robbie', 'Marmik', 'Natalie Portman', 'Priyanka Chopra',
        'Robert Downey Jr', 'Roger Federer', 'Tom Cruise', 'Vijay Deverakonda', 'Virat Kohli', 'Zac Efron']

# Initialize lists to store features and labels
X_train = []
y_train = []

# Iterate through each name
for name in names:
    # Iterate through each image index (up to 7)
    for idx in range(1, 8):
        img_path = get_image_paths(name, idx)
        try:
            # Extract features from the image
            features = featureExtraction(img_path)
            # Append features to X_train
            X_train.append(features)
            # Append label to y_train (you may need to encode the label)
            y_train.append(name)
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")

# Convert lists to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

# Print shapes of X_train and y_train to verify
print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)

'''