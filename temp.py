import numpy as np
import tensorflow as tf
#from tensorflow.keras.applications.vggface import preprocess_input
from keras_vggface.utils import preprocess_input

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
def preprocess_image(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to create batch dimension
    return preprocess_input(img_array)

# Define function to extract features from an image
def extract_features(img_path, model):
    preprocessed_img = preprocess_image(img_path)
    return model.predict(preprocessed_img)

def extract_features2(img_paths, model):
    preprocessed_imgs = np.concatenate([preprocess_image(img_path) for img_path in img_paths], axis=0)
    return model.predict(preprocessed_imgs)



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
            features = extract_features(img_path, feature_extraction_model)
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
def cosine_similarity(known_embedding, candidate_embedding):
    dot_product = np.dot(known_embedding, candidate_embedding)
    norm_vector1 = np.linalg.norm(known_embedding)
    norm_vector2 = np.linalg.norm(candidate_embedding)
    similarity_score = dot_product / (norm_vector1 * norm_vector2)
    return similarity_score
'''

def cosine_similarity(known_embedding, candidate_embedding):
    known_embedding = known_embedding.flatten()  # Flatten the arrays if needed
    candidate_embedding = candidate_embedding.flatten()
    dot_product = np.dot(known_embedding, candidate_embedding)
    norm_vector1 = np.linalg.norm(known_embedding)
    norm_vector2 = np.linalg.norm(candidate_embedding)
    similarity_score = dot_product / (norm_vector1 * norm_vector2)
    return similarity_score


def euclidean_distance(known_embedding, candidate_embedding): 
    squared_diff = np.square(known_embedding - candidate_embedding)
    sum_squared_diff = np.sum(squared_diff)
    distance = np.sqrt(sum_squared_diff)
    return distance

def manhattan_distance(known_embedding, candidate_embedding):
    distance = np.sum(np.abs(known_embedding - candidate_embedding))
    return distance

def chebyshev_distance(known_embedding, candidate_embedding):
    distance = np.max(np.abs(known_embedding - candidate_embedding))
    return distance

dj0 = 'Faces/Dwayne Johnson/Dwayne Johnson_0.jpg'
dj1 = 'Faces/Tom Cruise/Tom Cruise_0.jpg'

# Extract features from the image file
fV0 = extract_features(dj0, feature_extraction_model)
fV1 = extract_features(dj1, feature_extraction_model)

# Print the feature vector
print("Feature vector:", fV0)
print("Feature vector:", fV1)

print("euc dis:", euclidean_distance(fV0,fV1))
print("man dis:", manhattan_distance(fV0,fV1))
print("che dis:", chebyshev_distance(fV0,fV1))
cosine_dist = 1 - cosine_similarity(fV0,fV1)
print("cos dis:", cosine_dist)
