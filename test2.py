from scipy.spatial.distance import cosine
from numpy import asarray
import numpy as np
from keras_vggface.utils import preprocess_input
from PIL import Image
from mtcnn.mtcnn import MTCNN
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam

def featureExtraction(imgFiles):
    # Load MTCNN for face detection
    faceDetector = MTCNN()
    
    # Initialize list to store extracted features
    features = []
    
    # Iterate through each image file
    for imgFile in imgFiles:
        # Load and preprocess the image
        img = Image.open(imgFile)
        img = img.convert('RGB')  # Ensure image is in RGB format
        imgArray = np.array(img)
        
        # Detect faces in the image
        faces = faceDetector.detect_faces(imgArray)
        
        # Iterate through each detected face
        for face in faces:
            # Extract bounding box coordinates
            x, y, width, height = face['box']
            x1, y1 = max(x, 0), max(y, 0)
            x2, y2 = min(x + width, imgArray.shape[1]), min(y + height, imgArray.shape[0])
            
            # Crop face region and resize to 224x224
            faceImg = imgArray[y1:y2, x1:x2]
            faceImg = Image.fromarray(faceImg)
            faceImg = faceImg.resize((224, 224))
            
            # Convert to float array
            faceArray = np.array(faceImg, dtype=np.float32)  # Convert to float32
            
            # Preprocess input for your custom model
            faceArray = preprocess_input(faceArray, version=2)  # Assuming your model expects VGGFace preprocessing
            
            # Extract features using your custom model
            extractedFeature = custom_model.predict(np.expand_dims(faceArray, axis=0))[0]
            
            # Append extracted features to the list
            features.append(extractedFeature)
    
    # Convert list to numpy array
    features = np.array(features)
    
    return features


names = ['Akshay Kumar', 'Alexandra Daddario', 'Alia Bhatt', 'Amitabh Bachchan', 'Andy Samberg', 'Anushka Sharma',
        'Billie Eilish', 'Brad Pitt', 'Camila Cabello', 'Charlize Theron', 'Claire Holt', 'Courtney Cox',
        'Dwayne Johnson', 'Elizabeth Olsen', 'Ellen Degeneres', 'Henry Cavill', 'Hrithik Roshan', 'Hugh Jackman',
        'Jessica Alba', 'Kashyap', 'Lisa Kudrow', 'Margot Robbie', 'Marmik', 'Natalie Portman', 'Priyanka Chopra',
        'Robert Downey Jr', 'Roger Federer', 'Tom Cruise', 'Vijay Deverakonda', 'Virat Kohli', 'Zac Efron']

# Define your custom model
input_shape = (224, 224, 3)  # Assuming input shape for your custom model
inputs = Input(shape=input_shape)
# Define your custom model architecture here
# Example architecture:
x = Conv2D(32, (3, 3), activation='relu')(inputs)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
outputs = Dense(128, activation='relu')(x)

custom_model = Model(inputs, outputs)

# Compile your custom model
custom_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')  # Adjust optimizer and loss function as needed

# Load images and labels
imgFiles = []
labels = []
for name in names:
    imgFiles.append(f'Faces/{name}/{name}_0.jpg')
    labels.append(name)

# Extract features
embeddings = featureExtraction(imgFiles)

# Label encoding
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)


X_train, X_val, y_train, y_val = train_test_split(embeddings, encoded_labels, test_size=0.2, random_state=42)

# Reshape extracted features to match the input shape expected by the model
# Reshape the training data
X_train_reshaped = X_train.reshape(-1, 128)  # Assuming 128 features extracted by the custom model
# Reshape the validation data
X_val_reshaped = X_val.reshape(-1, 128)  # Assuming 128 features extracted by the custom model

# Define your custom model
input_shape = (128,)  # Assuming the output shape of your custom model is (None, 128)
inputs = Input(shape=input_shape)
# Define the rest of your custom model architecture here
# Example architecture:
x = Dense(64, activation='relu')(inputs)
outputs = Dense(128, activation='relu')(x)

history = custom_model.fit(X_train_reshaped, y_train, batch_size=5, epochs=5, validation_data=(X_val_reshaped, y_val))

# Evaluate the model on the validation set
test_loss, test_accuracy = custom_model.evaluate(X_val_reshaped, y_val)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)



# define imgFiles
imgFiles = ['sharon_stone1.jpg', 'sharon_stone2.jpg', 'sharon_stone3.jpg', 'Faces/Dwayne Johnson/Dwayne Johnson_0.jpg']

# define sharon stone
sharon_id = embeddings[0]

# determine if a candidate face is a match for a known face
def cosine_similarity(known_embedding, candidate_embedding, thresh=0.5):
    dot_product = np.dot(known_embedding, candidate_embedding)
    norm_vector1 = np.linalg.norm(known_embedding)
    norm_vector2 = np.linalg.norm(candidate_embedding)
    similarity_score = dot_product / (norm_vector1 * norm_vector2)

    if similarity_score >= thresh:
        print('> Face is a Match (Cosine Distance: %.3f >= %.3f)' % (similarity_score, thresh))
    else:
        print('> Face is NOT a Match (Cosine Distance: %.3f < %.3f)' % (similarity_score, thresh))

def euclidean_distance(known_embedding, candidate_embedding, thresh=121):
    squared_diff = np.square(known_embedding - candidate_embedding)
    sum_squared_diff = np.sum(squared_diff)
    distance = np.sqrt(sum_squared_diff)

    if distance <= thresh:
        print('> Face is a Match (Euclidean Distance: %.3f <= Threshold: %.3f)' % (distance, thresh))
    else:
        print('> Face is NOT a Match (Euclidean Distance: %.3f > Threshold: %.3f)' % (distance, thresh))

def manhattan_distance(known_embedding, candidate_embedding, thresh=3200):
    distance = np.sum(np.abs(known_embedding - candidate_embedding))

    if distance <= thresh:
        print('> Face is a Match (Manhattan Distance: %.3f <= Threshold: %.3f)' % (distance, thresh))
    else:
        print('> Face is NOT a Match (Manhattan Distance: %.3f > Threshold: %.3f)' % (distance, thresh))


# verify known photos of sharon
print("Euclidean")
print('Positive Tests')
euclidean_distance(embeddings[0], embeddings[1])
euclidean_distance(embeddings[0], embeddings[2])

# verify known photos of other people
print('Negative Tests')
euclidean_distance(embeddings[0], embeddings[3])

print("Cosine")
print('Positive Tests')
cosine_similarity(embeddings[0], embeddings[1])
cosine_similarity(embeddings[0], embeddings[2])

# verify known photos of other people
print('Negative Tests')
cosine_similarity(embeddings[0], embeddings[3])

print("Manhattan")
print('Positive Tests')
manhattan_distance(embeddings[0], embeddings[1])
manhattan_distance(embeddings[0], embeddings[2])

# verify known photos of other people
print('Negative Tests')
manhattan_distance(embeddings[0], embeddings[3])
