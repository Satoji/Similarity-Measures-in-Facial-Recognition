from scipy.spatial.distance import cosine
from numpy import asarray
import numpy as np
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from PIL import Image
from mtcnn.mtcnn import MTCNN
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam

'''
Preprocess face by detecting face using MTCNN, extract box of the
detected face, crops and resizes face to 224x224 and returns processed
image as an array
'''
def preprocessFace(imgFile, size=(224, 224)):
    pixelData = pyplot.imread(imgFile)
    faceDetect = MTCNN()
    detectOutput = faceDetect.detect_faces(pixelData)
    x1, y1, width, height = detectOutput[0]['box']
    x2, y2 = x1 + width, y1 + height
    face = pixelData[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize(size)
    faceArr = asarray(image)
    return faceArr

def featureExtraction(imgFiles):
    faces = [preprocessFace(i) for i in imgFiles]
    preprocessed = asarray(faces, 'float32')
    preprocessed = preprocess_input(preprocessed, version=2)
    model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    featureVectors = model.predict(preprocessed)
    return featureVectors

names = ['Akshay Kumar', 'Alexandra Daddario', 'Alia Bhatt', 'Amitabh Bachchan', 'Andy Samberg', 'Anushka Sharma',
        'Billie Eilish', 'Brad Pitt', 'Camila Cabello', 'Charlize Theron', 'Claire Holt', 'Courtney Cox',
        'Dwayne Johnson', 'Elizabeth Olsen', 'Ellen Degeneres', 'Henry Cavill', 'Hrithik Roshan', 'Hugh Jackman',
        'Jessica Alba', 'Kashyap', 'Lisa Kudrow', 'Margot Robbie', 'Marmik', 'Natalie Portman', 'Priyanka Chopra',
        'Robert Downey Jr', 'Roger Federer', 'Tom Cruise', 'Vijay Deverakonda', 'Virat Kohli', 'Zac Efron']

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

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(embeddings, encoded_labels, test_size=0.2, random_state=42)

# Define the model
model = Sequential([
    Dense(128, activation='relu', input_shape=(2048,)),
    Dense(len(names), activation='softmax')
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, batch_size=5, epochs=5, validation_data=(X_val, y_val))

# Evaluate the model on a separate test set
test_loss, test_accuracy = model.evaluate(X_val, y_val)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# determine if a candidate face is a match for a known face
def cosine_similarity(known_embedding, candidate_embedding, thresh=0.5):
    # cosine similarity between embeddings
    dot_product = np.dot(known_embedding, candidate_embedding)
    norm_vector1 = np.linalg.norm(known_embedding)
    norm_vector2 = np.linalg.norm(candidate_embedding)
    similarity_score = dot_product / (norm_vector1 * norm_vector2)

    #similarity_score = cosine_similarity(known_embedding, candidate_embedding)
    if similarity_score >= thresh:
        print('> Face is a Match (Cosine Distance: %.3f >= %.3f)' % (similarity_score, thresh))
    else:
        print('> Face is NOT a Match (Cosine Distance: %.3f < %.3f)' % (similarity_score, thresh))
    
# determine if a candidate face is a match for a known face
def euclidean_distance(known_embedding, candidate_embedding, thresh=121):  # Adjust threshold as needed
    # calculate Euclidean distance between embeddings
    squared_diff = np.square(known_embedding - candidate_embedding)
    sum_squared_diff = np.sum(squared_diff)
    distance = np.sqrt(sum_squared_diff)
    
	#distance = euclidean_distance(known_embedding, candidate_embedding)
    if distance <= thresh:
        print('> Face is a Match (Euclidean Distance: %.3f <= Threshold: %.3f)' % (distance, thresh))
    else:
        print('> Face is NOT a Match (Euclidean Distance: %.3f > Threshold: %.3f)' % (distance, thresh))
     
# determine if a candidate face is a match for a known face
def manhattan_distance(known_embedding, candidate_embedding, thresh=3200):  # Adjust threshold as needed
    # calculate Manhattan distance between embeddings
	distance = np.sum(np.abs(known_embedding - candidate_embedding))
    #distance = manhattan_distance(known_embedding, candidate_embedding)
	if distance <= thresh:
		print('> Face is a Match (Manhattan Distance: %.3f <= Threshold: %.3f)' % (distance, thresh))
	else:
		print('> Face is NOT a Match (Manhattan Distance: %.3f > Threshold: %.3f)' % (distance, thresh))


names = ['Akshay Kumar', 'Alexandra Daddario', 'Alia Bhatt', 'Amitabh Bachchan', 'Andy Samberg', 'Anushka Sharma',
        'Billie Eilish', 'Brad Pitt', 'Camila Cabello', 'Charlize Theron', 'Claire Holt', 'Courtney Cox',
		'Dwayne Johnson', 'Elizabeth Olsen', 'Ellen Degeneres', 'Henry Cavill', 'Hrithik Roshan', 'Hugh Jackman',
        'Jessica Alba', 'Kashyap', 'Lisa Kudrow', 'Margot Robbie', 'Marmik', 'Natalie Portman', 'Priyanka Chopra',
        'Robert Downey Jr', 'Roger Federer', 'Tom Cruise', 'Vijay Deverakonda', 'Virat Kohli', 'Zac Efron']

# define imgFiles
#imgFiles = ['sharon_stone1.jpg', 'sharon_stone2.jpg', 'sharon_stone3.jpg', 'Dwayne Johnson_0.jpg']
#imgFiles = ['sharon_stone1.jpg', 'sharon_stone2.jpg', 'sharon_stone3.jpg', 'Faces/Dwayne Johnson/Dwayne Johnson_0.jpg']
imgFiles = ['sharon_stone1.jpg', 'sharon_stone2.jpg', 'sharon_stone3.jpg', 'Faces/Camila Cabello/Camila Cabello_0.jpg']
# get embeddings file imgFiles
embeddings = featureExtraction(imgFiles)
# define sharon stone
sharon_id = embeddings[0]
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