from scipy.spatial.distance import cosine
from numpy import asarray
import numpy as np
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from PIL import Image
from mtcnn.mtcnn import MTCNN
from matplotlib import pyplot

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
    return similarity_score

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
    return distance

# determine if a candidate face is a match for a known face
def manhattan_distance(known_embedding, candidate_embedding, thresh=3200):  # Adjust threshold as needed
    # calculate Manhattan distance between embeddings
    distance = np.sum(np.abs(known_embedding - candidate_embedding))
    #distance = manhattan_distance(known_embedding, candidate_embedding)
    if distance <= thresh:
        print('> Face is a Match (Manhattan Distance: %.3f <= Threshold: %.3f)' % (distance, thresh))
    else:
        print('> Face is NOT a Match (Manhattan Distance: %.3f > Threshold: %.3f)' % (distance, thresh))
    return distance

# determine if a candidate face is a match for a known face
def similarity_metrics(known_embedding, candidate_embedding):
    # Euclidean distance between embeddings
    euclidean_dist = euclidean_distance(known_embedding, candidate_embedding)
    
    # Cosine similarity between embeddings
    cosine_sim = 1 - cosine_similarity(known_embedding, candidate_embedding)
    
    # Manhattan distance between embeddings
    manhattan_dist = manhattan_distance(known_embedding, candidate_embedding)
    
    return euclidean_dist, cosine_sim, manhattan_dist

names = ['Akshay Kumar', 'Alexandra Daddario', 'Alia Bhatt', 'Amitabh Bachchan', 'Andy Samberg', 'Anushka Sharma',
        'Billie Eilish', 'Brad Pitt', 'Camila Cabello', 'Charlize Theron', 'Claire Holt', 'Courtney Cox',
        'Dwayne Johnson', 'Elizabeth Olsen', 'Ellen Degeneres']
        
notinvited = ['Henry Cavill', 'Hrithik Roshan', 'Hugh Jackman',
        'Jessica Alba', 'Kashyap', 'Lisa Kudrow', 'Margot Robbie', 'Marmik', 'Natalie Portman', 'Priyanka Chopra',
        'Robert Downey Jr', 'Roger Federer', 'Tom Cruise', 'Vijay Deverakonda', 'Virat Kohli', 'Zac Efron']

all = ['Akshay Kumar', 'Alexandra Daddario', 'Alia Bhatt', 'Amitabh Bachchan', 'Andy Samberg', 'Anushka Sharma',
        'Billie Eilish', 'Brad Pitt', 'Camila Cabello', 'Charlize Theron', 'Claire Holt', 'Courtney Cox',
        'Dwayne Johnson', 'Elizabeth Olsen', 'Ellen Degeneres', 'Henry Cavill', 'Hrithik Roshan', 'Hugh Jackman',
        'Jessica Alba', 'Kashyap', 'Lisa Kudrow', 'Margot Robbie', 'Marmik', 'Natalie Portman', 'Priyanka Chopra',
        'Robert Downey Jr', 'Roger Federer', 'Tom Cruise', 'Vijay Deverakonda', 'Virat Kohli', 'Zac Efron']

# File path to images
def get_image_paths(name, idx):
    return f'Faces/{name}/{name}_{idx}.jpg'

# Generate embeddings for both arrays
names_embeddings = [featureExtraction([get_image_paths(name, 1), get_image_paths(name, 2)]) for name in names]
notinvited_embeddings = [featureExtraction([get_image_paths(name, 0)]) for name in notinvited]

# Combine embeddings
combined_embeddings = names_embeddings + notinvited_embeddings

# Output file path
output_file = "comparison_results.txt"

# Open file for writing
with open(output_file, "w") as file:
    for idx, combined_embedding in enumerate(combined_embeddings):
        file.write(f"Comparison results for image {idx + 1} ({all[idx]}):\n")
        for name_idx, name_embedding in enumerate(names_embeddings):
            euclidean_dist, cosine_sim, manhattan_dist = similarity_metrics(combined_embedding[0], name_embedding[0])
            file.write(f"Name: {names[name_idx]}, Euclidean Distance: {euclidean_dist}, Cosine Similarity: {cosine_sim}, Manhattan Distance: {manhattan_dist}\n")
        file.write("\n")










'''
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
'''