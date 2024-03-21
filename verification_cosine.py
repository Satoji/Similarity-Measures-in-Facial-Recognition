# face verification with the VGGFace2 model
from matplotlib import pyplot
from PIL import Image
from numpy import asarray
import numpy as np
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
#from keras_vggface.utils import_input
from keras_vggface.utils import preprocess_input

# extract a single face from a given photograph
def extract_face(filename, required_size=(224, 224)):
	# load image from file
	pixels = pyplot.imread(filename)
	# create the detector, using default weights
	detector = MTCNN()
	# detect faces in the image
	results = detector.detect_faces(pixels)
	# extract the bounding box from the first face
	x1, y1, width, height = results[0]['box']
	x2, y2 = x1 + width, y1 + height
	# extract the face
	face = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array

# extract faces and calculate face embeddings for a list of photo files
def get_embeddings(filenames):
	# extract faces
	faces = [extract_face(f) for f in filenames]
	# convert into an array of samples
	samples = asarray(faces, 'float32')
	# prepare the face for the model, e.g. center pixels
	samples = preprocess_input(samples, version=2)
	# create a vggface model
	model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
	# perform prediction
	yhat = model.predict(samples)
	return yhat

'''
def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity
'''

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

'''
def euclidean_distance(vector1, vector2):
    squared_diff = np.square(vector1 - vector2)
    sum_squared_diff = np.sum(squared_diff)
    distance = np.sqrt(sum_squared_diff)
    return distance
'''
    
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

'''
def manhattan_distance(vector1, vector2):
	distance = np.sum(np.abs(vector1 - vector2))
	return distance
'''
     
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

# define filenames
#filenames = ['sharon_stone1.jpg', 'sharon_stone2.jpg', 'sharon_stone3.jpg', 'Dwayne Johnson_0.jpg']
#filenames = ['sharon_stone1.jpg', 'sharon_stone2.jpg', 'sharon_stone3.jpg', 'Faces/Dwayne Johnson/Dwayne Johnson_0.jpg']
filenames = ['sharon_stone1.jpg', 'sharon_stone2.jpg', 'sharon_stone3.jpg', 'Faces/Camila Cabello/Camila Cabello_0.jpg']
# get embeddings file filenames
embeddings = get_embeddings(filenames)
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