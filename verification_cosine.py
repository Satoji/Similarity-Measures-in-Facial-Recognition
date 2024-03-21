from scipy.spatial.distance import cosine
from numpy import asarray
import numpy as np
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from PIL import Image
from mtcnn.mtcnn import MTCNN
from matplotlib import pyplot
import pandas

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
        'Dwayne Johnson', 'Elizabeth Olsen', 'Ellen Degeneres', 'Henry Cavill', 'Hrithik Roshan', 'Hugh Jackman',
        'Jessica Alba', 'Kashyap', 'Lisa Kudrow']
        
notinvited = ['Margot Robbie', 'Marmik', 'Natalie Portman', 'Priyanka Chopra',
        'Robert Downey Jr', 'Roger Federer', 'Tom Cruise', 'Vijay Deverakonda', 'Virat Kohli', 'Zac Efron']

all = names + notinvited

print(f"{all=}")


# File path to images
def get_image_paths(name, idx):
    return f'Faces/{name}/{name}_{idx}.jpg'


invited_embeddings = [featureExtraction([get_image_paths(name, 0)]) for name in names]
# Generate embeddings for both arrays
#names_embeddings = [featureExtraction([get_image_paths(name, 1), get_image_paths(name, 2)]) for name in names]
names_embeddings = [featureExtraction([get_image_paths(name, 1)]) for name in names]
notinvited_embeddings = [featureExtraction([get_image_paths(name, 0)]) for name in notinvited]

# Combine embeddings
combined_embeddings = names_embeddings + notinvited_embeddings

# Output file path
output_file = "comparison_results.txt"


def write_to_file(output_file, results):
    with open(output_file, 'w') as file:
        for result in results:
            file.write(result + '\n')

results = []
false_positives = {'cosine_similarity': [], 'euclidean_distance': [], 'manhattan_distance': []}
false_negatives = {'cosine_similarity': [], 'euclidean_distance': [], 'manhattan_distance': []}

# Initialize counts for true positives and true negatives
true_positives = {'cosine_similarity': 0, 'euclidean_distance': 0, 'manhattan_distance': 0}
true_negatives = {'cosine_similarity': 0, 'euclidean_distance': 0, 'manhattan_distance': 0}


for combined_embedding, combined_name in zip(combined_embeddings, all):
    results.append(f"\nComparison Results for {combined_name}:")

    for invited_embedding, invited_name in zip(invited_embeddings, names):
        combined_embedding_values = combined_embedding[0]  # Extracting the embedding value
        invited_embedding_values = invited_embedding[0]
        euclidean_dist, cosine_sim, manhattan_dist = similarity_metrics(combined_embedding_values, invited_embedding_values)
        result_str = f"{invited_name}: "
        result_str += f"Euclidean Distance: {euclidean_dist}, Cosine Similarity: {cosine_sim}, Manhattan Distance: {manhattan_dist}"
        results.append(result_str)
         # Check for false positives and false negatives
        if combined_name in notinvited and invited_name in names:
            if cosine_sim >= 0.5:  # Adjust threshold as needed
                false_positives['cosine_similarity'].append((combined_name, invited_name))
            if euclidean_dist <= 121:  # Adjust threshold as needed
                false_positives['euclidean_distance'].append((combined_name, invited_name))
            if manhattan_dist <= 3200:  # Adjust threshold as needed
                false_positives['manhattan_distance'].append((combined_name, invited_name))
        elif combined_name in names and invited_name in notinvited:
            if cosine_sim < 0.5:  # Adjust threshold as needed
                false_negatives['cosine_similarity'].append((combined_name, invited_name))
            if euclidean_dist > 121:  # Adjust threshold as needed
                false_negatives['euclidean_distance'].append((combined_name, invited_name))
            if manhattan_dist > 3200:  # Adjust threshold as needed
                false_negatives['manhattan_distance'].append((combined_name, invited_name))

        
write_to_file(output_file, results)

# Ensure all arrays have the same length
max_length = max(len(false_positives[key]) for key in false_positives)
for key in false_positives:
    while len(false_positives[key]) < max_length:
        false_positives[key].append(None)

max_length = max(len(false_negatives[key]) for key in false_negatives)
for key in false_negatives:
    while len(false_negatives[key]) < max_length:
        false_negatives[key].append(None)

# Create DataFrames for false positives and false negatives
false_positives_df = pandas.DataFrame.from_dict(false_positives)
false_negatives_df = pandas.DataFrame.from_dict(false_negatives)

# Visualize false positives and false negatives for each metric
metrics = ['cosine_similarity', 'euclidean_distance', 'manhattan_distance']
pyplot.figure(figsize=(10, 6))

for i, metric in enumerate(metrics, start=1):
    pyplot.subplot(1, 3, i)
    pyplot.bar(['False Positives', 'False Negatives'], [len(false_positives[metric]), len(false_negatives[metric])])
    pyplot.title(metric)
    pyplot.xlabel('Type of Error')
    pyplot.ylabel('Count')

pyplot.tight_layout()
pyplot.show()



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