from scipy.spatial.distance import cosine
from numpy import asarray
import numpy as np
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from PIL import Image
from mtcnn.mtcnn import MTCNN
from matplotlib import pyplot
import pandas
import time

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

cosine_threshold = 0.468
euclidean_threshold = 103.847
manhattan_threshold = 2750.149
chebyshev_threshold = 16.926

cosThresh0 = 0.0989
cosThresh100 = 0.838
cosThresh25 = cosThresh0 + 0.25 * (cosThresh100 - cosThresh0)
cosThresh75 = cosThresh0 + 0.75 * (cosThresh100 - cosThresh0)

eucThresh0 = 49.507
eucThresh100 = 158.187
eucThresh25 = eucThresh0 + 0.25 * (eucThresh100 - eucThresh0)
eucThresh75 = eucThresh0 + 0.75 * (eucThresh100 - eucThresh0)

manThresh0 = 1206.346
manThresh100 = 4293.950
manThresh25 = manThresh0 + 0.25 * (manThresh100 - manThresh0)
manThresh75 = manThresh0 + 0.75 * (manThresh100 - manThresh0)

cheThresh0 = 7.563
cheThresh100 = 26.288
cheThresh25 = cheThresh0 + 0.25 * (cheThresh100 - cheThresh0)
cheThresh75 = cheThresh0 + 0.75 * (cheThresh100 - cheThresh0)


# determine if a candidate face is a match for a known face
def cosine_similarity(known_embedding, candidate_embedding, thresh=cosThresh100):
    # cosine similarity between embeddings
    dot_product = np.dot(known_embedding, candidate_embedding)
    norm_vector1 = np.linalg.norm(known_embedding)
    norm_vector2 = np.linalg.norm(candidate_embedding)
    similarity_score = dot_product / (norm_vector1 * norm_vector2)

    #similarity_score = cosine_similarity(known_embedding, candidate_embedding)
    if similarity_score <= thresh:
        print('> Face is a Match (Cosine Distance: %.3f <= %.3f)' % (similarity_score, thresh))
    else:
        print('> Face is NOT a Match (Cosine Distance: %.3f > %.3f)' % (similarity_score, thresh))
    return similarity_score

# determine if a candidate face is a match for a known face
def euclidean_distance(known_embedding, candidate_embedding, thresh=eucThresh100):  # Adjust threshold as needed
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
def manhattan_distance(known_embedding, candidate_embedding, thresh=manThresh100):  # Adjust threshold as needed
    # calculate Manhattan distance between embeddings
    distance = np.sum(np.abs(known_embedding - candidate_embedding))
    #distance = manhattan_distance(known_embedding, candidate_embedding)
    if distance <= thresh:
        print('> Face is a Match (Manhattan Distance: %.3f <= Threshold: %.3f)' % (distance, thresh))
    else:
        print('> Face is NOT a Match (Manhattan Distance: %.3f > Threshold: %.3f)' % (distance, thresh))
    return distance

def chebyshev_distance(known_embedding, candidate_embedding, thresh=cheThresh100):  # Adjust threshold as needed
    # calculate Chebyshev distance between embeddings
    distance = np.max(np.abs(known_embedding - candidate_embedding))
    if distance <= thresh:
        print('> Face is a Match (Chebyshev Distance: %.3f <= Threshold: %.3f)' % (distance, thresh))
    else:
        print('> Face is NOT a Match (Chebyshev Distance: %.3f > Threshold: %.3f)' % (distance, thresh))
    return distance


names = ['Akshay Kumar', 'Alexandra Daddario', 'Alia Bhatt', 'Amitabh Bachchan', 'Andy Samberg', 'Anushka Sharma',
        'Billie Eilish', 'Brad Pitt', 'Camila Cabello', 'Charlize Theron', 'Claire Holt', 'Courtney Cox',
        'Dwayne Johnson', 'Elizabeth Olsen', 'Ellen Degeneres', 'Henry Cavill', 'Hrithik Roshan', 'Hugh Jackman',
        'Jessica Alba', 'Kashyap', 'Lisa Kudrow', 'Anne Hathaway', 'Arnold Schwarzenegger', 'Ben Afflek', 'Keanu Reeves',
        'Jerry Seinfeld', 'Kate Beckinsale', 'Lauren Cohan', 'Simon Pegg', 'Will Smith']
        
notinvited = ['Margot Robbie', 'Marmik', 'Natalie Portman', 'Priyanka Chopra',
        'Robert Downey Jr', 'Roger Federer', 'Tom Cruise', 'Vijay Deverakonda', 'Virat Kohli', 'Zac Efron']

#names = ['Akshay Kumar','Henry Cavill','Tom Cruise', 'Will Smith']
#notinvited = ['Zac Efron']


all = names + notinvited

print(f"{all=}")


# File path to images
def get_image_paths(name, idx):
    #return f'Faces/{name}/{name}_{idx}.jpg'
    return f'Faces/{name}/{name}_{idx}.jpg'


invited_embeddings = [featureExtraction([get_image_paths(name, 0)]) for name in names]              #registered
# Generate embeddings for both arrays
#names_embeddings = [featureExtraction([get_image_paths(name, 1), get_image_paths(name, 2)]) for name in names]
names_embeddings = [featureExtraction([get_image_paths(name, 1)]) for name in names]                #different images of registered
notinvited_embeddings = [featureExtraction([get_image_paths(name, 0)]) for name in notinvited]      #unregistered

# Combine embeddings
combined_embeddings = names_embeddings + notinvited_embeddings                                      #diff images of registered and unregistered

known_embeddings_array = np.array([embedding[0] for embedding in invited_embeddings])
covariance_matrix = np.cov(known_embeddings_array.T)

# Output file path
output_file = "comparison_results.txt"


def write_to_file(output_file, results):
    with open(output_file, 'w') as file:
        for result in results:
            file.write(result + '\n')


def measure_time(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time


results = []
true_positives = {'cosine_similarity': 0, 'euclidean_distance': 0, 'manhattan_distance': 0, 'chebyshev_dist': 0}
true_negatives = {'cosine_similarity': 0, 'euclidean_distance': 0, 'manhattan_distance': 0, 'chebyshev_dist': 0}
false_positives = {'cosine_similarity': 0, 'euclidean_distance': 0, 'manhattan_distance': 0, 'chebyshev_dist': 0}
false_negatives = {'cosine_similarity': 0, 'euclidean_distance': 0, 'manhattan_distance': 0, 'chebyshev_dist': 0}

outCount = 0
inCount = 0

total_time_euclidean = 0
total_time_cosine = 0
total_time_manhattan = 0
total_time_chebyshev = 0

cosDis = []
eucDis = []
manDis = []
cheDis = []

for combined_embedding, combined_name in zip(combined_embeddings, all):     #diff images of registered and unregistered
    print(f"1{combined_name=}")
    results.append(f"\nComparison Results for {combined_name}:")
    eucMatch = False  # Flag to check if there's any match
    cosMatch = False  # Flag to check if there's any match
    manMatch = False  # Flag to check if there's any match
    chebMatch = False
    minkMatch = False
    outCount += 1

    for invited_embedding, invited_name in zip(invited_embeddings, names):  #registered
        #print(f"2{combined_name=}")
        #print(f"2{invited_name=}")
        
        combined_embedding_values = combined_embedding[0]  # Extracting the embedding value
        invited_embedding_values = invited_embedding[0]
        #euclidean_dist, cosine_sim, manhattan_dist = similarity_metrics(combined_embedding_values, invited_embedding_values)
        
        #euclidean_dist, cosine_sim, manhattan_dist, chebyshev_dist = similarity_metrics(combined_embedding_values, invited_embedding_values)
        euclidean_dist, time_taken = measure_time(euclidean_distance, combined_embedding_values, invited_embedding_values)
        total_time_euclidean += time_taken
        cosine_sim, time_taken = measure_time(cosine_similarity, combined_embedding_values, invited_embedding_values)
        cosine_sim = 1 - cosine_sim
        total_time_cosine += time_taken
        manhattan_dist, time_taken = measure_time(manhattan_distance, combined_embedding_values, invited_embedding_values)
        total_time_manhattan += time_taken
        chebyshev_dist, time_taken = measure_time(chebyshev_distance, combined_embedding_values, invited_embedding_values)
        total_time_chebyshev += time_taken

        eucDis.append(euclidean_dist)
        cosDis.append(cosine_sim)
        manDis.append(manhattan_dist)
        cheDis.append(chebyshev_dist)

        
        result_str = f"{invited_name}: "
        result_str += f"Euclidean Distance: {euclidean_dist}, Cosine Similarity: {cosine_sim}, Manhattan Distance: {manhattan_dist}, chebyshev_dist: {chebyshev_dist}"
        results.append(result_str)

        #chebyshev_dist = chebyshev_distance(combined_embedding_values, invited_embedding_values)
        if chebyshev_dist <= cheThresh100: 
            chebMatch = True
            if combined_name in names and combined_name == invited_name: #if person entering is registered AND match
                true_positives['chebyshev_dist'] += 1
                results.append(f"CHTP: {combined_name} is registered and matches w/ registered faces {invited_name}")
            if combined_name in notinvited and combined_name != invited_name: #if person entering is not registered and matches
                false_positives['chebyshev_dist'] += 1
                results.append(f"CHFP: {combined_name} is not registered and matches w/ registered faces: {invited_name}")
        elif chebyshev_dist > cheThresh100:
            if combined_name in notinvited and combined_name != invited_name: #if person entering is nonregistered and DOES NOT match
                true_negatives['chebyshev_dist'] += 1
                results.append(f"CHTN: {combined_name} is not registered and DOES not match w/ {invited_name}")
            if combined_name in names and combined_name == invited_name: #if person entering is registered and DOES NOT match
                false_negatives['chebyshev_dist'] += 1
                results.append(f"CHFN: {combined_name} is registered and DOES not match w/ {invited_name}")

        if euclidean_dist <= eucThresh100:
            eucMatch = True
            if combined_name in names and combined_name == invited_name: #if person entering is registered AND match
                true_positives['euclidean_distance'] += 1
                results.append(f"ETP: {combined_name} is registered and matches w/ registered faces {invited_name}")
            if combined_name in notinvited and combined_name != invited_name: #if person entering is not registered and matches
                false_positives['euclidean_distance'] += 1
                results.append(f"EFP: {combined_name} is not registered and matches w/ registered faces: {invited_name}")
        elif euclidean_dist > eucThresh100:
            if combined_name in notinvited and combined_name != invited_name: #if person entering is nonregistered and DOES NOT match
                true_negatives['euclidean_distance'] += 1
                results.append(f"ETN: {combined_name} is not registered and DOES not match w/ {invited_name}")
            if combined_name in names and combined_name == invited_name: #if person entering is registered and DOES NOT match
                false_negatives['euclidean_distance'] += 1
                results.append(f"EFN: {combined_name} is registered and DOES not match w/ {invited_name}")
            #true positive: we correctly identify that the person is registered
            #true negative: we correctly identify that the person is not registered 
            #false positive: we incorrectly identify that the person who is not registered is shown that they are registered.
            #false negative: we incorrectly identify that the person who is registered is shown that they are not registered

            #combined embeddings is everyone trying to enter in (img1 from registered + img0 from nonregistered)
            #invited embeddings are those who are registered (img0 from registered)
            #notinvited embeddings are those who are NOT registered (img0 from nonregistered)
            results.append(f"Euclidean: {combined_name} who is entering the event did not match with {invited_name}")
            #print(f"We inside here")

        if cosine_sim <= cosThresh100:
            cosMatch = True
            if combined_name in names and combined_name == invited_name: #if person entering is registered AND match
                true_positives['cosine_similarity'] += 1
                results.append(f"CTP: {combined_name} is registered and matches w/ registered faces {invited_name}")
            if combined_name in notinvited and combined_name != invited_name: #if person entering is not registered and matches
                false_positives['cosine_similarity'] += 1
                results.append(f"CFP: {combined_name} is not registered and matches w/ registered faces: {invited_name}")
        elif cosine_sim > cosThresh100:
            if combined_name in notinvited and combined_name != invited_name: #if person entering is nonregistered and DOES NOT match
                true_negatives['cosine_similarity'] += 1
                results.append(f"CTN: {combined_name} is not registered and DOES not match w/ {invited_name}")
            if combined_name in names and combined_name == invited_name: #if person entering is registered and DOES NOT match
                false_negatives['cosine_similarity'] += 1
                results.append(f"CFN: {combined_name} is registered and DOES not match w/ {invited_name}")
            results.append(f"Cosine: {combined_name} who is entering the event did not match with {invited_name}")
            #print(f"We inside here")

        if manhattan_dist <= manThresh100:
            manMatch = True
            if combined_name in names and combined_name == invited_name: #if person entering is registered AND match
                true_positives['manhattan_distance'] += 1
                results.append(f"MTP: {combined_name} is registered and matches w/ registered faces {invited_name}")
            if combined_name in notinvited and combined_name != invited_name: #if person entering is not registered and matches
                false_positives['manhattan_distance'] += 1
                results.append(f"MFP: {combined_name} is not registered and matches w/ registered faces: {invited_name}")
        elif manhattan_dist > manThresh100:
            if combined_name in notinvited and combined_name != invited_name: #if person entering is nonregistered and DOES NOT match
                true_negatives['manhattan_distance'] += 1
                results.append(f"MTN: {combined_name} is not registered and DOES not match w/ {invited_name}")
            if combined_name in names and combined_name == invited_name: #if person entering is registered and DOES NOT match
                false_negatives['manhattan_distance'] += 1
                results.append(f"MFN: {combined_name} is registered and DOES not match w/ {invited_name}")
            results.append(f"Manhattan: {combined_name} who is entering the event did not match with {invited_name}")
            #print(f"We inside here")
        inCount += 1
        
        
write_to_file(output_file, results)

iterPerson = inCount/outCount

print("Euclidean Distance:")
print("TP:", true_positives['euclidean_distance'])
print("TN:", true_negatives['euclidean_distance'])
print("FP:", false_positives['euclidean_distance'])
print("FN:", false_negatives['euclidean_distance'])

print("\nCosine Similarity:")
print("TP:", true_positives['cosine_similarity'])
print("TN:", true_negatives['cosine_similarity'])
print("FP:", false_positives['cosine_similarity'])
print("FN:", false_negatives['cosine_similarity'])

print("\nManhattan Distance:")
print("TP:", true_positives['manhattan_distance'])
print("TN:", true_negatives['manhattan_distance'])
print("FP:", false_positives['manhattan_distance'])
print("FN:", false_negatives['manhattan_distance'])


print("\chebyshev_dist:")
print("TP:", true_positives['chebyshev_dist'])
print("TN:", true_negatives['chebyshev_dist'])
print("FP:", false_positives['chebyshev_dist'])
print("FN:", false_negatives['chebyshev_dist'])

print(f"{inCount=}")
print(f"{outCount=}")

# Time in seconds
print("Total time for Euclidean distance:", total_time_euclidean*1000, "ms")
print("Total time for Cosine similarity:", total_time_cosine*1000, "ms")
print("Total time for Manhattan distance:", total_time_manhattan*1000, "ms")
print("Total time for Chebyshev distance:", total_time_chebyshev*1000, "ms")

eucMin = min(eucDis)
eucMax = max(eucDis)
cosMin = min(cosDis)
cosMax = max(cosDis)
manMin = min(manDis)
manMax = max(manDis)
cheMin = min(cheDis)
cheMax = max(cheDis)

print(f"eucMin: {eucMin}")
print(f"eucMax: {eucMax}")
print(f"cosMin: {cosMin}")
print(f"cosMax: {cosMax}")
print(f"manMin: {manMin}")
print(f"manMax: {manMax}")
print(f"cheMin: {cheMin}")
print(f"cheMax: {cheMax}")

# Calculate FAR and FRR for Euclidean Distance
FAR_euclidean = false_positives['euclidean_distance'] / (false_positives['euclidean_distance'] + true_negatives['euclidean_distance'])
FRR_euclidean = false_negatives['euclidean_distance'] / (false_negatives['euclidean_distance'] + true_positives['euclidean_distance'])

# Calculate FAR and FRR for Cosine Similarity
FAR_cosine = false_positives['cosine_similarity'] / (false_positives['cosine_similarity'] + true_negatives['cosine_similarity'])
FRR_cosine = false_negatives['cosine_similarity'] / (false_negatives['cosine_similarity'] + true_positives['cosine_similarity'])

# Calculate FAR and FRR for Manhattan Distance
FAR_manhattan = false_positives['manhattan_distance'] / (false_positives['manhattan_distance'] + true_negatives['manhattan_distance'])
FRR_manhattan = false_negatives['manhattan_distance'] / (false_negatives['manhattan_distance'] + true_positives['manhattan_distance'])

# Calculate FAR and FRR for Chebyshev Distance
FAR_chebyshev = false_positives['chebyshev_dist'] / (false_positives['chebyshev_dist'] + true_negatives['chebyshev_dist'])
FRR_chebyshev = false_negatives['chebyshev_dist'] / (false_negatives['chebyshev_dist'] + true_positives['chebyshev_dist'])

print("FAR for Euclidean Distance:", FAR_euclidean)
print("FRR for Euclidean Distance:", FRR_euclidean)

print("FAR for Cosine Similarity:", FAR_cosine)
print("FRR for Cosine Similarity:", FRR_cosine)

print("FAR for Manhattan Distance:", FAR_manhattan)
print("FRR for Manhattan Distance:", FRR_manhattan)

print("FAR for Chebyshev Distance:", FAR_chebyshev)
print("FRR for Chebyshev Distance:", FRR_chebyshev)

print("cosThresh25:",cosThresh25)
print("cosThresh75:",cosThresh75)
print("eucThresh25:",eucThresh25)
print("eucThresh75:",eucThresh75)
print("manThresh25:",manThresh25)
print("manThresh75:",manThresh75)
print("cheThresh25:",cheThresh25)
print("cheThresh75:",cheThresh75)