import numpy as np
import tensorflow as tf
#from tensorflow.keras.applications.vggface import preprocess_input
from keras_vggface.utils import preprocess_input


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


test1 = [[ 10.98553,     -6.77058,      4.685156]]
test2 =  [[  6.8754864,   -1.2830105,    5.1923075]]
print("euc dis:", euclidean_distance(test1,test2))
print("man dis:", manhattan_distance(test1,test2))
print("che dis:", chebyshev_distance(test1,test2))
cosine_dist = 1 - cosine_similarity(test1,test2)
print("cos dis:", cosine_dist)
