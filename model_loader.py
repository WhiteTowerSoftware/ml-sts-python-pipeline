"""This will be used as an entry point when serving the model"""
from sagemaker_containers.beta.framework import content_types, encoders
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
import string
import numpy as np
import joblib
import json
import os


def count_words(list_of_words):
    """"""
    corpus_dict = {}
    for w in list_of_words:
        corpus_dict[w] = corpus_dict.get(w, 0.0) + 1.0
        
    return corpus_dict


def CountVectorizer(string_dict, set_both_strings):
    """Whit padding included""" 
    vector = []
    for key in set_both_strings: 
        if key in string_dict: 
            vector.append(int(string_dict[key])) 
        else:
            vector.append(0)
    return vector

def min_max_range(x, range_values):
    return [round(((xx-min(x))/(1.0*(max(x)-min(x))))*(range_values[1]-range_values[0])+range_values[0],5) for xx in x]


def preprocess_request(s1, s2):
    _VALID_METRICS_ = [
        'euclidean', 'l2', 'l1', 'manhattan', 'cityblock', 'braycurtis', 
        'canberra', 'chebyshev', 'correlation', 'cosine', 'dice', 'hamming',
        'jaccard', 'kulsinski', 'matching', 'minkowski', 'rogerstanimoto',
        'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 
        'sqeuclidean', 'yule']

    distances_matrix = []

    s1 = s1.translate(str.maketrans("", "", string.punctuation))
    s2 = s2.translate(str.maketrans("", "", string.punctuation))

    s1 = count_words(s1.split())
    s2 = count_words(s2.split())

    c = set(s1).union(set(s2))

    s1 = CountVectorizer(s1, c)
    s2 = CountVectorizer(s2, c)

    s1 = np.array(s1,dtype=np.int32)
    s2 = np.array(s2,dtype=np.int32)

    s1 = s1.reshape(1,-1)
    s2 = s2.reshape(1,-1)

    # get all distances
    vector = []
    for distance in _VALID_METRICS_:
        _, dist =  pairwise_distances_argmin_min(s1, s2, axis=1, metric=distance)
        vector.append(dist[0])

    distances_matrix.append(vector)

    '''
    Scaling
    '''
    _DISTANCE_MATRIX_NORM_ = []
    for vector in distances_matrix:
        _DISTANCE_MATRIX_NORM_.append(min_max_range(vector, (0.0,1.0)))
    distances_matrix = np.array(_DISTANCE_MATRIX_NORM_)

    '''
    Clean null values if any
    '''

    distances_matrix[np.isnan(distances_matrix)] = np.nanmean(distances_matrix)
    
    return distances_matrix


def model_fn(model_dir):
    """Deserialized and return fitted model
    Note that this should have the same name as the serialized model in the main method
    """
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf


def input_fn(input_data: str, content_type: str) -> np.array:
    """Assumes this endpoint will receive a json with two strings

    {
        "s1": "some string",
        "s2": "some other string"
    }
    
    return a np.array of distances for the model
    """
    try:
        data = json.loads(input_data)
        if 's1' not in data:
            raise Exception("Missing required data")
        if 's2' not in data:
            raise Exception("Missing required data")
    except ValueError:
        print("Decoding JSON Error")
        raise

    return preprocess_request(data.get('s1'), data.get('s2'))
