import json
import numpy as np
from predictor import calc_dist

def load_dict_from_json(filename):
    """
        Get a python dictionary file from a json dump.
    """
    with open(filename) as json_file:
        embeddings_dict = json.load(json_file) 

    return embeddings_dict

def get_name_from_embeddings(embeddings_dict, embeddings):
    """
        Get the name of the nearest face in the database from the json file.
    """
    embeddings = np.asarray(embeddings)
    min_dist = 5000.00
    face_name = "Unknown"

    for key, val in embeddings_dict.items():
        curr_embeddings = np.asarray(val)

        face_dist = calc_dist(curr_embeddings, embeddings)

        if face_dist < min_dist:
            min_dist = face_dist 
            face_name = key 

    return face_name