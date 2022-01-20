import os
import json
import argparse 
import cv2 as cv 
import numpy as np
from predictor import resize_image, preprocess_img, calc_embeddings, normalize_embeddings
from utils import load_dict_from_json

def get_embeddings_dict():
    """
        Load all the pictures on the 'images' folder. Detect faces on each of them.
        Calculate the embeddings of each of the faces detected.
        Return the result in a dictionary.
    """
    img_dir = os.getcwd() + "/images"
    img_files = os.listdir(img_dir)
    name_to_embeddings = {}

    for file in img_files:
        file_path = os.getcwd() + "/images/" + file
        file_name = file.split('.')[0]
        
        raw_image = cv.imread(file_path)
        cvt_image = cv.cvtColor(raw_image, cv.COLOR_BGR2RGB)
        faces = face_cascade.detectMultiScale(cvt_image)

        assert(len(faces) == 1)

        (x, y, w, h) = faces[0]
        face_img = cvt_image[y:y+h, x:x+w]
        face_img = resize_image(face_img, 160)
        face_img = preprocess_img(face_img)
        face_img = np.expand_dims(face_img, axis=0)
        face_embeddings = normalize_embeddings(calc_embeddings(face_img)).flatten().tolist()
        name_to_embeddings[file_name] = face_embeddings
    
    return name_to_embeddings

def dump_dict_to_json():
    embeddings_dict = get_embeddings_dict()

    with open("database.json", "w") as file:
        json.dump(embeddings_dict, file) 


if __name__=='__main__':
    # Create a cascade classifier object
    face_cascade = cv.CascadeClassifier()

    # Load all the cascade models
    parser = argparse.ArgumentParser(description='Parser for cascade classifiers')
    parser.add_argument('--face_cascade', help='Path to face cascade', default='data/haarcascades/haarcascade_frontalface_alt.xml')
    args = parser.parse_args()
    face_cascade_name = args.face_cascade

    if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
        print("Couldn't load the face cascade model.")
        exit()

    dump_dict_to_json()
    my_dict = load_dict_from_json('database.json')
    print(my_dict['Nishan_Poudel'])