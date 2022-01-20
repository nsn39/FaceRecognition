import numpy as np 
import os 
import matplotlib.pyplot as plt 
from keras.models import load_model
from code.inception_resnet_v1 import InceptionResNetV1
from skimage.transform import resize

# Load weights to the model.
weights_path = 'model/weights/facenet_keras_weights.h5'
model = InceptionResNetV1(weights_path=weights_path)

# Preprocess the image
def preprocess_img(img):
    if img.ndim != 3:
        raise ValueError("The dimension of input tensor must be 3.") 
    
    # Mean normalization on the image.
    mean = np.mean(img, axis=(0,1), keepdims=True)
    std = np.std(img, axis=(0,1), keepdims=True)
    img = (img - mean) / std
    return img

# Load and align the image
# Resize the image to a fixed size
def resize_image(img, size):
    return resize(img, (size, size), mode='reflect')


# Calculating the embeddings to our image.
def calc_embeddings(image):
    return model.predict(image) 

def calc_dist(embd1, embd2):
    """
        Calculates the euclidean distance between two embeddings.
        Both parameters must be numpy array.
    """
    if embd2 is None:
        return 0
    
    diff = (embd1 - embd2).flatten()
    return np.sqrt(np.sum(np.square(diff)))


def normalize_embeddings(embd):
    magnitude = np.sqrt(np.sum(np.square(embd)))
    return embd/magnitude 
