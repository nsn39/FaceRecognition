import tkinter as tk
import numpy as np 
import cv2 as cv
from PIL import Image, ImageTk
import argparse 
from enum import Enum
from predictor import preprocess_img, resize_image, calc_embeddings, calc_dist, normalize_embeddings

func_identifier = None

# Create a variable to keep track of the program state
class ProgState(Enum):
    NONE = 0
    CAMERA_BUFFER = 1
    DETECT_FACE = 2
    RECOGNIZE_FACE = 3

curr_state = ProgState.CAMERA_BUFFER
prev_embeddings = None

def capture_image():
    pass 

##
def detect_faces():
    global func_identifier
    print(func_identifier)

    global curr_state
    print(curr_state)

    print("Detecting faces...")
    # Set the current state
    curr_state = ProgState.DETECT_FACE

    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        return

    # Our operations on the frame come here
    cv2image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Detect faces
    faces = face_cascade.detectMultiScale(cv2image)

    for (x, y, w, h) in faces:
        rect_start_point = (x, y)
        rect_end_point = (x+w, y+h)
        color = (0, 255, 0) #BGR format
        thickness = 2

        cv2image = cv.rectangle(cv2image, rect_start_point, rect_end_point, color, thickness)
        # For each face:
        # 1. slice the image portion containing image.
        crop_img = cv2image[y:y+h, x:x+w]
        cv.imwrite('crop/one.png', crop_img)
    
        # 2. convert the image into a numpy array
        crop_img = resize_image(crop_img, 160)
        crop_img = preprocess_img(crop_img)
        
        crop_img = np.expand_dims(crop_img, axis=0)
        #print(crop_img.shape)
        # 3. generate embeddings for that image
        embeddings = normalize_embeddings(calc_embeddings(crop_img))
        print(type(embeddings))
        # Calculate that distance from last embeddings
        global prev_embeddings
        print("Distance: " , calc_dist(embeddings, prev_embeddings))
        prev_embeddings = embeddings
        #print(embeddings)
        #print(embeddings.shape)
        # 4. compare the generated embeddings against all the saved embeddings.
        # 5. if found, assign a name to the image and display it.

        
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)

    picture_label.imgtk = imgtk
    picture_label.configure(image=imgtk)

    if curr_state == ProgState.DETECT_FACE:
        print("2Reached here..")
        print(func_identifier)
        window.after_cancel(func_identifier)
        picture_label.after(1, detect_faces)

## Recognize the faces from saved faces..
def recognize_faces():
    pass 

def load_image():
    pass 

def capture_frame_and_show():
    print(curr_state)
    print("Original loop")

    if curr_state != ProgState.CAMERA_BUFFER:
        exit()

    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        return

    # Our operations on the frame come here
    cv2image = cv.cvtColor(frame, cv.COLOR_BGR2RGBA)

    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)

    picture_label.imgtk = imgtk
    picture_label.configure(image=imgtk)

    # Check the current state of the program
    if curr_state == ProgState.CAMERA_BUFFER:
        print("Reached here..")
        global func_identifier
        func_identifier = picture_label.after(1, capture_frame_and_show)
        

window = tk.Tk()

# Creating buttons frame
frame1 = tk.Frame(master=window)
frame1.pack()

# Create buttons within the first frame
load_btn = tk.Button(text="Load Image", master=frame1, command=load_image)
capture_btn = tk.Button(text="Capture", master=frame1, command=capture_image)
detect_btn = tk.Button(text="Detect Faces", master=frame1, command=detect_faces)
recognize_btn = tk.Button(text="Recognize Faces", master=frame1, command=recognize_faces)

load_btn.grid(row=0, column=0, sticky="w")
detect_btn.grid(row=0, column=1, sticky="w")
recognize_btn.grid(row=0, column=2, sticky="w")
capture_btn.grid(row=0, column=3, sticky="w")

# Creating viewing frame
frame2 = tk.Frame(master=window, width=500, height=500, bg="Green")
frame2.pack()

picture_label = tk.Label(master=frame2)
picture_label.pack()

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

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()


curr_state = ProgState.CAMERA_BUFFER
capture_frame_and_show()
#detect_faces()
window.mainloop()