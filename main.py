import tkinter as tk
import numpy as np 
import cv2 as cv
from PIL import Image, ImageTk

def capture_image():
    pass 

##
def detect_faces():
    print("Detecting faces...")

    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        return

    # Our operations on the frame come here
    cv2image = cv.cvtColor(frame, cv.COLOR_BGR2RGBA)

    cv2image = cv.cvtColor(cv2image, cv.COLOR_BGR2GRAY)
    cv2image = cv.equalizeHist(cv2image)

    # Detect faces
    faces = face_cascade.detectMultiScale(cv2image)

    for (x, y, w, h) in faces:
        rect_start_point = (x, y)
        rect_end_point = (x+w, y+h)
        color = (0, 255, 0) #BGR format
        thickness = 2

        cv2image = cv.rectangle(cv2image, rect_start_point, rect_end_point, color, thickness)

    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)

    picture_label.imgtk = imgtk
    picture_label.configure(image=imgtk)
    picture_label.after(1, detect_faces)

##
def recognize_faces():
    pass 

def load_image():
    pass 

def capture_frame_and_show():
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
    picture_label.after(1, capture_frame_and_show)

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


cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Create a cascade classifier object
face_cascade = cv.CascadeClassifier()

capture_frame_and_show()
window.mainloop()