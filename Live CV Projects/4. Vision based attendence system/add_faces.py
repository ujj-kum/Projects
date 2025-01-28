import cv2
import pickle
import numpy as np
import os
import time
from datetime import datetime

# Load Haar Cascade Classifier for Face Detection
facedetect = cv2.CascadeClassifier(filename="Data/haarcascade_frontalface_default.xml")

# List for storing face data
faces_data = []

# Counter to keep track of no. of frames processed
i = 0

# Input student's name
name = input("Enter your name : ").strip()
# name = "uk"
# Open video camera using default webcam
video = cv2.VideoCapture(index=0)

# Loop to capture video frames and detect faces
while True:
    # Capture a frame from video
    # success is a boolean variable
    # frame contains actual dataset
    success, frame = video.read()
    
    # Convert the frame to Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = facedetect.detectMultiScale(image=gray, scaleFactor=1.3, minNeighbors=5)

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Crop face region from frame
        crop_img = frame[y:y+h, x:x+w]
        # Resize the cropped face image to 50x50 px
        resized_image = cv2.resize(src=crop_img, dsize=(50, 50))

        # Append resized face image to faces_data every 5th frame
        # Appenf 5 images, for better accuracy, each taken after 5 frames
        if len(faces_data)<=5 and (i%5==0):
            faces_data.append(resized_image)
        
        i+=1

        # Display count of captures faces on the frame
        # (50x50) = Size of the frame
        # 1 = No. of Channels in a grayscale image
        # (50, 50, 255) = (size, size, no. of pixels)
        cv2.putText(img=frame, text=str(len(faces_data)), org=(50, 50), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, \
                    color=(50, 50, 255))

        # Draw a rectangle around detected face
        # Color is in BGR format
        cv2.rectangle(img=frame, pt1=(x, y), pt2=(x+w, y+h), color=(50, 50, 255), thickness=1)

    # Display the current frame with annotations
    cv2.imshow(winname="Frame", mat=frame)

    # Wait for a key press or until 5 faces are captures
    # The program will wait for 1 millisecond for a key press. 
    # After this brief delay, the program proceeds to the next frame
    k = cv2.waitKey(10)
    if k==ord('q') or len(faces_data)==5:
        break

# Release the video capture object or until 5 faces are captured
video.release()
cv2.destroyAllWindows()


# Convert the list of faces images to Numpy array
faces_data = np.asarray(a=faces_data)
faces_data = faces_data.reshape(5, -1)

# Check if names.pkl is present in 'Data' directory
if 'names.pkl' not in os.listdir(path='Data/'):
    names = [name] * 5
    with open(file='Data/names.pkl', mode='wb') as f:
        pickle.dump(obj=names, file=f)
else:
    with open(file='Data/names.pkl', mode='rb') as f:
        names = pickle.load(file=f)
    
    names = names + [name]*5

    with open(file='Data/names.pkl', mode='wb') as f:
        pickle.dump(obj=names, file=f)

# Creating pickle object for face data
if 'faces_data.pkl' not in os.listdir(path="Data/"):
    with open(file="Data/faces_data.pkl", mode="wb") as f:
        pickle.dump(obj=faces_data, file=f)
else:
    with open(file="Data/faces_data.pkl", mode="rb") as f:
        faces = pickle.load(file=f)

        # Append the new array 'faces_data' to existing array, vertically, as new row
        faces = np.append(arr=faces, values=faces_data, axis=0)

    with open(file="Data/faces_data.pkl", mode="wb") as f:
        pickle.dump(obj=faces, file=f)