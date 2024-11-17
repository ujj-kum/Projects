from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
# import text-to-speech functionality
from win32com.client import Dispatch

# Function to speak text while capturing
def speak(msg):
    # Speech API
    speak = Dispatch(("SAPI.SpVoice"))
    speak.Speak(msg)

# Open a video capture
video = cv2.VideoCapture(index=0)

# Load the Harr Cascade Classifier for face detection
facedetect = cv2.CascadeClassifier(filename="Data/haarcascade_frontalface_default.xml")

# Load pre-trained face recognition data from pickle files
with open(file="Data/names.pkl", mode='rb') as w:
    LABELS = pickle.load(w)
with open(file="Data/faces_data.pkl", mode='rb') as f:
    FACES = pickle.load(f)

print(f"Shape of Faces matrix --> {FACES.shape}")

# Initilize KNN model
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model
knn.fit(X=FACES, y=LABELS)

# Define column names for attendance CSV file
COL_NAMES = ['NAME', 'TIME']

# Start an infinite loop for real time face recognition
while True:
    success, frame = video.read()

    # Convert the frame to Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = facedetect.detectMultiScale(image=gray, scaleFactor=1.3, minNeighbors=5)

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Crop the face region from the frame
        crop_img = frame[y:y+h, x:x+w]

        # Resize the crop image and flatten it to a single row
        # ML modl expects input in this format
        resized_img = cv2.resize(src=crop_img, dsize=(50, 50)).flatten().reshape(1, -1)

        # Predict the identity of the face
        output = knn.predict(X=resized_img)

        # Get current timestamp
        ts = time.time()
        date = datetime.fromtimestamp(timestamp=ts).strftime(format="%d-%m-%Y")
        timestamp = datetime.fromtimestamp(timestamp=ts).strftime(format="%H:%M:%S")

        # Check if attendance file for current data exists or not
        attendance_sheet = "Attendance_" + date + ".csv"
        file_exist = os.path.isfile(path=attendance_sheet)

        # Draw rectangles and text on the frame for visualization
        cv2.rectangle(img=frame, pt1=(x, y), pt2=(x+w, y+h), color=(0, 0, 255), thickness=1)
        cv2.rectangle(img=frame, pt1=(x, y), pt2=(x+w, y+h), color=(50, 50, 255), thickness=2)
        cv2.rectangle(img=frame, pt1=(x, y-40), pt2=(x+w, y), color=(50, 50, 255), thickness=-1)
        cv2.putText(img=frame, text=str(output[0]), org=(x, y-15), fontFace=cv2.FONT_HERSHEY_COMPLEX, \
                    fontScale=1, color=(255, 255, 255), thickness=1)
        cv2.rectangle(img=frame, pt1=(x, y), pt2=(x+w, y+h), color=(50, 50, 255), thickness=1)

        # Create an attendance record with predicted identity and timestamp
        attendance = [str(output[0]), str(timestamp)]

    # Display the current frame with annotations
    cv2.imshow(winname="Frame", mat=frame)

    # Wait for a key press
    k = cv2.waitKey(1)

    # If 'o' is pressed, announce attendance and save it to a CSV file
    if k == ord('o'):
        speak(msg="Attendance Taken...")
        time.sleep(5)
        # If file exists, append attendance to it
        if file_exist:
            with open(file=attendance_sheet, mode='+a') as csvwriter:
                writer = csv.writer(csvwriter)
                writer.writerow(attendance)
            csvwriter.close()
        # Else create the file and append attendance to it
        else:
            with open(file=attendance_sheet, mode='+a') as csvwriter:
                writer = csv.writer(csvwriter)
                writer.writerow(COL_NAMES)
                writer.writerow(attendance)
            csvwriter.close()

    if k == ord('q'):
        break

# Release the video capture object and close all windows
video.release()
cv2.destroyAllWindows()

