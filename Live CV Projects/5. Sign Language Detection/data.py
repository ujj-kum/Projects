from function import *
from time import sleep

base_directory = "MP_Data/"


if not os.path.exists(base_directory):
    os.makedirs(base_directory)

# Loop through each action and sequence and create directory to store data
for action in actions:
    for sequence in range(no_sequences):
            directory = ""
            directory = os.path.join(base_directory, action, str(sequence))
            os.makedirs(directory, exist_ok=True)

# Initialize Mediapipe Hands model for hand tracking
# Higher complexity, more accurace , more slow
#Detection Confidence: Used to find the object (hand) in the frame initially.
#Tracking Confidence: Used to follow the detected object across frames once it has been identified.
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    # Loop through each action
    for action in actions:

        # Loop through each sequence (video)
        for sequence in range(no_sequences):

            # Loop through each frame in the sequence
            for frame_num in range(sequence_length):
                frame = cv2.imread(f"Image/{action}/{sequence}.png")

                # Make detection using Mediapipe Hands model
                image, results = mediapipe_detection(frame, hands)

                # Draw landmarks on the image
                draw_styled_landmarks(image, results)

                # Display message indicating start of frame collection
                if frame_num==0:
                    # Set starting collection message on the image
                    cv2.putText(img=image, text=f"STARTING COLLECTION", \
                                org=(120, 200), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, \
                                color=(50, 50, 255), thickness=4, lineType=cv2.LINE_AA)

                    # Display the message with action and sequence info
                    cv2.putText(img=image, text=f"Collecting frames for {action}, Video Number {sequence}", \
                                org=(15, 12), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, \
                                color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)

                    # Show the image on screen
                    cv2.imshow(winname="OpenCV Feed", mat=image)
                    # cv2.resizeWindow("OpenCV Feed", new_width, new_height)

                    # Wait 200 ms before showing the message
                    cv2.waitKey(delay=200)
            
                else:
                    # Display the message with action and sequence info
                    cv2.putText(img=image, text=f"Collecting frames for {action}, Video Number {sequence}", \
                                org=(15, 12), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, \
                                color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
                    
                    cv2.imshow(winname="OpenCV Feed", mat=image)
                    # cv2.resizeWindow("OpenCV Feed", new_width, new_height)

                # Extract keypoints from the detection results
                # Coordinates of bounding boxes
                keypoints = extract_keypoints(results)

                # Define the path to save keypoints in a .npy file
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))

                # Save the keypoints as a .npy file
                np.save(npy_path, keypoints)

                if cv2.waitKey(10) & 0xFF==ord('q'):
                    break

cv2.destroyAllWindows()