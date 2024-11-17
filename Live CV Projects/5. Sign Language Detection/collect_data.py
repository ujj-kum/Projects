import os
import cv2
import string

# Initialize video capture from default camera
cap = cv2.VideoCapture(index=0)

# Change to current directory
os.chdir(path=os.path.dirname(os.path.abspath(__file__)))
# Directory for saving captured images
base_directory = "Image/"


if not os.path.exists(base_directory):
    os.makedirs(base_directory)

for letter in string.ascii_uppercase:
    directory = ""
    directory = os.path.join(base_directory, letter)
    os.makedirs(directory, exist_ok=True)  # Create the directory if it doesn't already exist


while True:
    # Read frame from video capture
    success, frame = cap.read()

    count = {
    letter: len(os.listdir(base_directory + "/" + letter)) 
    for letter in string.ascii_uppercase
    }

    # Count no. of images in subdirectories A-Z
      
    # Obtain the dimension of the frame
    row = frame.shape[1] # 640
    col = frame.shape[0] # 480
    
    # Draw a rectangle at the top of the frame as a background
    cv2.rectangle(img=frame, pt1=(0, 40), pt2=(300, 400), color=(255, 255, 255), thickness=2)

    # Show the frame in a window named "data"
    cv2.imshow(winname="data", mat=frame)

    # Show the ROI in a seperate window
    cv2.imshow(winname="ROI", mat=frame[40:400, 0:300])

    # Crop the frame to the ROI
    frame = frame[40:400, 0:300]

    # Wait for a key press event
    interrupt = cv2.waitKey(10)
   
    if interrupt & 0xFF == ord('A'):
        cv2.imwrite(base_directory+'A/'+str(count['A'])+'.png',frame)
    if interrupt & 0xFF == ord('B'):
        cv2.imwrite(base_directory+'B/'+str(count['B'])+'.png',frame)
    if interrupt & 0xFF == ord('C'):
        cv2.imwrite(base_directory+'C/'+str(count['C'])+'.png',frame)
    if interrupt & 0xFF == ord('d'):
        cv2.imwrite(directory+'D/'+str(count['d'])+'.png',frame)
    if interrupt & 0xFF == ord('e'):
        cv2.imwrite(directory+'E/'+str(count['e'])+'.png',frame)
    if interrupt & 0xFF == ord('f'):
        cv2.imwrite(directory+'F/'+str(count['f'])+'.png',frame)
    if interrupt & 0xFF == ord('g'):
        cv2.imwrite(directory+'G/'+str(count['g'])+'.png',frame)
    if interrupt & 0xFF == ord('h'):
        cv2.imwrite(directory+'H/'+str(count['h'])+'.png',frame)
    if interrupt & 0xFF == ord('i'):
        cv2.imwrite(directory+'I/'+str(count['i'])+'.png',frame)
    if interrupt & 0xFF == ord('j'):
        cv2.imwrite(directory+'J/'+str(count['j'])+'.png',frame)
    if interrupt & 0xFF == ord('k'):
        cv2.imwrite(directory+'K/'+str(count['k'])+'.png',frame)
    if interrupt & 0xFF == ord('l'):
        cv2.imwrite(directory+'L/'+str(count['l'])+'.png',frame)
    if interrupt & 0xFF == ord('m'):
        cv2.imwrite(directory+'M/'+str(count['m'])+'.png',frame)
    if interrupt & 0xFF == ord('n'):
        cv2.imwrite(directory+'N/'+str(count['n'])+'.png',frame)
    if interrupt & 0xFF == ord('o'):
        cv2.imwrite(directory+'O/'+str(count['o'])+'.png',frame)
    if interrupt & 0xFF == ord('p'):
        cv2.imwrite(directory+'P/'+str(count['p'])+'.png',frame)
    if interrupt & 0xFF == ord('q'):
        cv2.imwrite(directory+'Q/'+str(count['q'])+'.png',frame)
    if interrupt & 0xFF == ord('r'):
        cv2.imwrite(directory+'R/'+str(count['r'])+'.png',frame)
    if interrupt & 0xFF == ord('s'):
        cv2.imwrite(directory+'S/'+str(count['s'])+'.png',frame)
    if interrupt & 0xFF == ord('t'):
        cv2.imwrite(directory+'T/'+str(count['t'])+'.png',frame)
    if interrupt & 0xFF == ord('u'):
        cv2.imwrite(directory+'U/'+str(count['u'])+'.png',frame)
    if interrupt & 0xFF == ord('v'):
        cv2.imwrite(directory+'V/'+str(count['v'])+'.png',frame)
    if interrupt & 0xFF == ord('w'):
        cv2.imwrite(directory+'W/'+str(count['w'])+'.png',frame)
    if interrupt & 0xFF == ord('x'):
        cv2.imwrite(directory+'X/'+str(count['x'])+'.png',frame)
    if interrupt & 0xFF == ord('y'):
        cv2.imwrite(directory+'Y/'+str(count['y'])+'.png',frame)
    if interrupt & 0xFF == ord('z'):
        cv2.imwrite(directory+'Z/'+str(count['z'])+'.png',frame)

cap.release()
cv2.destroyAllWindows()