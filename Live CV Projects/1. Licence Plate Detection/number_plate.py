import cv2

harcascade = "D:\WORK STUDY\Online_Courses\PW\Live CV Projects\model\haarcascade_russian_plate_number.xml"

# Capture Video through webcam
cap = cv2.VideoCapture(index=0)

# 3: Frame width and 4: Frame height
# Captures screen of 640 x 480 pixels
cap.set(propId=3, value=640)
cap.set(propId=4, value=480)

# Min area for a detected region to be considered as a license plate
min_area = 500

count = 0

while True:
    # Reading frames from webcam
    success, img = cap.read()

    # Creating a licence plate classifier
    plate_cascade = cv2.CascadeClassifier(harcascade)

    # Converting frame to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Image size is rescaled for multi-scale detection
    # Min_neighbors is used to filter out the false +ve cases
    plates = plate_cascade.detectMultiScale(image=img_gray, scaleFactor=1.1, minNeighbors=4)
    
    
    # iterating through the detected plates
    # x, y are top left coordinates
    for (x, y, w, h) in plates:
        
        area = w*h
        # If area is greater than the min. area, draw a rectangle
        # around the plate and display the text "LICENSE_PLATE" on top corner of the rectangle
        # Also display REGION OF INTEREST(ROI) of licence plate
        if area > min_area:
            cv2.rectangle(image=img, start_point=(x, y), end_point=(x+w, y+h), color=(0, 255, 0), thickness=2)
            cv2.putText(image=img, text='License Plate', org=(x, y-5), font=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(255, 0, 255), thickness=2)

            # Display ROI of license plate
            img_roi = img[y:y+h, x:x+w]
            cv2.imshow(winname='ROI', mat=img_roi)
        
    cv2.imshow(winname="Result", mat=img)
        
    # Saving the plate when 'S' is pressed
    # Delay=1, Program will wait for 1 millisecond
    if cv2.waitKey(delay=1) & 0xFF==ord('S'):
        cv2.imwrite(filename="plates/scanned_img_"+str(count)+".jpg", img=img_roi)
        cv2.rectangle(image=img, start_point=(0, 200), end_point=(640, 300), color=(0, 255, 0), thickness=2)
        cv2.putText(image=img, text='Plate Saved', org=(150, 265), font=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1, color=(0, 0, 255), thickness=2)
        cv2.imshow(winname="Results", mat=img)
        cv2.waitKey(delay=500)
        count+=1




    