import cv2

# Use cascades to scan the screen and match pixels to a face
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Use the webcame through opencv
webcam = cv2.VideoCapture(0)

# Set up an infinite loop to continue to detect faces in real time
while True:
    ret, frame = webcam.read()

    # convert to greyscale
    grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # Detect facial features within a converted greyscale image
    faces = face_cascade.detectMultiScale(grey, 1.1, 4)

    # Actually tell the user where the face is within the screen by drawing a rectangle
    for(x1,y1,x2,y2) in faces:
        cv2.rectangle(frame, (x1, y1), (x1+x2, y1+y2), (150, 15, 150), 4)

        cv2.imshow('Facial Detection Capture', frame)
    
    # Wait for the escape key to be pressed and when it is, close the window and finish the program.
    escapeKey = cv2.waitKey(1)
    if escapeKey == 27:
        break

webcam.release()