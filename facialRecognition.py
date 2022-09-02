import cv2
from simple_facerec import SimpleFacerec

# Load faces from the image folder provided (Can be filled with whatever images of any faces you want to comapre to)
facialRec = SimpleFacerec()
facialRec.loadImages("imgStandards/")

# Tell opencv to use the webcam
webcam = cv2.VideoCapture(0)

# Create an infinite loop to constantly gather data from the live video from the webcam
while True:
    frameBool, frame = webcam.read()

    # Facial Detection within simple_facerec.py file
    faceLocs, names = facialRec.facialDetection(frame)

    # Display the rectangle frame and assigned name to each face detected within the frame.
    for location, name in zip(faceLocs, names):
        y1, x2, y2, x1 = location[0], location[1], location[2], location[3]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 4)
        cv2.putText(frame, name,(x1, y1 - 10), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 200), 2)
        

    # Display the actual window with the live footage and recognition.
    cv2.imshow("Facial Recognition Capture", frame)

    # Close the window when the escape key is pressed.
    escape = cv2.waitKey(1)
    if escape == 27:
        break
# Release webcam capture and close windows.
webcam.release()
cv2.destroyAllWindows()
