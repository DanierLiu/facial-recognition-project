import face_recognition
import cv2
import os
import glob
import numpy as np

class SimpleFacerec:
    def __init__(self):
        self.faceComparisons = []
        self.names = []

        # Resize the frame so that we can scan the frame quicker.
        self.frame_resizing = 0.25

    def loadImages(self, imagesFolder):
        """
        Load images from path
        :param imagesFolder:
        :return:
        """
        # Load Images from the file
        imagesFolder = glob.glob(os.path.join(imagesFolder, "*.*"))

        print("{} images found.".format(len(imagesFolder)))

        # Store the encoded images and their respective labels
        for path in imagesFolder:
            image = cv2.imread(path)
            rgbImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Get the base file name
            basename = os.path.basename(path)
            (filename, ext) = os.path.splitext(basename)
            # Get encoding
            encodedImage = face_recognition.face_encodings(rgbImage)[0]

            # Store both the file name and the encoded images
            self.faceComparisons.append(encodedImage)
            self.names.append(filename)
        print("Comparison images loaded")

    def facialDetection(self, frame):
        # Resize the frame
        resizedCapture = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        # Detect all faces
        # Convert the image from BGR to RGB color as we move from opencv to face_recognition
        rgbCapture = cv2.cvtColor(resizedCapture, cv2.COLOR_BGR2RGB)
        facialPositions = face_recognition.face_locations(rgbCapture)
        face_encodings = face_recognition.face_encodings(rgbCapture, facialPositions)

        labels = []
        for face_encoding in face_encodings:
            # Compare the face with the images provided and see if it matches up
            matches = face_recognition.compare_faces(self.faceComparisons, face_encoding)
            name = "Unknown"

            # Choose the best match for the face out of the provided image options
            face_distances = face_recognition.face_distance(self.faceComparisons, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.names[best_match_index]
            labels.append(name)

        # Convert to numpy array to adjust coordinates taking heed of frame resizing
        facialPositions = np.array(facialPositions)
        facialPositions = facialPositions / self.frame_resizing
        return facialPositions.astype(int), labels
