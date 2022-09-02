Facial Detection and Facial Recognition

Facial Detection
Firstly, facial detection. This program will try and detect any faces through the live feed given by the webcam.
Facial detection was achieved through the use of using the opencv library.
To run this program, first make sure that opencv is installed on your device.
You can do this by running pip install opencv-python (since this is a python project).
Then, simply run "python facialDetection.py" in your LINUX terminal.
If you have multiple versions of python installed on your device, run it specifically using python3
by typing "python3 facialDetection.py" in your LINUX terminal.
This program was completed with the help of the opencv documentation as well as the YouTube tutorial
"Face Detection in 2 Minutes using OpenCV and Python" by Adarsh Menon.
In addition, the backworkings of the program were analyzed from things I learned in this class as well as my Image Processing Class
to better understand how cascading works as well as object recognition.


Facial Recognition
After achieving the base step of detecting faces, we can now start comparisons to test data sets/images
to achieve facial recognition to the point where the program can accurately tell who is in the live
camera footage. To do so, we use the opencv library again as well as the face_recognition library.
Similar to the last program, we will need the opencv library installed, following the instructions above.
We will also need the face library downloaded on our systems. We can also download this
using pip through the command pip install face_recognition.
We feed the data set/images to the program through the imgStandards folder. 
To run the program, do the same as the facial detection program:
Run "python facialRecognition.py" in the LINUX terminal on your device.
If there are multiple versions of python installed on your system, run it specifically using python 3
by typing "python3 facialRecognition.py" in the command line.
This program was completed with the help of the OpenCV documentation and face_recognition documentation as well as the
YouTube tutorial "Face recognition in real-time | with Opencv and Python" by PySource and their documentation on the topic.

