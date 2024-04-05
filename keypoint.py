import cv2
import numpy as np
import imutils
import dlib
import pandas as pd
from imutils import face_utils

# Initialize dlib's face detector (HOG-based) and the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Load the input image, resize it, and convert it to grayscale
def get_keypoint(img_name):
    image = cv2.imread(img_name)
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    rects = detector(gray, 1)

    # Initialize an empty list to store all facial landmarks
    all_landmarks = []

    # Loop over the face detections
    for rect in rects:
        # Determine the facial landmarks for the face region
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        all_landmarks.extend(shape)

    # Convert the list of landmarks into a NumPy array
    landmarks_array = np.array(all_landmarks)

    # Create a DataFrame from the landmarks array
    df = pd.DataFrame(landmarks_array, columns=['x', 'y'])
    return df