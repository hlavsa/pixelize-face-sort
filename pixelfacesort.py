import cv2
import numpy as np
import time

# Load the cascade for face detection
face_cascade = cv2.CascadeClassifier(
    "data/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("data/haarcascade_eye.xml")


# Initialize the video capture object
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5)

    # Iterate over all detected faces
    for (x, y, w, h) in faces:

        # Pixelize the face region
        roi = frame[y:y+h, x:x+w]
        pixelized = cv2.resize(roi, (10, 10), interpolation=cv2.INTER_NEAREST)
        pixelized = cv2.resize(
            pixelized, (w, h), interpolation=cv2.INTER_NEAREST)

        # Sort the pixels by the highest RGB value
        pixels = np.reshape(pixelized, (w * h, 3))
        pixels = sorted(pixels, key=lambda x: sum(x), reverse=True)
        pixelized = np.reshape(pixels, (h, w, 3))

        # Replace the face region in the original frame with the pixelized face region
        frame[y:y+h, x:x+w] = pixelized

    # Display the frame with the detected faces
    cv2.imshow('Face Pixelizer', frame)

    # the way out
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break


# Release the video capture object and destroy all windows
cap.release()
cv2.destroyAllWindows()
