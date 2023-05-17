import random
import cv2
import numpy as np
import time
from gtts import gTTS
import os
import threading
import pyautogui

# Initialize variables for storing the original frame and the time when the face was detected
original_frame = None
first_detection_time = None
last_detection_time = None

# Specify the buffer time before starting the pixelization process
buffer_time = 2

# Initialize variables for storing the pixels and the pixel count
pixels = []
pixel_count = 0

# Initialize the face cascade classifier and the video capture object
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)


def play_audio(pixel_count):
    # Adjust the file name format if needed
    audio_file = f'number_{pixel_count}.mp3'
    # Replace 'path/to/' with the actual file path on your PC
    os.system(f'mpg123 {audio_file}')


def destroy_window():
    cv2.destroyWindow('Art Project')


# Set the dimensions of the final image
width = 3
height = 3

# Adjust width and height to match the screen resolution aspect ratio
screen_width = 1920
screen_height = 1080
aspect_ratio = screen_width / screen_height
if aspect_ratio < 1:
    width = int(height * aspect_ratio)
else:
    height = int(width / aspect_ratio)

# Initialize the result image
result = np.zeros((height, width, 3), dtype=np.uint8)


def add_pixel(pixel, pixel_count, width, height):
    global result

    # Calculate the position of the new pixel based on the pixel count
    x = pixel_count % width
    y = pixel_count // width

    def update_pixel_color():
        result[y:y + 1, x:x + 1] = pixel

    if pixel_count >= width * height:
        # Calculate the position of the pixel to be replaced based on the pixel count
        replace_x = (pixel_count - width * height) % width
        replace_y = (pixel_count - width * height) // width

        # Replace the pixel at the calculated position with white
        result[replace_y:replace_y + 1,
               replace_x:replace_x + 1] = [255, 255, 255]

        # Update the x and y coordinates for the new pixel
        x = replace_x
        y = replace_y

    # Add the pixel as white initially
    result[y:y + 1, x:x + 1] = [255, 255, 255]

    # Schedule a timer to update the pixel color after a certain amount of time has passed
    threading.Timer(2, update_pixel_color).start()

    return pixel_count + 1


while True:
    # Simulate a key press every 30 seconds to prevent the computer from going to sleep or dimming its screen
    if int(time.time()) % 30 == 0:
        pyautogui.press('shift')
    if int(time.time()) % (30*60) == 0:
        time.sleep(3)
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=5, minSize=(80, 80))

    if len(faces) > 0:
        # Store the original frame and the time when the face was first detected
        if original_frame is None:
            # Use the first detected face
            (x, y, w, h) = faces[0]
            original_frame = frame.copy()  # make a copy to keep the original frame unchanged
            first_detection_time = time.time()
            last_detection_time = time.time()

        elif frame is not original_frame:
            # Update the original frame when a new face is detected
            original_frame = frame.copy()  # make a copy to keep the original frame unchanged
            last_detection_time = time.time()

        # Draw the green rectangle around the face region on the original frame
        (x, y, w, h) = faces[0]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # If it has been more than buffer_time since the first face was detected, display the pixelized face
        if last_detection_time - first_detection_time >= buffer_time:

            (x, y, w, h) = faces[0]
            face = frame[y:y+h, x:x+w]

            # Add audio output when a face is detected
            os.system('mpg123 detected.mp3')
            avg_color = np.mean(face, axis=(0, 1)).astype(int)
            pixel = np.zeros((1, 1, 3), dtype=np.uint8)
            pixel[0, 0] = avg_color

            add_pixel(pixel, pixel_count, width, height)
            pixel_count += 1

            result_scaled = cv2.resize(
                result, (1920, 1080), interpolation=cv2.INTER_NEAREST)

            audio_thread = threading.Thread(
                target=play_audio, args=(pixel_count,))
            audio_thread.start()

            cv2.namedWindow('Art Project', cv2.WINDOW_FULLSCREEN)
            cv2.setWindowProperty(
                'Art Project', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow('Art Project', result_scaled)
            cv2.waitKey(5000)  # wait for 3 seconds

            audio_thread.join()
            first_detection_time = time.time() + 2

        else:
            cv2.imshow('Environment', frame)
            last_detection_time = time.time()

    else:
        cv2.imshow('Environment', frame)
        original_frame = None
        last_detection_time = None
        last_pixelized = None

    # Exit the program if the user presses the 'Esc' key
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
