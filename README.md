# pixelize-face-sort
Python code. Detect face infront of a camera. Pixelize it and sort it by the highest rgb value.

## Inspo
Pixelface https://github.com/janiswalser/Pixelface
Sort https://github.com/satyarth/pixelsort
Sort https://github.com/jeffThompson/PixelSorting

## For MA thesis

## How does it work?
In this code, we first load the cascade classifier for face detection from the OpenCV data directory. Then, we initialize a video capture object to read frames from the camera. We use a while loop to continuously read frames from the camera and process them.

For each frame, we first convert it to grayscale and then detect faces using the cascade classifier. We iterate over all detected faces and pixelize the face region by first resizing it to a small size using nearest neighbor interpolation, and then resizing it back to the original size using nearest neighbor interpolation again.

Next, we reshape the pixelized face region into a 2D array of pixels and sort the pixels by the highest RGB value using Python's sorted function and a lambda function that sums the RGB values of each pixel. We then reshape the sorted pixels back into a 3D array and replace the face region in the original frame with the sorted pixelized face region.

Finally, we display the resulting frame with the detected and sorted pixelized faces using OpenCV's imshow function. We also wait for a key press and exit the loop if 'esc' is pressed. Finally, we release the video capture object and destroy all windows.
