import os
import cv2
import time


# Set the path to the directory containing the captured images from the camera
images_path = os.path.join('workspace', 'capturedimages')


# create the directory if it doesn't exist
if not os.path.exists(images_path):
    os.makedirs(images_path)


# Set the camera to capture 9 images of presented objects
for i in range(10):
    capture = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not capture.isOpened():
        raise IOError("Cannot open webcam")

    # Capture the image
    ret, frame = capture.read()

    # Display the resulting frame
    imagename = os.path.join(images_path, 'image' + str(i) + '.jpg')

    # Save the image to the directory
    cv2.imwrite(os.path.join(images_path, 'image_{}.jpg'.format(i)), frame)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Wait for 5 second
    time.sleep(5)

    # When everything done, release the capture
    capture.release()
    cv2.destroyAllWindows()