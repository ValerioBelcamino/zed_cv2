import cv2
import numpy as np
import os

# Defining the dimensions of checkerboard
CHECKERBOARD = (7,4)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

cap = cv2.VideoCapture(0)

#resolution stuff
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(width, height)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(width, height)

image_id = 0
saving_path = "./images_logitech2/"
while True:

    ret, frame = cap.read()
    frame_resize = cv2.resize(frame, (int(width/2), int(height/2)), interpolation=cv2.INTER_CUBIC)
    cv2.imshow('frame', frame_resize)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        print(print(saving_path + f'image_{image_id}.jpg'))
        cv2.imwrite(saving_path + f'image_{image_id}.jpg', frame)
        image_id += 1
    
    elif key == ord('q'):
        break
    
    