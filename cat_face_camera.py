import cv2
import numpy as np
import matplotlib.pyplot as plt
cascade_file = "haarcascade_frontalface_alt.xml"
cascade = cv2.CascadeClassifier(cascade_file)

import os
assert os.path.isfile(cascade_file), 'haarcascade_frontalface_default.xml がない'
def cat_face(img,rect):
    (x1,y1,x2,y2) = rect

    w = x2 - x1
    h = y2 - y1

    cat = cv2.imread("img/cat_face.jpg")
    cat = cv2.resize(cat,(w,h))

    img2 = img.copy()
    img2[y1:y2, x1:x2] = cat
    return img2 

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face_list = cascade.detectMultiScale(frame_gray, 1.5, 3)
    for (x,y,w,h) in face_list:
        frame = cat_face(frame,(x,y,x+w,y+h))
    cv2.imshow('OpenCV web Camera',frame)
    k = cv2.waitKey(1)
    if k == 27 or k == 13:break

cap.release()
cv2.destroyAllWindows()
