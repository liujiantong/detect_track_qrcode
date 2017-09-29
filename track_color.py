#!/usr/bin/env python
# coding: utf-8


import cv2
# import numpy as np
import imutils

import shapedetector as sd


cap = cv2.VideoCapture(0)
wb = cv2.xphoto.createSimpleWB()

while True:
    _, frame = cap.read()

    image = wb.balanceWhite(frame)
    h, w = image.shape[:2]
    if h > 800:
        image = imutils.resize(image, height=600)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    colorful_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    founds = sd.find_contours(gray)

    print 'found.len:', len(founds)

    if len(founds) > 0:
        colors, bbox = sd.detect_color_from_contours(image, founds)
        if bbox is not None:
            cv2.drawContours(colorful_gray, [bbox], 0, (0, 0, 255), 2)
            print 'colors:', colors

    cv2.imshow('video_fg', colorful_gray)
    k = cv2.waitKey(30) & 0xff
    if k == 27 or k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

