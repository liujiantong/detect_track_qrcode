#!/usr/bin/env python
# coding: utf-8


import numpy as np
import cv2
import imutils


def frame_size(w, h, max_width=800):
    if w < max_width:
        return w, h
    ratio = max_width / float(w)
    return 800, np.int32(ratio * h)


cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)
width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
size = frame_size(width, height, 800)
# size = (np.int32(width), np.int32(height))

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output.avi', fourcc, fps, size)

while cap.isOpened():
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=800)

    # write the flipped frame
    out.write(frame)
    cv2.imshow('frame', frame)

    k = cv2.waitKey(3) & 0xFF
    if k == 27 or k == ord('q'):
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()

