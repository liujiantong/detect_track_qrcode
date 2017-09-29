#!/usr/bin/env python
# coding: utf-8

# import numpy as np
import cv2
from shapely.geometry import box


def cluster_boxes(box_list):
    if len(box_list) < 2:
        return box_list

    box_list = sorted(box_list, key=lambda b: b.area, reverse=True)
    results = []
    b0 = box_list[0]
    results.append(b0)

    for b in box_list[1:]:
        if b0.intersects(b) or b0.touches(b):
            b0 = b0.union(b)
        else:
            results.append(b)

    return results


cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()

history = 15
for i in xrange(history):
    fgbg.setHistory(history)


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

track_init = False

while True:
    _, frame = cap.read()
    fg_mask = fgbg.apply(frame)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    _, fg_mask = cv2.threshold(fg_mask, 60, 255, cv2.THRESH_BINARY)
    im2, contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 200:
            continue
        x, y, w, h = cv2.boundingRect(c)
        boxes.append(box(x, y, (x + w), (y + h)))

    boxes = cluster_boxes(boxes)

    for b in boxes:
        x0, y0, x1, y1 = b.bounds
        cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)

    cv2.imshow('video_fg', frame)

    k = cv2.waitKey(30) & 0xff
    # wait for 's' key to save and exit
    if k == ord('s'):
        cv2.imwrite('frame.png', frame)
    elif k == 27 or k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

