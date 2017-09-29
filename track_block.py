#!/usr/bin/env python
# coding: utf-8


import cv2
import numpy as np
import shapely.geometry as geom

import shapedetector as sd


def detect_sq(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 3)

    edges = cv2.Canny(blurred, 60, 200)
    # edges = cv2.Canny(gray, 100, 200)
    _, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy is None:
        return

    hierarchy = hierarchy[0]

    found = []
    for cnt_idx in range(len(contours)):
        area = cv2.contourArea(contours[cnt_idx], oriented=True)
        if area < 400:
            continue

        k = cnt_idx
        c = 0
        while hierarchy[k][2] != -1:
            k = hierarchy[k][2]
            c = c + 1
        if c > 2:
            found.append(cnt_idx)

    for i, cnt_idx in enumerate(found):
        is_square, c = sd.detect_square(contours[cnt_idx])
        if is_square:
            print 'found squared'
            x, y, width, height = cv2.boundingRect(contours[cnt_idx])
            roi = image[y:y + height, x:x + width]
            cv2.imwrite("roi.png", roi)
            colors, dst = sd.detect_color_in(roi, c)
            print 'color:%d:' % i, colors


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
fgbg.setHistory(30)

# tracker = cv2.TrackerKCF_create()

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
        boxes.append(geom.box(x, y, (x + w), (y + h)))

    boxes = cluster_boxes(boxes)

    for box in boxes:
        x0, y0, x1, y1 = np.int32(box.bounds)
        # cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)

    detect_sq(frame)
    cv2.imshow('video_fg', frame)

    k = cv2.waitKey(30) & 0xff
    # wait for 's' key to save and exit
    if k == ord('s'):
        cv2.imwrite('frame.png', frame)
    elif k == 27 or k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

