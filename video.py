#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cv2


def cluster_boxes1(box_list):
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


def union_rect(r1, r2):
    x = min(r1[0], r2[0])
    y = min(r1[1], r2[1])
    w = max(r1[0] + r1[2], r2[0] + r2[2]) - x
    h = max(r1[1] + r1[3], r2[1] + r2[3]) - y
    return x, y, w, h


def group_rects(box_list):
    if len(box_list) < 2:
        return box_list

    box_list = sorted(box_list, key=lambda b: b[2]*b[3], reverse=True)
    x0, y0, w0, h0 = box_list[0]
    cx0, cy0 = (x0 + 0.5 * w0), (y0 + 0.5 * h0)

    results = []
    b0 = (x0, y0, w0, h0)
    results.append(b0)

    for (x1, y1, w1, h1) in box_list[1:]:
        w, h = (w0 + w1) * 0.5, (h0 + h1) * 0.5
        cx1, cy1 = (x1 + 0.5 * w1), (y1 + 0.5 * h1)
        dx = np.absolute(np.absolute(cx0 - cx1) - w)
        dy = np.absolute(np.absolute(cy0 - cy1) - h)
        wt, ht = np.max([w0, w1]) * 0.3, np.max([h0, h1]) * 0.3
        if dx < wt and dy < ht:
            b0 = union_rect(b0, (x1, y1, w1, h1))
        else:
            results.append((x1, y1, w1, h1))
    return results


def union_rects(rects):
    if not rects:
        return None

    pnts = []
    for x, y, w, h in rects:
        pnts.append((x, y))
        pnts.append((x + w, y + h))
    return cv2.boundingRect(np.array(pnts))


cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()
fgbg.setHistory(30)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# track_init = False

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
        if area < 20:
            continue
        boxes.append(cv2.boundingRect(c))
        # x, y, w, h = cv2.boundingRect(c)
        # boxes.append(geom.box(x, y, (x + w), (y + h)))
    # boxes = cluster_boxes(boxes)

    for b in boxes:
        x0, y0, w, h = b
        cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (0, 0, 255), 2)

    urect = union_rects(boxes)
    # grp_boxes = group_rects(boxes)
    # print 'boxes:', boxes, 'weights:', weights

    if urect:
        x0, y0, w, h = urect
        cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (0, 255, 0), 2)

    cv2.imshow('video_fg', frame)

    k = cv2.waitKey(10) & 0xff
    # wait for 's' key to save and exit
    if k == ord('s'):
        cv2.imwrite('frame.png', frame)
    elif k == 27 or k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
