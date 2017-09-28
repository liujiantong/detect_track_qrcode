#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
from scipy.spatial import distance
import imutils


red_range1 = (0, 30)
red_range2 = (150, 180)
green_range = (30, 90)
blue_range = (90, 140)


def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return np.abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1)*np.dot(d2, d2)))


def detect_square(cnt):
    peri = cv2.arcLength(cnt, True)
    cnt = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    if len(cnt) == 4 and cv2.isContourConvex(cnt):
        cnt = cnt.reshape(-1, 2)
        ws = [distance.euclidean(cnt[i], cnt[(i + 1) % 4]) for i in xrange(4)]
        max_cos = np.max([angle_cos(cnt[i], cnt[(i + 1) % 4], cnt[(i + 2) % 4]) for i in xrange(4)])
        z_val = np.std(ws) / np.mean(ws)
        if z_val < 0.05 and max_cos < 0.2:
            return True, cnt
    return False, None


def detect_color(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, (0., 20., 0.), (180., 255., 255.))
    white_mask = cv2.inRange(hsv, (0., 0., 200.), (180., 20., 255.))

    h, w = white_mask.shape[:2]
    cnz = cv2.countNonZero(white_mask)
    if np.float32(cnz) / (h * w) > 0.85:
        return 'white'

    hst = cv2.calcHist([hsv], [0], mask, [180], [0, 180])
    rval = np.sum(hst[red_range1[0]:red_range1[1]]) + np.sum(hst[red_range2[0]:red_range1[1]])
    gval = np.sum(hst[green_range[0]:green_range[1]])
    bval = np.sum(hst[blue_range[0]:blue_range[1]])

    if rval > 0.8:
        return 'red'
    elif gval > 0.8:
        return 'green'
    elif bval > 0.8:
        return 'blue'
    return 'unknown'


def detect_color_in(img, cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    cnt = cnt.reshape(-1, 2)
    square_pnts = np.float32([[0, 0], [100, 0], [100, 100]])
    mtx = cv2.getAffineTransform(np.float32(cnt[:3]), square_pnts)
    dst = cv2.warpAffine(img, mtx, (w, h))

    roi1, roi2, roi3, roi4 = dst[0:50, 0:50], dst[50:100, 0:50], dst[50:100, 50:100], dst[0:50, 50:100]
    colours = detect_color(roi1), detect_color(roi2), detect_color(roi3), detect_color(roi4)
    for idx, color in enumerate(colours):
        if color == 'white':
            colours = colours[idx:] + colours[:idx]
    return colours, dst


if __name__ == '__main__':
    image = cv2.imread('image/rotated_block.png')
    # image = cv2.imread('image/colorblock02.png')
    image = imutils.resize(image, height=600)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 3)
    colorful_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    edges = cv2.Canny(blurred, 100, 200)
    # edges = cv2.Canny(gray, 100, 200)
    _, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0]

    print 'hierarchy.len:', len(hierarchy)

    found = []
    # for h in hierarchy:
    #     nxt, prv, child, parent = h

    for cnt_idx in range(len(contours)):
        area = cv2.contourArea(contours[cnt_idx], oriented=True)
        # if area < 0
        if area < 400:
            continue

        k = cnt_idx
        c = 0
        while hierarchy[k][2] != -1:
            k = hierarchy[k][2]
            c = c + 1
        if c > 2:
            found.append(cnt_idx)

    print 'found.len:', len(found)

    for i, cnt_idx in enumerate(found):
        is_square, c = detect_square(contours[cnt_idx])
        if is_square:
            colors, dst = detect_color_in(image, c)
            cv2.imshow('rotated:%d' % i, dst)
            print 'color:%d:' % i, colors
            # cv2.drawContours(colorful_gray, [c], 0, (0, 255, 0), 2)

    # cv2.imshow('edge', edges)
    # cv2.imshow("colorful_gray", colorful_gray)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


