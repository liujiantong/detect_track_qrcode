#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import imutils


class ShapeDetector:
    def __init__(self):
        pass

    @staticmethod
    def detect(c):
        # initialize the shape name and approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.06 * peri, True)

        if len(approx) == 3:
            shape = "triangle"
        elif len(approx) == 4:
            # compute the bounding box of the contour and use the
            # bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)

            # a square will have an aspect ratio that is approximately
            # equal to one, otherwise, the shape is a rectangle
            shape = "square" if 0.95 <= ar <= 1.05 else "rectangle"
        elif len(approx) == 5:
            shape = "pentagon"
        else:
            shape = "unidentified"

        # return the name of the shape
        return shape


def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return np.abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1)*np.dot(d2, d2)))


def detect_square(cnt):
    peri = cv2.arcLength(cnt, True)
    cnt = cv2.approxPolyDP(cnt, 0.04 * peri, True)
    if len(cnt) == 4 and cv2.isContourConvex(cnt) and cv2.contourArea(cnt) > 400:
        cnt = cnt.reshape(-1, 2)
        max_cos = np.max([angle_cos(cnt[i], cnt[(i + 1) % 4], cnt[(i + 2) % 4]) for i in xrange(4)])
        if max_cos < 0.15:
            return True, cnt
    return False, None


def find_squares(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    squares = []
    for gray in cv2.split(img):
        for thrs in xrange(0, 255, 26):
            if thrs == 0:
                bin = cv2.Canny(gray, 0, 50, apertureSize=5)
                bin = cv2.dilate(bin, None)
            else:
                retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
            bin, contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
                if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
                    if max_cos < 0.1:
                        squares.append(cnt)
    return squares


def draw_contours(cnts, color):
    for i in cnts:
        cnt = contours[i]
        # compute the center of the contour
        # M = cv2.moments(cnt)
        # cX = int(M["m10"] / M["m00"])
        # cY = int(M["m01"] / M["m00"])

        shape = sd.detect(cnt)
        if shape != 'square':
            continue

        cnt = cnt.astype("int")
        cv2.drawContours(colorful_gray, [cnt], -1, color, 2)
        # cv2.putText(colorful_gray, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


if __name__ == '__main__':
    # image = cv2.imread('block.png')
    image = cv2.imread('colorblock02.png')
    image = imutils.resize(image, width=1000)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 3)
    colorful_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    edges = cv2.Canny(blurred, 100, 200)
    _, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0]

    print 'hierarchy.len:', len(hierarchy)

    found = []
    # found_dict = defaultdict(list)
    # for h in hierarchy:
    #     nxt, prv, child, parent = h

    # explored = set()
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area < 200:
            continue

        k = i
        c = 0
        while hierarchy[k][2] != -1:
            k = hierarchy[k][2]
            c = c + 1
        # if c > 2:
        if c > 3:
            found.append(i)

    print 'found.len:', len(found)

    # sd = ShapeDetector()
    # for i in found:
    #     if 'square' == sd.detect(contours[i]):
    #         cv2.drawContours(colorful_gray, contours, i, (0, 255, 0), 2)

    for i in found:
        flag, c = detect_square(contours[i])
        if flag:
            cv2.drawContours(colorful_gray, [c], 0, (0, 255, 0), 2)

    # colors = [
    #     (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)
    # ]
    # for h in xrange(0, 4):
    #     draw_contours(found_dict[h], colors[h])
    # draw_contours(found_dict[0], colors[0])

    cv2.imshow('edge', edges)
    cv2.imshow("colorful_gray", colorful_gray)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


