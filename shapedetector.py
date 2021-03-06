#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
from scipy.spatial import distance

import imutils

from polygon import check_cnt_contain


red_range1 = (0, 30)
red_range2 = (150, 180)
green_range = (30, 90)
blue_range = (90, 140)

block_size = 100
half_block_size = (block_size / 2)


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
        # print 'z_val:%s, max_cos:%s' % (z_val, max_cos)
        if z_val < 0.18 and max_cos < 0.35:
            return True, cnt
    return False, None


def detect_color(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, (0., 20., 0.), (180., 255., 255.))
    white_mask = cv2.inRange(hsv, (0., 0., 200.), (180., 60., 255.))

    h, w = white_mask.shape[:2]
    cnz = cv2.countNonZero(white_mask)
    wr = np.float32(cnz) / (h * w)
    # print (h, w), 'cnz:', cnz, 'white rate:', wr
    if wr > 0.80:
        return 'white'

    hst = cv2.calcHist([hsv], [0], mask, [180], [0, 180])
    cv2.normalize(hst, hst, 0, 1, cv2.NORM_MINMAX)

    rval = np.sum(hst[red_range1[0]:red_range1[1]]) + np.sum(hst[red_range2[0]:red_range1[1]])
    gval = np.sum(hst[green_range[0]:green_range[1]])
    bval = np.sum(hst[blue_range[0]:blue_range[1]])
    # print 'color values:', rval, gval, bval

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
    square_pnts = np.float32([[0, 0], [block_size, 0], [block_size, block_size]])
    mtx = cv2.getAffineTransform(np.float32(cnt[:3]), square_pnts)
    dst = cv2.warpAffine(img, mtx, (w, h))

    roi1, roi2, roi3, roi4 = dst[0:half_block_size, 0:half_block_size], \
                             dst[half_block_size:block_size, 0:half_block_size], \
                             dst[half_block_size:block_size, half_block_size:block_size], \
                             dst[0:half_block_size, half_block_size:block_size]
    colours = detect_color(roi1), detect_color(roi2), detect_color(roi3), detect_color(roi4)
    for idx, color in enumerate(colours):
        if color == 'white':
            colours = colours[idx:] + colours[:idx]
    return colours, dst


def find_contours(gray):
    # global contours, found, cnt_idx, c
    blurred = cv2.medianBlur(gray, 5)
    edges = cv2.Canny(blurred, 100, 120)

    found_cnts = []
    _, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None or hierarchy.size == 0:
        return found_cnts

    hierarchy = hierarchy[0]
    # print 'hierarchy.len:', len(hierarchy)
    # print 'hierarchy:', hierarchy

    for cnt_idx in range(len(contours)):
        area = cv2.contourArea(contours[cnt_idx], oriented=True)
        # if area < 0
        if area < 100:
            continue

        k = cnt_idx
        c = 0
        while hierarchy[k][2] != -1:
            k = hierarchy[k][2]
            # area1 = cv2.contourArea(contours[k])
            # if area1 < 36:
            #     continue
            c = c + 1
        if c > 0:
            found_cnts.append(contours[cnt_idx])

    return found_cnts


def detect_color_from_contours(img, cnts):
    # global i, bbox, colors, dst
    square_cnts = []
    for i, cnt0 in enumerate(cnts):
        is_square, c = detect_square(cnt0)
        if is_square:
            square_cnts.append(c)
            # cv2.drawContours(colorful_gray, [c], 0, (0, 255, 0), 2)
            # cv2.imshow('rotated:%d' % i, dst)
            # print 'color:%d:' % i, colors

    if not square_cnts:
        return [], None

    bound_cnt = check_cnt_contain(square_cnts)
    colors, dst = detect_color_in(img, bound_cnt)
    return colors, bound_cnt


if __name__ == '__main__':
    # image = cv2.imread('roi_test.png')
    image = cv2.imread('image/pic01.jpg')
    # image = cv2.imread('image/pic02.jpg')
    # image = cv2.imread('image/pic03.jpg')
    # image = cv2.imread('image/colorblock02.png')
    # image = cv2.imread('kalman_frame.png')

    wb = cv2.xphoto.createSimpleWB()
    image = wb.balanceWhite(image)
    h, w = image.shape[:2]
    if h > 800:
        image = imutils.resize(image, height=600)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    colorful_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    founds = find_contours(gray)

    print 'found.len:', len(founds)

    if len(founds) > 0:
        colors, cnt = detect_color_from_contours(image, founds)
        cv2.drawContours(colorful_gray, [cnt], 0, (0, 0, 255), 2)

        print 'colors:', colors
        # cv2.imshow('edge', edges)
        cv2.imshow("colorful_gray", colorful_gray)
        cv2.waitKey(0)

        cv2.destroyAllWindows()
