#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np


def get_frame_size(fw, fh, max_width=1024):
    if fw < max_width:
        return np.int32(fw), np.int32(fh)
    ratio = max_width / float(fw)
    return np.int32(max_width), np.int32(ratio * fh)


def union_rects(rects):
    if not rects:
        return None
    pnts = []
    for x, y, w, h in rects:
        pnts.append((x, y))
        pnts.append((x + w, y + h))
    return cv2.boundingRect(np.array(pnts))


def center(points):
    cx = (points[0][0] + points[1][0] + points[2][0] + points[3][0]) / 4.0
    cy = (points[0][1] + points[1][1] + points[2][1] + points[3][1]) / 4.0
    return np.array([np.float32(cx), np.float32(cy)], np.float32)


def contour_center(contour):
    M = cv2.moments(contour)
    return np.array([(M["m10"] / M["m00"]), (M["m01"] / M["m00"])], np.float32)


def calc_distance(pt1, pt2):
    """
    Calculate the distance between two points
    :param pt1: tuple , 2D point
    :param pt2:  tuple, 2D point
    :return: distance between given points
    """
    x1, y1 = pt1
    x2, y2 = pt2
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def angle_cos(p0, p1, p2):
    d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
    return np.abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))

