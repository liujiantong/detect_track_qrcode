#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np


roi = cv2.imread('block01.png')
hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

target = cv2.imread('colorblock02.png')
hsvt = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)

# calculating object histogram
mask = cv2.inRange(hsv, np.array((0., 20., 20.)), np.array((180., 255., 255.)))
# roihist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
roihist = cv2.calcHist([hsv], [0], mask, [180], [0, 180])

# normalize histogram and apply backprojection
cv2.normalize(roihist, roihist, 0, 255, cv2.NORM_MINMAX)
dst = cv2.calcBackProject([hsvt], [0], roihist, [0, 180], 1)

# Now convolute with circular disc
disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
cv2.filter2D(dst, -1, disc, dst)

# threshold and binary AND
ret, thresh = cv2.threshold(dst, 50, 255, 0)
thresh = cv2.merge((thresh, thresh, thresh))
res = cv2.bitwise_and(target, thresh)
res = np.vstack((target, thresh, res))
cv2.imwrite('proj_res.jpg', res)

