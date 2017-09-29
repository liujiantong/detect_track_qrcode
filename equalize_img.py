#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np


def hist_equal_color(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    # print len(channels)
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img


img = cv2.imread('roi_test.png')

wb = cv2.xphoto.createSimpleWB()
img_output = wb.balanceWhite(img)


# img_output = hist_equal_color(img)

# ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
#
# # equalize the histogram of the Y channel
# ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
#
# # convert the YUV image back to RGB format
# img_output = cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR)

cv2.imshow('Color input image', img)
cv2.imshow('Histogram equalized', img_output)

cv2.waitKey(0)

