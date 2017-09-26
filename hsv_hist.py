#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
from matplotlib import pyplot as plt


"""
H =   0°...30° => RED
H = 30°..90° => Yellow
H = 90°..150° => Green
H = 150°..210° => Cyan
H = 210°..270° => Blue
H = 270°..330° => Magenta
H = 330°..360° => RED

Dividing Hue by 2 you have corresponding value ready for OpenCV:

Red   : 0<= H <30 and 150<= H <180
Green : 30<= H < 90
Blue  : 90<= H < 150
"""


red_range1 = (0, 30)
red_range2 = (150, 180)
green_range = (30, 90)
blue_range = (90, 140)


def detect_color(h):
    rval = np.sum(h[red_range1[0]:red_range1[1]]) + np.sum(h[red_range2[0]:red_range1[1]])
    gval = np.sum(h[green_range[0]:green_range[1]])
    bval = np.sum(h[blue_range[0]:blue_range[1]])

    colors = []
    if rval > 0.5:
        colors.append('red')
    if gval > 0.5:
        colors.append('green')
    if bval > 0.5:
        colors.append('blue')

    return colors


img = cv2.imread('block01.png')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, np.array((0., 20., 0.)), np.array((180., 255., 255.)))
# colors = ('h', 's', 'v')
draw_colors = ('r', 'g', 'b')
max_vals = (180, 256, 256)

# for i, col in enumerate(draw_colors):
#     hist = cv2.calcHist([hsv], [i], None, [36], [0, max_vals[i]])
#     plt.plot(hist, color=col)
#     plt.xlim([0, 256])

x = np.arange(180) + 0.5
hist = cv2.calcHist([hsv], [0], mask, [180], [0, 180])
# print "type(hist):", type(hist)
# print "hist.shape:", hist.shape
cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
colors = detect_color(hist)
print ','.join(colors)
plt.bar(x, hist)

plt.show()

