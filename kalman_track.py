#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cv2

import imutils
import shapedetector as detector


def init_kalman():
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1],
                                        [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                                       [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03
    return kalman


def get_frame_size(fw, fh, max_width=1024):
    if fw < max_width:
        return fw, fh
    ratio = max_width / float(fw)
    return max_width, np.int32(ratio * fh)


def union_rects(rects):
    if not rects:
        return None

    pnts = []
    for x, y, w, h in rects:
        pnts.append((x, y))
        pnts.append((x + w, y + h))
    return cv2.boundingRect(np.array(pnts))


def compute_bound_rect(fgbg, frame, max_x, max_y):
    fg_mask = fgbg.apply(frame)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    _, fg_mask = cv2.threshold(fg_mask, 60, 255, cv2.THRESH_BINARY)
    im2, contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 100]
    if not boxes:
        return None

    x, y, w, h = union_rects(boxes)
    w = w if x+w < max_x else max_x-x
    h = h if y+h < max_y else max_y-y
    return np.int32([x, y, w, h])


def center(points):
    cx = (points[0][0] + points[1][0] + points[2][0] + points[3][0]) / 4.0
    cy = (points[0][1] + points[1][1] + points[2][1] + points[3][1]) / 4.0
    return np.array([np.float32(cx), np.float32(cy)], np.float32)


if __name__ == '__main__':
    kalman = init_kalman()
    measurement = np.array((2, 1), np.float32)
    prediction = np.zeros((2, 1), np.float32)

    fgbg = cv2.createBackgroundSubtractorMOG2(history=300)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    wb = cv2.xphoto.createSimpleWB()

    cap = cv2.VideoCapture(0)
    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_size = get_frame_size(width, height)

    while True:
        _, frame = cap.read()
        frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_AREA)
        frame = cv2.flip(frame, flipCode=1)

        # united_rect out of range
        united_rect = compute_bound_rect(fgbg, frame, width, height)
        if united_rect is not None:
            # print 'united_rect:', united_rect
            x0, y0, w, h = united_rect
            cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (0, 255, 0), 2)

            rx, ry, rw, rh = united_rect
            roi_image = frame[rx:rx+rw, ry:ry+rh]
            if roi_image.size == 0:
                cv2.imshow('frame', frame)
                continue

            roi_image = wb.balanceWhite(roi_image)
            roi_gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)

            founds = detector.find_contours(roi_gray)
            if not founds:
                cv2.imshow('frame', frame)
                continue

            colors, cnt = detector.detect_color_from_contours(roi_image, founds)
            # print 'colors:', colors, 'cnt:', cnt, 'type(cnt):', type(cnt)
            if cnt is not None:
                print 'colors:', colors
                cnt = cnt + np.array([rx, ry])
                cv2.drawContours(frame, [cnt], 0, (0, 0, 255), 2)
                kalman.correct(center(cnt))
                prediction = kalman.predict()
                (px, py), p_radius = cv2.minEnclosingCircle(cnt)
                cv2.circle(frame, (prediction[0], prediction[1]), np.int32(p_radius), (255, 0, 0))

            cv2.imshow('frame', frame)
            k = cv2.waitKey(10) & 0xFF
            if k == 27 or k == ord('q'):
                break

    cv2.destroyAllWindows()
    cap.release()

