#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cv2

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


def compute_bound_rect(fg_bg, frm, max_x, max_y):
    fg_mask = fg_bg.apply(frm)
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


def show_video(frm, roi=None):
    cv2.imshow('frame', frm)
    k = cv2.waitKey(10) & 0xFF
    if k == 27 or k == ord('q'):
        return True
    elif k == ord('s'):
        cv2.imwrite('kalman_frame.png', frm)
        if not roi:
            cv2.imwrite('kalman_roi.png', roi)
    return False


# video_src = 0
video_src = 'output.avi'


if __name__ == '__main__':
    kalman = init_kalman()
    measurement = np.array((2, 1), np.float32)
    prediction = np.zeros((2, 1), np.float32)

    fgbg = cv2.createBackgroundSubtractorMOG2(history=300)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    wb = cv2.xphoto.createSimpleWB()

    # cap = cv2.VideoCapture(video_src)
    cap = cv2.VideoCapture(video_src)
    width0, height0 = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width, height = get_frame_size(width0, height0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # if video_src == 0:
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        frame = cv2.flip(frame, flipCode=1)

        # united_rect out of range
        united_rect = compute_bound_rect(fgbg, frame, width, height)
        if united_rect is not None:
            # print 'united_rect:', united_rect
            roi_x, roi_y, roi_w, roi_h = united_rect
            cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)

            roi_image = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

            roi_image = wb.balanceWhite(roi_image)
            roi_gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)

            founds = detector.find_contours(roi_gray)
            if not founds:
                if show_video(frame, roi_image):
                    break
                continue

            roi_image = wb.balanceWhite(roi_image)
            colors, cnt = detector.detect_color_from_contours(roi_image, founds)
            # print 'colors:', colors, 'cnt:', cnt, 'type(cnt):', type(cnt)
            if cnt is not None:
                print 'colors:', colors
                cnt = cnt + np.array([roi_x, roi_y])
                cv2.drawContours(frame, [cnt], 0, (0, 0, 255), 2)
                kalman.correct(center(cnt))
                prediction = kalman.predict()
                (px, py), p_radius = cv2.minEnclosingCircle(cnt)
                cv2.circle(frame, (prediction[0], prediction[1]), np.int32(p_radius), (255, 0, 0))

            if show_video(frame, roi_image):
                break

    cv2.destroyAllWindows()
    cap.release()

