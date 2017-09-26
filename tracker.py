#!/usr/bin/env python
# coding: utf-8

import cv2
import sys


if __name__ == '__main__':
    # Set up tracker.
    # Instead of MIL, you can also use
    # BOOSTING, KCF, TLD, MEDIANFLOW or GOTURN

    # tracker = cv2.TrackerMIL_create()
    tracker = cv2.TrackerMedianFlow_create()

    # Read video
    video = cv2.VideoCapture(0)

    ok, frame = video.read()
    for i in xrange(30):
        ok, frame = video.read()
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow("Tracking", frame)

    # bbox = cv2.selectROI(frame, False)
    bbox = (280, 128, 174, 149)

    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)

    while True:
        # Read a new frame
        ok, frame = video.read()
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # Update tracker
        ok, bbox = tracker.update(frame)

        # Draw bounding box
        if ok:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (0, 0, 255))

        # Display result
        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        k = cv2.waitKey(10) & 0xff
        if k == 27:
            break

    video.release()
    cv2.destroyAllWindows()

