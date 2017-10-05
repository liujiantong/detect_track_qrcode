#!/usr/bin/env python
# coding: utf-8

import cv2

from tracker import ToyTracker
from camera import SimpleCamera


def tracking_callback():
    # frame = tracker.get_frame()
    debug_frame = tracker.get_debug_image()
    # toy_center = tracker.get_last_toy_center()
    toy_colors = tracker.get_toy_colors()

    # cv2.imshow("original frame", frame)
    cv2.imshow("debug frame", debug_frame)
    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        tracker.stop_tracking()

    # print "toy center: {0}".format(toy_center)
    print "toy colors: {0}".format(toy_colors)


if __name__ == "__main__":
    cam = SimpleCamera(video_src=0)
    # cam = SimpleCamera(video_src='../output.avi')
    cam.start_camera()

    tracker = ToyTracker(camera=cam, max_nb_of_centers=10)
    tracker.set_tracking_callback(tracking_callback=tracking_callback)
    tracker.track()

    cam.release_camera()

