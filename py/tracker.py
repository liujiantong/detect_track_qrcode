#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np

from collections import deque
from types import FunctionType

from detector import ToyDetector
import helper


class ToyTracker(object):

    def __init__(self, camera, max_nb_of_centers=None, debug=True):
        """
        :param camera: Camera object which parent is a Camera object (like WebCamera)
        :param max_nb_of_centers: Maxmimum number of points for storing. If it is set
        to None than it means there is no limit
        :param debug: When it's true than we can see the visualization of the captured points etc...
        """
        self._camera = camera
        self._tracker_centers = None
        self._debug = debug
        self._max_nb_of_centers = max_nb_of_centers
        self._tracking_callback = None
        self._is_running = False
        self._frame = None
        self._debug_frame = None
        self._kalman = None
        self._fgbg = None

        self._united_fg = None
        self._toy_contour = None
        self._toy_radius = 0
        self._toy_colors = None
        self._measurement = np.array((2, 1), np.float32)
        self._toy_prediction = np.zeros((2, 1), np.float32)

        self._init_tracker()
        self._create_tracker_center_history()

    def _init_tracker(self):
        w, h = self._camera.get_frame_width_and_height()
        self._frame_width, self._frame_height = helper.get_frame_size(w, h)

        self._kalman = cv2.KalmanFilter(4, 2)
        self._kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self._kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1],
                                            [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self._kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                                           [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03

        self._fgbg = cv2.createBackgroundSubtractorMOG2(history=300, detectShadows=False)

    def _create_tracker_center_history(self):
        """
        Initialize the tracker point list
        """
        if self._max_nb_of_centers:
            self._tracker_centers = deque(maxlen=self._max_nb_of_centers)
        else:
            self._tracker_centers = deque()

    def set_tracking_callback(self, tracking_callback):
        if not isinstance(tracking_callback, FunctionType):
            raise Exception("tracking_callback is not a valid Function with type: FunctionType!")
        self._tracking_callback = tracking_callback

    def _read_from_camera(self):
        ret, frame = self._camera.read()
        if ret:
            frame = cv2.resize(frame, (self._frame_width, self._frame_height), interpolation=cv2.INTER_AREA)
            self._frame = cv2.flip(frame, flipCode=1)
        else:
            import warnings
            warnings.warn("There is no camera feed! Stop tracking.")
            self.stop_tracking()

    def track(self):
        self._is_running = True

        detector = ToyDetector()
        wb = cv2.xphoto.createSimpleWB()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

        while True:
            self._read_from_camera()
            self._debug_frame = self._frame.copy()

            united_rect = self._compute_bound_rect(self._frame, self._frame_width, self._frame_height, kernel)
            if united_rect is not None:
                self._united_fg = united_rect
                roi_x, roi_y, roi_w, roi_h = self._united_fg

                roi_image = self._frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
                # roi_image = wb.balanceWhite(roi_image)
                roi_gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)

                founds = detector.find_contours(roi_gray)
                if founds:
                    roi_image = wb.balanceWhite(roi_image)
                    colors, cnt = detector.detect_color_from_contours(roi_image, founds)
                    if cnt is not None:
                        self._toy_colors = colors
                        self._toy_contour = cnt + np.array([roi_x, roi_y])
                        self._measurement = helper.center(self._toy_contour)
                        self._kalman.correct(self._measurement)
                        self._toy_prediction = self._kalman.predict()
                        _, self._toy_radius = cv2.minEnclosingCircle(cnt)
                        self._add_new_tracker_point()
            else:
                self._clear_debug_things()

            if self._debug:
                self._draw_debug_things(draw_fg=False)

            if self._tracking_callback is not None:
                try:
                    self._tracking_callback()
                except TypeError:
                    import warnings
                    warnings.warn(
                        "Tracker callback function is not working because of wrong arguments! "
                        "It takes zero arguments")

            if not self._is_running:
                break

    def _compute_bound_rect(self, frm, max_x, max_y, kernel):
        fg_mask = self._fgbg.apply(frm)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        _, fg_mask = cv2.threshold(fg_mask, 60, 255, cv2.THRESH_BINARY)
        im2, contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 100]
        if not boxes:
            return None

        x, y, w, h = helper.union_rects(boxes)
        w = w if x + w < max_x else max_x - x
        h = h if y + h < max_y else max_y - y
        return np.int32([x, y, w, h])

    def _add_new_tracker_point(self, min_distance=20, max_distance=np.inf):
        try:
            dst = helper.calc_distance(self._tracker_centers[-1], self._measurement)
            if max_distance > dst > min_distance:
                self._tracker_centers.append(self._measurement)
        except IndexError:
            # It happens only when the queue is empty and we need a starting point
            self._tracker_centers.append(self._measurement)

    def _draw_debug_things(self, draw_fg=True, draw_contour=True, draw_prediction=True):
        if draw_fg and self._united_fg is not None:
            roi_x, roi_y, roi_w, roi_h = self._united_fg
            cv2.rectangle(self._debug_frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
        if draw_contour and self._toy_contour is not None:
            cv2.drawContours(self._debug_frame, [self._toy_contour], 0, (0, 0, 255), 2)
        if draw_prediction and self._toy_radius > 0:
            cv2.circle(self._debug_frame, (self._toy_prediction[0], self._toy_prediction[1]),
                       np.int32(self._toy_radius), (255, 0, 0))

    def _clear_debug_things(self):
        """
        clear debug image
        """
        self._toy_colors = None
        self._toy_contour = None
        self._toy_prediction = None
        self._toy_radius = 0

    def stop_tracking(self):
        """
        Stop the color tracking
        """
        self._is_running = False

    def get_debug_image(self):
        return self._debug_frame

    def get_frame(self):
        return self._frame

    def get_last_toy_center(self):
        return self._measurement

    def get_toy_colors(self):
        return self._toy_colors
