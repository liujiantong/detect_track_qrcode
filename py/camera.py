#!/usr/bin/env python
# coding: utf-8

import cv2
import threading


class SimpleCamera(object):

    def __init__(self, video_src=0):
        self._cam = None
        self._fps = None
        self._frame = None
        self._frame_width = None
        self._frame_height = None
        self._ret = False
        self._video_src = video_src
        self._stop_event = threading.Event()

    def _init_camera(self):
        self._cam = cv2.VideoCapture(self._video_src)
        self._fps = self._cam.get(cv2.CAP_PROP_FPS)
        self._ret, self._frame = self._cam.read()
        if not self._ret:
            raise Exception("No camera feed")
        self._frame_height, self._frame_width, c = self._frame.shape
        return self._ret

    def start_camera(self):
        """
        Start the running of the camera, without this we can't capture frames
        Camera runs on a separate thread so we can reach a higher FPS
        """
        self._init_camera()
        self._stop_event.clear()
        threading.Thread(target=self._update_camera, args=()).start()

    def read(self):
        """
        With this you can grab the last frame from the camera
        :return (boolean, np.array): return value and frame
        """
        if self.is_running:
            return self._ret, self._frame
        else:
            import warnings
            warnings.warn("Camera is not started, you should start it with start_camera()")
            return False, None

    def _update_camera(self):
        """
        Grabs the frames from the camera
        """
        while self.is_running:
            ret, frm = self._cam.read()
            self._ret, self._frame = (True, frm) if ret else (False, None)

    def get_frame_width_and_height(self):
        """
        Returns the width and height of the grabbed images
        :return (int int): width and height
        """
        return self._frame_width, self._frame_height

    def get_fps(self):
        return self._fps

    def release_camera(self):
        """
        Stop the camera
        """
        self._stop_event.set()
        # self._cam.release()

    @property
    def is_running(self):
        return not self._stop_event.is_set()
