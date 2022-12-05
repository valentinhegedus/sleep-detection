import cv2
import time
import numpy as np
import datetime

from utils.logger import Logger


class MotionDetector:
    def __init__(self, verbose, config):
        self._verbose = verbose
        self._config = config
        self._prev_frame = None
        self._fireup_count = 0
        self._movements = 0
        self._movement_start_time = None
        self._logger = Logger(verbose)


    def get_name(self):
        return self.__class__.__name__


    def detect(self, frame):
        if self._prev_frame is None:
            self._prev_frame = frame
            self._prev_frame = cv2.cvtColor(self._prev_frame, cv2.COLOR_BGR2GRAY)
            self._prev_frame = cv2.GaussianBlur(self._prev_frame, (21, 21), 0)
            #print(self._prev_frame is None)
            return { 
                "message": "NO_PREV_FRAME",
                "status": 0,
                "image": frame,
                "name": self.__class__.__name__
            }

        if self._fireup_count < self._config["max_fireup"]:
            self._fireup_count += 1
            return { 
                "message": "FIREUP",
                "status": 0,
                "image": frame,
                "name": self.__class__.__name__
            }

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        difference = cv2.absdiff(gray, self._prev_frame)
        threshold_frame = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)[1]
        self._prev_frame = gray

        cv2.putText(threshold_frame, "Difference: " + str(int(threshold_frame.sum() / 100)), (20, int(frame.shape[0] / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.325, (255, 255, 255), 1, 2)
        cv2.putText(threshold_frame, "Threshold: " + str(self._config["difference_threshold"]), (20, int(frame.shape[0] / 2) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.325, (255, 255, 255), 1, 2)


        if int(threshold_frame.sum() / 100) > self._config["difference_threshold"]:
            if self._movement_start_time is None:
                self._movement_start_time = datetime.datetime.now()

            self._movements += 1
        
        if self._movements > self._config["max_movement"]:
            self._movement_start_time = None
            self._movements = 0
            return { 
                "message": "AWAKE",
                "status": 1,
                "image": cv2.cvtColor(threshold_frame, cv2.COLOR_GRAY2RGB),
                "name": self.__class__.__name__
            }

        if self._movement_start_time is not None:
            delta = datetime.datetime.now() - self._movement_start_time
            delta = delta.total_seconds() * 1000
        
            if delta > self._config["movement_time_range"]:
                self._movement_start_time = None
                self._movements = 0

        return { 
            "message": "NOT_AWAKE",
            "status": 1,
            "image": cv2.cvtColor(threshold_frame, cv2.COLOR_GRAY2RGB),
            "name": self.__class__.__name__
        }
