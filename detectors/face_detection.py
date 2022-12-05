import cv2
import mediapipe as mp
import numpy as np
import datetime
from utils.logger import Logger


mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


class FaceDetector:
    def __init__(self, verbose, config):
        self._verbose = verbose
        self._config = config
        self._fireup_count = 0
        self._logger = Logger(verbose)


    def get_name(self):
        return self.__class__.__name__


    def detect(self, frame):
        if self._fireup_count < self._config["max_fireup"]:
            self._fireup_count += 1
            return { 
                "message": "FIREUP",
                "status": 0,
                "image": frame,
                "name": self.__class__.__name__
            }

        with mp_face_detection.FaceDetection(model_selection = 1, min_detection_confidence = 0.5) as face_detection:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            results = face_detection.process(image)
            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(image, detection)

                return { 
                    "message": "NOT_AWAKE",
                    "status": 1,
                    "image": image,
                    "name": self.__class__.__name__
                }

            return { 
                "message": "AWAKE",
                "status": 1,
                "image": frame,
                "name": self.__class__.__name__
            }
        