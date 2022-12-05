import cv2
import mediapipe as mp
import numpy as np
import datetime
from utils.logger import Logger
import math


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


class EyeDetector:
    def __init__(self, verbose, config):
        self._verbose = verbose
        self._config = config
        self._fireup_count = 0
        self._eye_close_count = 0
        self._sleep_start_time = None
        self._logger = Logger(verbose)


    def get_name(self):
        return self.__class__.__name__


    def distance(self, p1, p2):
        return math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2 + (p2.z - p1.z)**2)

    def detect(self, frame):
        if self._fireup_count < self._config["max_fireup"]:
            self._fireup_count += 1

            return { 
                "message": "FIREUP",
                "status": 0,
                "image": frame,
                "name": self.__class__.__name__
            }

        with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)
                
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results and results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
                """for landm in results.multi_face_landmarks:
                    for i in range(0, 468):
                        pt1 = landm.landmark[i]
                        x = int(pt1.x * frame.shape[1])
                        y = int(pt1.y * frame.shape[0])
                        cv2.putText(image, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.275, (255, 255, 255), 1, 2)"""

                """left_eye_top = 159
                left_eye_bottom = 145
                left_eye_left = 33
                left_eye_right = 133

                right_eye_top = 386
                right_eye_bottom = 374
                right_eye_left = 362
                right_eye_right = 263"""
                
                landmarks = results.multi_face_landmarks[0].landmark
                let = landmarks[159]
                leb = landmarks[145]
                lef = landmarks[33]
                ler = landmarks[133]

                ret = landmarks[386]
                reb = landmarks[374]
                rel = landmarks[362]
                rer = landmarks[263]

                points = [let, leb, lef, ler, ret, reb, rel, rer]

                for p in points:
                    x = int(p.x * frame.shape[1])
                    y = int(p.y * frame.shape[0])
                    cv2.circle(image, (x, y), 2, (0, 0, 255), -1)

                left_distance = self.distance(let, leb)
                right_distance = self.distance(ret, reb)

                value = left_distance + right_distance

                cv2.putText(image, "Difference: " + str(value), (20, int(frame.shape[0] / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.325, (255, 255, 255), 1, 2)
                cv2.putText(image, "Threshold: " + str(self._config["threshold"]), (20, int(frame.shape[0] / 2) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.325, (255, 255, 255), 1, 2)
                
                if value < self._config["threshold"]:
                    if self._sleep_start_time is None:
                        self._sleep_start_time = datetime.datetime.now()

                    self._eye_close_count += 1
                else:
                    self._eye_close_count = 0
                    self._sleep_start_time = None
                    

                if self._sleep_start_time is not None:
                    delta = datetime.datetime.now() - self._sleep_start_time
                    delta = delta.total_seconds() * 1000
                
                    if delta > self._config["sleep_time_threshold"]:
                        return { 
                            "message": "NOT_AWAKE",
                            "status": 1,
                            "image": image,
                            "name": self.__class__.__name__
                        }

            return { 
                "message": "AWAKE",
                "status": 1,
                "image": image,
                "name": self.__class__.__name__
            }
        