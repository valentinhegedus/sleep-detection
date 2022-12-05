import cv2
import mediapipe as mp
import numpy as np
import datetime
from utils.logger import Logger
import math

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()


class BodyPartDetector:
    def __init__(self, verbose, config):
        self._verbose = verbose
        self._config = config
        self._prev_frame = None
        self._fireup_count = 0
        self._movements = 0
        self._movement_start_time = None
        self._prev_landmarks = []
        self._logger = Logger(verbose)


    def get_name(self):
        return self.__class__.__name__
        

    def __get_landmarks_diff(self, l_curr, l_prev):
        diff = 0
        for i in range(0, len(l_prev)):
            diff += math.sqrt((l_prev[i].x - l_curr[i].x)**2 + (l_prev[i].y - l_curr[i].y)**2 )
            #diff += abs((l_prev[i].x - l_curr[i].x) - (l_prev[i].y - l_curr[i].y) - (l_prev[i].z - l_curr[i].z)) 
        return diff


    def detect(self, frame):
        if self._prev_frame is None:
            self.prev_frame = frame
            self._prev_frame = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
            self._prev_frame = cv2.GaussianBlur(self.prev_frame, (21, 21), 0)
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

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        prev_frame = img_rgb
        results = pose.process(img_rgb)

        if results.pose_landmarks:
            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS) 

            landmarks = []
            for idx, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                #cv2.circle(frame, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
                landmarks.append(lm)
            
            if self._prev_landmarks:
                diff = self.__get_landmarks_diff(landmarks, self._prev_landmarks)

                cv2.putText(frame, "Difference: " + str(diff), (20, int(frame.shape[0] / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.325, (255, 255, 255), 1, 2)
                cv2.putText(frame, "Threshold: " + str(self._config["difference_threshold"]), (20, int(frame.shape[0] / 2) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.325, (255, 255, 255), 1, 2)

                if diff > self._config["difference_threshold"]:
                    if self._movement_start_time is None:
                        self._movement_start_time = datetime.datetime.now()

                    self._movements += 1

                if self._movements > self._config["max_movement"]:
                    self._movement_start_time = None
                    self._movements = 0

                    return { 
                        "message": "AWAKE",
                        "status": 1,
                        "image": frame,
                        "name": self.__class__.__name__
                    }

                if self._movement_start_time is not None:
                    delta = datetime.datetime.now() - self._movement_start_time
                    delta = delta.total_seconds() * 1000
                
                    if delta > self._config["movement_time_range"]:
                        self._movement_start_time = None
                        self._movements = 0
            
            self._prev_landmarks = landmarks

        return { 
            "message": "NOT_AWAKE",
            "status": 1,
            "image": frame,
            "name": self.__class__.__name__
        }
        