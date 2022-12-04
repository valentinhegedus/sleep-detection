import cv2
import time
import numpy as np
import datetime
import threading
import os 

from playsound import playsound
from utils.logger import Logger


CAP = None


def play_alarm():
    playsound(f"{os.getcwd()}/assets/sounds/alarm.wav")


def detect_motion(config=None, video=None):
    global CAP
    if video is None:
        CAP = cv2.VideoCapture(0)
    else:
        CAP = cv2.VideoCapture(f"{os.getcwd()}/assets/videos/{video}")

    _, prev_frame = CAP.read()
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_frame = cv2.GaussianBlur(prev_frame, (21, 21), 0)

    movements = 0
    movement_start_time = None
    
    fireup_count = 0

    prev_frame_time = 0
    new_frame_time = 0

    while True:
        _, frame = CAP.read()

        # Video has no more frame
        if frame is None:
            break

        if fireup_count < config["max_fireup"] and video is None:
            fireup_count += 1
            Logger.log("Waiting for camera...")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        difference = cv2.absdiff(gray, prev_frame)
        threshold_frame = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)[1]
        prev_frame = gray

        if threshold_frame.sum() > config["difference_threshold"]:
            if movement_start_time is None:
                movement_start_time = datetime.datetime.now()

            movements += 1
            Logger.log(f"Movements: {movements}")
        
        if movements > config["max_movement"]:
            threading.Thread(target=play_alarm).start()
            movement_start_time = None
            movements = 0
            Logger.log("Movement detected!")

        if movement_start_time is not None:
            delta = datetime.datetime.now() - movement_start_time
            delta = delta.total_seconds() * 1000
        
            if delta > config["movement_time_range"]:
                movement_start_time = None
                movements = 0
                Logger.log("Reset")
                

        numpy_horizontal_concat = np.concatenate((frame, cv2.cvtColor(threshold_frame, cv2.COLOR_GRAY2RGB)), axis=1)

        new_frame_time = time.time()
        fps = 1 / (new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)

        cv2.putText(numpy_horizontal_concat, "FPS: " + str(fps), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.325, (255, 255, 255), 1, 2)
        cv2.putText(numpy_horizontal_concat, "Movements: " + str(movements), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, 2)

        cv2.imshow("Video", numpy_horizontal_concat)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
     

if __name__ == "__main__":
    config = {
        "difference_threshold": 100000,
        "max_movement": 50,
        "movement_time_range": 5 * 1000,
        "max_fireup": 20
    }
    
    detect_motion(config=config, video="test_2.mp4")

    CAP.release()
    cv2.destroyAllWindows()
