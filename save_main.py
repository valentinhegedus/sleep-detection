import cv2
import os 
import time
import sys
import copy
import numpy as np
import datetime

from detectors.body_parts_detection import BodyPartDetector
from detectors.motion_detection import MotionDetector
from detectors.face_detection import FaceDetector
from detectors.eye_detection import EyeDetector
from utils.logger import Logger


logger = Logger(True)

detector_results = []


def main(video=None):
    cap = None
    if video is None:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(f"{os.getcwd()}/assets/videos/{video}")

    prev_frame_time = 0
    new_frame_time = 0

    detectors = [
        BodyPartDetector(verbose=False, config={
            "difference_threshold": 1,
            "max_movement": 10,
            "movement_time_range": 5 * 1000,
            "max_fireup": 20
        }),
        MotionDetector(verbose=False, config={
            "difference_threshold": 100000,
            "max_movement": 20,
            "movement_time_range": 5 * 1000,
            "max_fireup": 20
        }),
        FaceDetector(verbose=False, config={
            "max_fireup": 20
        }),
        EyeDetector(verbose=False, config={
            "threshold": 0.015,
            "sleep_time_threshold": 2 * 1000,
            "max_fireup": 20
        }),
    ]

    for i in range(0, len(detectors)):
        detector_results.append({ "status": 0, "last_active": None, "name": None })

    while True:
        _, frame = cap.read()
        
        for dec_res in detector_results:
            if dec_res["status"] == 1:
                delta = datetime.datetime.now() - dec_res["last_active"]
                delta = delta.total_seconds() * 1000

                if delta > 2 * 1000:
                    dec_res["status"] = 0
                    dec_res["last_active"] = None
                    dec_res["name"] = None


        res_image = detection(frame, detectors)

        new_frame_time = time.time()
        fps = 1 / (new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        cv2.putText(frame, "FPS: " + str(fps), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.325, (255, 255, 255), 1, 2)

        #cv2.imshow("Video", frame)

        scale_percent = 75 
        width = int(res_image.shape[1] * scale_percent / 100)
        height = int(res_image.shape[0] * scale_percent / 100)
        dim = (width, height)
        
        # resize image
        resized = cv2.resize(res_image, dim, interpolation = cv2.INTER_AREA)
        cv2.imshow("Video", resized)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def get_majority_vote(votes):
    awake = 0
    not_awake = 0
    unknown = 0

    for vote in votes:
        if vote["status"] == 1:
            if vote["message"] == "AWAKE":
                awake += 1
            else:
                not_awake += 1
        else:
            unknown += 1

    return {
        "awake": awake,
        "not_awake": not_awake,
        "unknown": unknown
    }


def get_majority_vote_dec_result(votes):
    awake = 0
    not_awake = 0

    for vote in votes:
        if vote["status"] == 1:
            if vote["name"] == "EyeDetector":
                awake += 2
            elif vote["name"] == "FaceDetector":
                awake += 1
            else:
                awake += 1
        else:
            if vote["name"] == "FaceDetector":
                not_awake += 0
            else:
                not_awake += 1

    return {
        "awake": awake,
        "not_awake": not_awake,
    }

def detection(frame, detectors):
    results = []

    for i in range(0, len(detectors)):
        res = detectors[i].detect(np.copy(frame))
        if res["status"] == 1 and res["message"] == "AWAKE":
            detector_results[i]["status"] = 1
            detector_results[i]["last_active"] = datetime.datetime.now()
            detector_results[i]["name"] = detectors[i].get_name()

        results.append(res)

    votes = get_majority_vote_dec_result(detector_results)
    #logger.log("Main", votes)
    #logger.log("Main", max(votes, key=votes.get))

    return show_detectors(results)
        

def show_detectors(results):
    concat = results[0]["image"]
    awake_text = "Yes" if detector_results[0]["status"] == 1 else "No"
    color = (0, 255, 0) if detector_results[0]["status"] == 1 else (255, 255, 255)
    cv2.putText(results[0]["image"], results[0]["name"], (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.325, (255, 255, 255), 1, 2)
    cv2.putText(results[0]["image"], "Awake: " + awake_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.325, color, 1, 2)

    for i in range(1, len(results)):
        awake_text = "Yes" if detector_results[i]["status"] == 1 else "No"
        color = (0, 255, 0) if detector_results[i]["status"] == 1 else (255, 255, 255)
        cv2.putText(results[i]["image"], results[i]["name"], (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.325, (255, 255, 255), 1, 2)
        cv2.putText(results[i]["image"], "Awake: " + awake_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.325, color, 1, 2)
        concat = np.concatenate((concat, results[i]["image"]), axis=1)

    cv2.rectangle(concat, (0, concat.shape[0] - 50), (concat.shape[1], concat.shape[0]), (0, 0, 0), -1)

    votes = get_majority_vote_dec_result(detector_results)
    print(votes)

    awake_text = "Yes" if max(votes, key=votes.get) == "awake" else "No"
    color = (0, 255, 0) if awake_text == "Yes" else (255, 255, 255)
    cv2.putText(concat, "Awake: " + awake_text, (int(concat.shape[1] / 2), concat.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.325, color, 1, 2)

    return concat
    
if __name__ == "__main__":
    main()
