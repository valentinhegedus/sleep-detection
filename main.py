import cv2
import os 
import time
import sys

from detectors.body_parts_detection import BodyPartDetector
from utils.logger import Logger


logger = Logger(True)


def main(video=None):
    cap = None
    if video is None:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(f"{os.getcwd()}/assets/videos/{video}")

    prev_frame_time = 0
    new_frame_time = 0

    """
    A listaba rakjatok a detektorokat
    A detektornak egy ilyet kell visszaadnia: { "status":  <staus>, "message": <message> }
        status:
            status == 0 -> Valami rossz vagy meg nem all keszen
            status == 1 -> Detektalt valamit
        message:
            message == "AWAKE" -> Ebren
            message == "NOT_AWAKE" -> Nem ebren
            message == ~valami mas -> Nem jelent semmit, csak hogy trackeljuk az eventeket
    """
    detectors = [
        BodyPartDetector(verbose=False, config={
            "difference_threshold": 1,
            "max_movement": 50,
            "movement_time_range": 5 * 1000,
            "max_fireup": 20
        })
    ]

    while True:
        _, frame = cap.read()
        
        detection(frame, detectors)

        new_frame_time = time.time()
        fps = 1 / (new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        cv2.putText(frame, "FPS: " + str(fps), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.325, (255, 255, 255), 1, 2)

        cv2.imshow("Video", frame)

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


def detection(frame, detectors):
    results = []
    for detector in detectors:
        results.append(detector.detect(frame))

    votes = get_majority_vote(results)
    logger.log("Main", max(votes, key=votes.get))
        
    
if __name__ == "__main__":
    main()
