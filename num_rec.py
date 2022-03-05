import sys

import HandTrackingModule as htm_module

from const import *

import time

import cv2


def main():
    # set landmark id that we want to spot on it (u can change that if u want)
    lm_id = 4

    # how to capture a video from web camara - VideoCapture(0) or VideoCapture(cam_id)
    camara_cap = cv2.VideoCapture(DEFAULT_CAMERA)

    # creating frame rate
    prev_time = 0

    # Create hand detector object by module
    detector = htm_module.Hand_detector()

    # frame loop
    while STREAM:
        # getting frame from video
        status, frame = camara_cap.read()

        if status:

            # mirror webcam frame
            frame = cv2.flip(frame, 1)

            # using our module we can detect hands and draw on it, and we will get new painted frame
            frame = detector.find_hands(frame)

            # getting all landmarks position
            landmarks = detector.find_landmarks_pos(frame)

            # checking if landmarks are existing
            if len(landmarks) > 0:
                # printing the info of specific landmark on hand according to the map
                print(str(landmarks[lm_id]))

            # update current time zone
            current_time = time.time()

            # calculating the FPS
            fps = 1 / (current_time - prev_time)

            # update previous time zone
            prev_time = current_time

            # Display frame rate on screen
            cv2.putText(frame, str(int(fps)), POS_FPS, cv2.FONT_HERSHEY_PLAIN, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)

            cv2.imshow(WIN_NAME, frame)

            if cv2.waitKey(MILI_SEC) & 0xFF == ord('q'):
                sys.exit()
        else:
            break


if __name__ == '__main__':
    main()
