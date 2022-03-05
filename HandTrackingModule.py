import cv2

# we will use the framework MEDIAPIPE that developed by GOOGLE
import mediapipe as mp

# import all constants
from const import *


class Hand_detector:
    def __init__(self,
                 mode=False,
                 max_num_hands=2,
                 model_complexity=1,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):

        # set class properties
        self.results = None
        self.mode = mode
        self.max_num_hands = max_num_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # creating MEDIAPIPE hand object
        self.mp_hand = mp.solutions.hands

        # creating MEDIAPIPE drawing object
        self.mp_draw = mp.solutions.drawing_utils

        # creating hands object for track and analyze each hand
        self.hands = self.mp_hand.Hands(self.mode,
                                        self.max_num_hands,
                                        self.model_complexity,
                                        self.min_detection_confidence,
                                        self.min_tracking_confidence)

    def find_hands(self, frame, is_draw=True):

        # convert video frame to RGB color because hands object use only rgb images
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # using the hand object we can process what will be the result.
        # Note that process() methode use rgb img
        self.results = self.hands.process(frame_rgb)

        # now we need to extract the information that we need in results parameter if it's not None
        if self.results.multi_hand_landmarks:
            # loop through each hand in results
            for hand_landmarks in self.results.multi_hand_landmarks:
                # check if user choice to draw landmarks on hands
                if is_draw:
                    # we will draw all the landmarks and the connections
                    # between landmark for each hand on each hand using the mp draw object
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hand.HAND_CONNECTIONS)

        # since drawing is an option we will return the frame whether it changed or not
        return frame

    def find_landmarks_pos(self, frame, hand_id=0, is_draw=True):

        # creating list of all landmarks position
        lms_list = []

        # now we need to extract the information that we need in results parameter if it's not none
        if self.results.multi_hand_landmarks:
            # checking which hand of user to find cords
            user_hand = self.results.multi_hand_landmarks[hand_id]

            # we will loop through all landmarks to get the position of every landmark on screen (pixels)
            for lm_id, landmark in enumerate(user_hand.landmark):
                height, width, _ = frame.shape

                # calculating the cords of each landmark on screen
                landmark_cord_x = int(landmark.x * width)
                landmark_cord_y = int(landmark.y * height)

                # adding landmark position to the list
                lms_list.append([lm_id, landmark_cord_x, landmark_cord_y])

                # checking if user wants to draw landmarks on his hand
                if is_draw:
                    cv2.circle(frame, (landmark_cord_x, landmark_cord_y), LMS_RADIUS, LMS_RGB, cv2.FILLED)

                # print("Landmark id: " + str(lm_id) + " cords: (" + str(landmark_cord_x) + "," + str(
                #   landmark_cord_y) + ")")

        return lms_list
