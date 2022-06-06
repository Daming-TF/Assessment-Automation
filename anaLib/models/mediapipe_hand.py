import cv2
import numpy as np
import mediapipe as mp


def convert_to_2d(hand_landmarks, width, height):
    keypoints = np.zeros([21, 3])
    for k in range(21):
        keypoints[k, :2] = hand_landmarks.landmark[k].x * width, hand_landmarks.landmark[k].y * height
        keypoints[k, 2] = 2
    return keypoints


class Mediapipe:
    def __init__(self, model_complexity=1):
        self.model = mp.solutions.hands.Hands(model_complexity=model_complexity)

    def __call__(self, image, width, height):
        # To improve performance, optionally mark the image as not writeable to pass by reference.
        results = self.model.process(image)

        hands_list = []
        if results.multi_hand_landmarks:
            for index, hand_landmarks in enumerate(results.multi_hand_landmarks):
                keypoints = convert_to_2d(hand_landmarks, width, height)
                hands_list.append(keypoints)

        return hands_list
