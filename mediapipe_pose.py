
import copy
import argparse

from collections import deque
import cv2 as cv
import numpy as np
import mediapipe as mp


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


class CvFpsCalc(object):
    def __init__(self, buffer_len=1):
        self._start_tick = cv.getTickCount()
        self._freq = 1000.0 / cv.getTickFrequency()
        self._difftimes = deque(maxlen=buffer_len)

    def get(self):
        current_tick = cv.getTickCount()
        different_time = (current_tick - self._start_tick) * self._freq
        self._start_tick = current_tick

        self._difftimes.append(different_time)

        fps = 1000.0 / (sum(self._difftimes) / len(self._difftimes))
        fps_rounded = round(fps, 2)

        return fps_rounded




def main():

    cap_device = 0
    cap_width = 1080
    cap_height = 1080


    # camera prep
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)


    # Model Load

    mp_pose = mp.solutions.pose

    pose = mp_pose.Pose(static_image_mode=True, model_complexity=2,enable_segmentation=True)

    cvFpsCalc = CvFpsCalc(buffer_len=10)


    


    while True:
        fps = cvFpsCalc.get()

        # Process Key (ESC: end)
        key=cv.waitKey(10)
        if key==27: #ESC
            break



        # Camera capture
        ret, image = cap.read()
        if not ret:
            break

        image = cv.flip(image,1)
        debug_image = copy.deepcopy(image)


        # Detection Implementation
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True


        if results.pose_landmarks!=None:

            mp_drawing.draw_landmarks(debug_image, 
                                      results.pose_landmarks, 
                                      mp_pose.POSE_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())


        cv.imshow("Pose Estimation : ", debug_image)

    cap.release()
    cv.destroyAllWindows()




if __name__ == '__main__':
    main()
