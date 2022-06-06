import cv2
import numpy as np


def mp_drawer(image, hand_landmarks, width, height):
    keypoints = np.zeros([21, 3])
    for k in range(21):
        keypoints[k, :2] = hand_landmarks.landmark[k].x * width, hand_landmarks.landmark[k].y * height
        keypoints[k, 2] = 2

    return draw_2d_points(keypoints, image)

line_colour = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0), (255, 0, 0)]


def draw_2d_points(points, im_ori, finger_num = 21):
    '''

    Parameters
    ----------
    points: An array that satisfies the 21*3 format
    im_ori: oright image
    finger_num: 标识输出landmarks模式 (landmarks数量)

    Returns
    -------
    im: a image with landmarks
    '''
    line_num = 1    # 5
    circle_num = 3  # 9

    if finger_num == 0:
        raise SystemExit('the param "finger_num" is not allowed to set to "0"')
    im = im_ori.copy()
    NUM_JOINTS = points.shape[0]
    for i in range(NUM_JOINTS):
        point = points[i]
        x = int(point[0])
        y = int(point[1])

        if i == 0:
            # 记录手腕关键点
            # rootx,rooty表示手腕坐标
            rootx = x
            rooty = y
            prex = 0
            prey = 0

        if i == 1 or i == 5 or i == 9 or i == 13 or i == 17:
            prex = rootx
            prey = rooty

        #
        if x == 0 and y == 0:
            prex = 0
            prey = 0
            continue

        # add new “if prex != 0 and prey != 0:” 是为了预防手腕关键点没有识别到？
        if prex != 0 and prey != 0:
            if (i > 0) and (i <= 4):
                cv2.line(im, (prex, prey), (x, y), line_colour[0], line_num, lineType=cv2.LINE_AA)
                cv2.circle(im, (x, y), circle_num, (0, 0, 255), -1)
                finger_num = 0
            if (i > 4) and (i <= 8):
                cv2.line(im, (prex, prey), (x, y), line_colour[1], line_num, lineType=cv2.LINE_AA)
                cv2.circle(im, (x, y), circle_num, (0, 255, 255), -1)
                finger_num = 1
            if (i > 8) and (i <= 12):
                cv2.line(im, (prex, prey), (x, y), line_colour[2], line_num, lineType=cv2.LINE_AA)
                cv2.circle(im, (x, y), circle_num, (0, 255, 0), -1)
                finger_num = 2
            if (i > 12) and (i <= 16):
                cv2.line(im, (prex, prey), (x, y), line_colour[3], line_num, lineType=cv2.LINE_AA)
                cv2.circle(im, (x, y), circle_num, (255, 255, 0), -1)
                finger_num = 3
            if (i > 16) and (i <= 20):
                cv2.line(im, (prex, prey), (x, y), line_colour[4], line_num, lineType=cv2.LINE_AA)
                cv2.circle(im, (x, y), circle_num, (255, 0, 0), -1)
                finger_num = 4
        else:
            if (i > 0) and (i <= 4):
                cv2.circle(im, (x, y), circle_num, line_colour[0], -1)
                finger_num = 0
            if (i > 4) and (i <= 8):
                cv2.circle(im, (x, y), circle_num, line_colour[1], -1)
                finger_num = 1
            if (i > 8) and (i <= 12):
                cv2.circle(im, (x, y), circle_num, line_colour[2], -1)
                finger_num = 2
            if (i > 12) and (i <= 16):
                cv2.circle(im, (x, y), circle_num, line_colour[3], -1)
                finger_num = 3
            if (i > 16) and (i <= 20):
                cv2.circle(im, (x, y), circle_num, line_colour[4], -1)
                finger_num = 4


        prex = x
        prey = y

    return im