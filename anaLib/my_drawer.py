import cv2
import numpy as np


class Mydrawer():
    def __init__(self,
                 line_colour=[(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0), (255, 0, 0)],
                 line_thickness=1,
                 circle_size=3
    ):
        # drawer
        self.line_colour = line_colour
        self.line_thickness = line_thickness
        self.circle_size = circle_size

    def draw_image(self, image, anno_list):
        for anno in anno_list:
            point = anno['keypoints']
            bbox = anno['bbox']
            x_left, y_top, x_right, y_down = bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]
            image = cv2.rectangle(image, (int(x_left), int(y_top)), (int(x_right), int(y_down)), (0, 0, 255), 2)
            image = self.draw_2d_points(np.array(point).reshape(21, 3), image)
        return image

    def draw_2d_points(self, points, im_ori, finger_num=21):
        if finger_num == 0:
            raise SystemExit('the param "finger_num" is not allowed to set to "0"')
        im = im_ori.copy()
        NUM_JOINTS = points.shape[0]
        for i in range(NUM_JOINTS):
            point = points[i]
            x, y = int(point[0]), int(point[1])

            if i == 0:      # # rootx,rooty表示手腕坐标     prex,prey表示上一个记录点坐标
                rootx, rooty = x, y
                prex, prey = 0, 0

            if i == 1 or i == 5 or i == 9 or i == 13 or i == 17:
                prex, prey = rootx, rooty

            if x == 0 and y == 0:
                prex, prey = 0, 0
                continue

            if prex != 0 and prey != 0:
                if (i > 0) and (i <= 4):        # finger_num = 0
                    cv2.line(im, (prex, prey), (x, y), self.line_colour[0], self.line_thickness, lineType=cv2.LINE_AA)
                    cv2.circle(im, (x, y), self.circle_size, (0, 0, 255), -1)
                if (i > 4) and (i <= 8):        # finger_num = 1
                    cv2.line(im, (prex, prey), (x, y), self.line_colour[1], self.line_thickness, lineType=cv2.LINE_AA)
                    cv2.circle(im, (x, y), self.circle_size, (0, 255, 255), -1)
                if (i > 8) and (i <= 12):       # finger_num = 2
                    cv2.line(im, (prex, prey), (x, y), self.line_colour[2], self.line_thickness, lineType=cv2.LINE_AA)
                    cv2.circle(im, (x, y), self.circle_size, (0, 255, 0), -1)
                if (i > 12) and (i <= 16):      # finger_num = 3
                    cv2.line(im, (prex, prey), (x, y), self.line_colour[3], self.line_thickness, lineType=cv2.LINE_AA)
                    cv2.circle(im, (x, y), self.circle_size, (255, 255, 0), -1)
                if (i > 16) and (i <= 20):      # finger_num = 4
                    cv2.line(im, (prex, prey), (x, y), self.line_colour[4], self.line_thickness, lineType=cv2.LINE_AA)
                    cv2.circle(im, (x, y), self.circle_size, (255, 0, 0), -1)

            else:
                if (i > 0) and (i <= 4):        # finger_num = 0
                    cv2.circle(im, (x, y), self.circle_size, self.line_colour[0], -1)
                if (i > 4) and (i <= 8):        # finger_num = 1
                    cv2.circle(im, (x, y), self.circle_size, self.line_colour[1], -1)
                if (i > 8) and (i <= 12):       # finger_num = 2
                    cv2.circle(im, (x, y), self.circle_size, self.line_colour[2], -1)
                if (i > 12) and (i <= 16):      # finger_num = 3
                    cv2.circle(im, (x, y), self.circle_size, self.line_colour[3], -1)
                if (i > 16) and (i <= 20):      # finger_num = 4
                    cv2.circle(im, (x, y), self.circle_size, self.line_colour[4], -1)

            prex, prey = x, y

        return im

