# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import numpy as np
from collections import defaultdict
import json
import shutil

sys.path.append('../../python')
from openpose import pyopenpose as op

from anaLib.merge_images import MergeImages


# 输出2d特征点
def draw_2d_points(points, im_ori, finger_num=21):
    line_colour = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0), (255, 0, 0)]
    line_num = 5
    circle_num = 9

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

        # cv2.putText(im, text=str(i), org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3,
        #             color=line_colour[finger_num], thickness=1)

        prex = x
        prey = y

    return im


def coord_to_box(coords, box_factor=3):
    coord_min = np.min(coords, axis=0)
    coord_max = np.max(coords, axis=0)
    box_c = (coord_max + coord_min) / 2.0
    box_size = np.max(coord_max - coord_min) * box_factor

    x_left = box_c[0] - box_size / 2.0
    y_top = box_c[1] - box_size / 2.0
    w = box_size
    h = w
    return x_left, y_top, w, h


def load_json_data(json_path):
    annotations_dict = defaultdict(list)
    images_dict = {}
    with open(json_path, 'r')as f:
        dataset = json.load(f)

    for ann in dataset['annotations']:
        annotations_dict[ann['image_id']].append(ann)

    for img in dataset['images']:
        images_dict[img['id']] = img

    return images_dict, annotations_dict


class OpenPose_hand():
    def __init__(self):
        self.args = None
        self.params = None
        self.op_wrapper = None
        self.mode = None
        self._init()

        # debug
        self.merger = MergeImages(image_size=256)
        self.count = 0
        self.save_interval = 0

        self.rectangles = [
            [
                op.Rectangle(0., 0., 0., 0.),
                op.Rectangle(0., 0., 0., 0.),
            ],
        ]

    def run(self, image_path, anno_unit_list):
        print(image_path)
        image = cv2.imread(image_path)

        # left_keypoints = np.array(anno_unit_list[1]['keypoints']).reshape(21, 3)
        # left_box = coord_to_box(left_keypoints, 2.2)
        #
        # right_keypoints = np.array(anno_unit_list[0]['keypoints']).reshape(21, 3)
        # right_box = coord_to_box(right_keypoints, 2.2)
        left_box, right_box = None, None
        if anno_unit_list is not None:
            for anno_unit in anno_unit_list:
                keypoints = np.array(anno_unit['keypoints']).reshape(21, 3)
                box = coord_to_box(keypoints, 2.2)
                hand_type = anno_unit['hand_type']
                if hand_type == 'left':
                    left_box = box
                elif hand_type == 'right':
                    right_box = box
                if left_box is None:
                    left_box = [0, 0, 0, 0]
                if right_box is None:
                    right_box = [0, 0, 0, 0]
        else:
            left_box = [0, 0, 0, 0]
            right_box = [0, 0, 0, 0]

        # reset
        # scaling_factor = 1.5
        # x_left = w / 2 - min(w, h)*scaling_factor / 2
        # y_top = h / 2 - min(w, h)*scaling_factor / 2

        # print(f"h:{h}\tw:{w}")
        #
        # for hand_index, hands_rectangle in enumerate(self.rectangles[0]):
        #     if hands_rectangle is None:
        #         self.rectangles[0][hand_index] = \
        #             op.Rectangle(x_left, y_top, min(w, h)*scaling_factor, min(w, h)*scaling_factor)

        self.rectangles = [
            # Left/Right hands person 0
            [
                op.Rectangle(left_box[0], left_box[1], left_box[2], left_box[3]),
                op.Rectangle(right_box[0], right_box[1], right_box[2], right_box[3]),
            ]
        ]

        self.datum.cvInputData = image
        self.datum.handRectangles = self.rectangles

        # Process and display image
        self.op_wrapper.emplaceAndPop(op.VectorDatum([self.datum]))

        keypoints_list, score_list = [], []
        for hand_index in range(2):
            keypoints = self.datum.handKeypoints[hand_index][0]
            keypoints_list.append(keypoints)
            score = np.mean(keypoints[:, 2])
            score_list.append(score)

        image = self.draw_box(image, left_box, keypoints_list[0], 'left', score_list[0])
        image = self.draw_box(image, right_box, keypoints_list[1], 'right', score_list[1])

        save_dir = os.path.join(self.args[0].save_path, os.path.basename(image_path))
        print(f"save path:{save_dir}")
        cv2.imwrite(save_dir, image)
        self.count += 1

        # save intermediate process(debug)
        # if self.count % 500 == 0:
        #     self.save_interval += 1
        #     if self.save_interval % 7 == 0:
        #         self.merger.merge_images(self.args[0].save_path, self.count)
        #     shutil.rmtree(self.args[0].save_path)
        #     os.makedirs(self.args[0].save_path)
        return keypoints_list

    def _init(self):
        self.get_parameter()
        # Starting OpenPose
        self.op_wrapper = op.WrapperPython()
        self.op_wrapper.configure(self.params)
        self.op_wrapper.start()

        # Read image and face rectangle locations
        image = cv2.imread(self.args[0].image_path)
        rectangles = [
            # Left/Right hands person 0
            [
                op.Rectangle(320.035889, 377.675049, 69.300949, 69.300949),
                op.Rectangle(0., 0., 0., 0.),
            ],
            # Left/Right hands person 1
            [
                op.Rectangle(80.155792, 407.673492, 80.812706, 80.812706),
                op.Rectangle(46.449715, 404.559753, 98.898178, 98.898178),
            ],
            # Left/Right hands person 2
            [
                op.Rectangle(185.692673, 303.112244, 157.587555, 157.587555),
                op.Rectangle(88.984360, 268.866547, 117.818230, 117.818230),
            ]
        ]

        # Create new datum
        self.datum = op.Datum()
        self.datum.cvInputData = image
        self.datum.handRectangles = rectangles

        # Process and display image
        self.op_wrapper.emplaceAndPop(op.VectorDatum([self.datum]))
        print("Left hand keypoints: \n" + str(self.datum.handKeypoints[0]))
        print("Right hand keypoints: \n" + str(self.datum.handKeypoints[1]))

    def get_parameter(self):
        # Flags
        parser = argparse.ArgumentParser()
        parser.add_argument("--image_path",
                            default="../../../examples/media/COCO_val2014_000000000241.jpg",
                            help="Check for normal operation")
        parser.add_argument("--data_path",
                            default="/workspace/nas-data/test_data",
                            help=" ")
        parser.add_argument("--save_path",
                            default="/workspace/nas-data/test_data_output/image_res",
                            help=" ")
        args = parser.parse_known_args()

        self.args = args

        # Custom Params (refer to include/openpose/flags.hpp for more parameters)
        params = dict()
        params["model_folder"] = "../../../models/"
        params["hand"] = True
        params["hand_detector"] = 2
        params["body"] = 0

        self.params = params

        save_path = args[0].save_path
        os.makedirs(save_path, exist_ok=True)

        # Add others in path?
        for i in range(0, len(args[1])):
            curr_item = args[1][i]
            if i != len(args[1])-1: next_item = args[1][i+1]
            else: next_item = "1"
            if "--" in curr_item and "--" in next_item:
                key = curr_item.replace('-', '')
                if key not in params:  params[key] = "1"
            elif "--" in curr_item and "--" not in next_item:
                key = curr_item.replace('-', '')
                if key not in params: params[key] = next_item

    def draw_box(self, image, bbox, keypoints, txt, score):
        print(f"draw {txt}~")
        image = cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                              (0, 0, 255), 2)
        image = draw_2d_points(keypoints, image)
        image = cv2.putText(image, txt, (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 5)
        image = cv2.putText(image, str(score), (int(bbox[0]), int(bbox[1]+bbox[3])),
                    cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 5)

        return image


def main():
    image_id = 1400292
    image_path = fr'/workspace/nas-data/openpose/00000{image_id}.jpg'
    json_path = fr'/workspace/nas-data/openpose/00000{image_id}.json'

    _, annotations_dict = load_json_data(json_path)

    for image_id in  annotations_dict.keys():
        anno_unit_list = annotations_dict[image_id]

        mode = OpenPose_hand()
        mode.run(image_path, anno_unit_list)
        print("Success~")


if __name__ == '__main__':
    main()
