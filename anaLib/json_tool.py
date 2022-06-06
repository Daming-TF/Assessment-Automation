import os.path

import numpy as np
import json
from collections import defaultdict


def make_json_head():
    json_struct = dict()

    json_struct['info'] = dict()
    json_struct['info']['description'] = r'The hand keypoint dataset of HUYA.'
    json_struct['info']['url'] = r'Unavailable'
    json_struct['info']['version'] = r'2.0'
    json_struct['info']['year'] = 2021
    json_struct['info']['contributor'] = r'Vision Team of HUYA AI Center'
    json_struct['info']['date_created'] = r'2021-09-21 15:47:40'

    json_struct['licenses'] = list()
    json_struct['licenses'].append(dict({'url': 'Unavailable', 'id': 1, 'name': 'HUYA License'}))

    json_struct['categories'] = list()
    json_struct['categories'].append(dict({
        'supercategory': 'hand',
        'id': 1,
        'name': 'hand',
        'keypoints': [
            'wrist',  # 1
            'thumb1', 'thumb2', 'thumb3', 'thumb4',  # 2-5
            'index1', 'index2', 'index3', 'index4',  # 6-9
            'middle1', 'middle2', 'middle3', 'middle4',  # 10-13
            'ring1', 'ring2', 'ring3', 'ring4',  # 14-17
            'pinky1', 'pinky2', 'pinky3', 'pinky4'  # 18-21
        ],
        'skeleton': [
            [1, 2], [2, 3], [3, 4], [4, 5],
            [1, 6], [6, 7], [7, 8], [8, 9],
            [1, 10], [10, 11], [11, 12], [12, 13],
            [1, 14], [14, 15], [15, 16], [16, 17],
            [1, 18], [18, 19], [19, 20], [20, 21],
            [6, 10], [10, 14], [14, 18]
        ]
    }))

    json_struct['images'] = list()
    json_struct['annotations'] = list()
    return json_struct


def convert_coco_format(json_file, json_total, keypoints, image_id, hand_coco_id,
                        hand_index, height, width, bbox_factor=1.5):
    file_name = str(image_id).zfill(12) + '.jpg'
    assert isinstance(image_id, int)

    # Get visible coordinates
    coco_kps = keypoints.copy()
    coco_kps[:, :2] = keypoints[:, :2]
    coco_kps[:, 2] = keypoints[:, 2]
    kps_valid_bool = coco_kps[:, -1].astype(bool)
    coco_kps[~kps_valid_bool, :2] = 0
    key_pts = coco_kps[:, :2][kps_valid_bool]

    hand_min = np.min(key_pts, axis=0)  # (2,)
    hand_max = np.max(key_pts, axis=0)  # (2,)
    hand_box_c = (hand_max + hand_min) / 2  # (2, )
    half_size = int(np.max(hand_max - hand_min) * bbox_factor / 2.)  # int

    # Get bbox
    x_left = int(hand_box_c[0] - half_size)
    y_top = int(hand_box_c[1] - half_size)
    x_right = x_left + 2 * half_size
    y_bottom = y_top + 2 * half_size
    box_w = x_right - x_left
    box_h = y_bottom - y_top

    # Record the coco information that needs to be stored
    image_dict = dict({
        'license': 1,
        'file_name': file_name,
        'coco_url': 'Unavailable',
        'height': height,
        'width': width,
        'flickr_url': 'Unavailable',
        'id': image_id,
    })

    coco_kps = coco_kps.flatten().tolist()

    anno_dict = dict({
        'segmentation': [[x_left, y_top, x_right, y_top, x_right, y_bottom, x_left, y_bottom]],
        'num_keypoints': 21,
        'area': box_h * box_w,
        'iscrowd': 0,
        'keypoints': coco_kps,
        'image_id': image_id,
        'bbox': [x_left, y_top, box_w, box_h],  # 1.5 expand
        'category_id': 1,
        'id': hand_coco_id,
        'score': 0.5
    })

    if hand_index == 0:
        json_total['images'].append(image_dict)
        json_file['images'].append(image_dict)
    json_total['annotations'].append(anno_dict)
    json_file['annotations'].append(anno_dict)


def load_json_data(json_path):
    print(f"loading json path >>{os.path.dirname(json_path)}<< ......")
    annotations_dict = defaultdict(list)
    images_dict = {}
    with open(json_path, 'r')as f:
        dataset = json.load(f)

    for ann in dataset['annotations']:
        annotations_dict[ann['image_id']].append(ann)

    for img in dataset['images']:
        images_dict[img['id']] = img

    return images_dict, annotations_dict


def get_ids(data_dict):
    return list(data_dict.keys())