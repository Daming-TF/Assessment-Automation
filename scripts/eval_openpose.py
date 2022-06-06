import json
import cv2
import os
import numpy as np
from argparse import ArgumentParser
from copy import deepcopy
from collections import defaultdict

from demo_openpose import OpenPose_hand
from anaLib.EvalTools import EvaluationIntegration, WriteExcel
from anaLib.json_tool import make_json_head, convert_coco_format
from anaLib.my_drawer import Mydrawer


def main(opts):
    image_divide_ids = ['000001402094.jpg', '000001404081.jpg', '000001404682.jpg', '000001410139.jpg',
                        '000001410905.jpg', '000001412448.jpg', '000001414397.jpg', '000001416186.jpg',
                        '000001419204.jpg', '000001423669.jpg']
    video_ids = ['hand_test_01', 'hand_test_02', 'hand_test_03', 'hand_test_04', 'hand_test_05', 'hand_test_06',
                 'hand_test_07', 'hand_test_08', 'hand_test_09', 'hand_test_10']

    excel_save_path, json_save_path, json_head, json_total, _, hand_coco_id, bbox_factor = init(opts)
    _, annotations_dict = load_json_data(os.path.join(opts.gt_dir, 'gt.json'))

    # init
    mode = OpenPose_hand()
    my_eval = EvaluationIntegration()
    drawer = Mydrawer()

    print(f"image divide:{image_divide_ids}")

    # init
    json_file = deepcopy(json_head)
    unit_json_save_dir = os.path.join(opts.save_dir, 'anno_res')
    os.makedirs(unit_json_save_dir, exist_ok=True)
    remove_file(unit_json_save_dir, image_divide_ids)

    # process
    start_image_id = 1400000
    last_image_id = 1400000
    video_index = 0
    images_names = os.listdir(opts.images_dir)
    images_names.sort()

    for index, image_name in enumerate(images_names):
        if not image_name.endswith('.jpg'):
            continue

        unit_json_save_path = os.path.join(unit_json_save_dir, image_name.replace('.jpg', '.json'))
        image_id = int(image_name.split('.jpg')[0])
        if os.path.exists(unit_json_save_path):
            hand_coco_id = (image_id - start_image_id) * 2 + 1
            continue

        print(f"image_id:{image_id}")

        if image_id in annotations_dict.keys():
            anno_unit_list = annotations_dict[image_id]
        else:
            anno_unit_list = None

        image_path = os.path.join(opts.images_dir, image_name)
        hands_list = mode.run(image_path, anno_unit_list)

        image = cv2.imread(image_path)
        height, width = image.shape[:2]

        json_file_unit = deepcopy(json_head)

        del_list = []
        for hand_index, keypoints in enumerate(hands_list):
            if np.all(keypoints == 0):
                del_list.append(hand_index)

        for del_index in del_list[::-1]:
            del hands_list[del_index]


        print(hands_list)

        for hand_index, keypoints in enumerate(hands_list):
            convert_coco_format(json_file_unit, json_total, keypoints, image_id, hand_coco_id,
                                hand_index, int(height), int(width), bbox_factor=bbox_factor)
            image = drawer.draw_2d_points(keypoints, image)
            hand_coco_id += 1

        write_unit_json(unit_json_save_path, json_file_unit)

        if image_name in image_divide_ids:
            video_id = video_ids[video_index]
            # save dt json
            save_path = os.path.join(opts.save_dir, video_id + '.json')
            now_image_id = int(image_name.split('.jpg')[0])
            convert_json(save_path, unit_json_save_dir, last_image_id, now_image_id)

            # eval
            gt_path = os.path.join(opts.gt_dir, video_id + '--gt.json')
            res = my_eval.eval(gt_path, save_path)

            # write excel
            video_unit_excel_save_path = os.path.join(opts.save_dir, video_id + '.xlsx')
            writer = WriteExcel(video_unit_excel_save_path, opts.model_name)
            write_info = [video_id] + res
            print(f"video id:{video_id}\tres:{res}")
            writer.write(write_info, index)
            writer.close()

            # update info
            last_image_id = now_image_id
            video_index += 1

    # total eval
    end_image_id = image_divide_ids[-1].split('.jpg')[0]
    convert_json(json_save_path, unit_json_save_dir, start_image_id, int(end_image_id))

    gt_path = os.path.join(opts.gt_dir, 'gt.json')
    res = my_eval.eval(gt_path, json_save_path)

    writer = WriteExcel(excel_save_path, opts.model_name)
    write_info = ["total"] + res
    writer.write(write_info, len(video_ids))
    writer.close()


def remove_file(unit_json_save_dir, image_divide_ids):
    for divide_id in image_divide_ids:
        file_path = os.path.join(unit_json_save_dir, divide_id.replace('.jpg', '.json'))
        if os.path.exists(file_path):
            print(f"remove {file_path}")
            os.remove(file_path)


def write_unit_json(save_path, json_file):
    with open(save_path, 'w')as f:
        json.dump(json_file, f)


def convert_json(save_path, unit_json_dir, start_id, end_id):
    annotations_dict = defaultdict(list)
    images_dict = {}

    for image_id in range(start_id, end_id+1):
        json_name = str(image_id).zfill(12)+'.json'
        json_path = os.path.join(unit_json_dir, json_name)
        if not os.path.exists(json_path):
            continue

        with open(json_path, 'r') as f:
            dataset = json.load(f)

        for ann in dataset['annotations']:
            annotations_dict[ann['image_id']].append(ann)

        for img in dataset['images']:
            images_dict[img['id']] = img

    write_video_json(images_dict, annotations_dict, save_path)


def write_video_json(images_dict, annotations_dict, save_path):
    json_head = make_json_head()
    images_ids = list(images_dict.keys())
    for image_id in images_ids:
        image_info = images_dict[image_id]
        json_head["images"].append(image_info)

        annotation_info_list = annotations_dict[image_id]
        for annotation_info in annotation_info_list:
            json_head["annotations"].append(annotation_info)

    print(f"writing {save_path}......")
    with open(save_path, 'w')as f:
        json.dump(json_head, f)


def write_unit_json(save_path, json_file):
    print(f"writing {save_path}......")
    with open(save_path, 'w') as f:
        json.dump(json_file, f)


def init(opts):
    # cv2.namedWindow('test', cv2.WINDOW_NORMAL)
    os.makedirs(opts.save_dir, exist_ok=True)
    json_save_path = os.path.join(opts.save_dir, "total.json")  # json save path(all pic)
    excel_save_path = os.path.join(opts.save_dir, 'eval.xlsx')
    json_head = make_json_head()
    json_total = deepcopy(json_head)

    image_id = opts.start_id
    hand_id = 1
    bbox_factor = 1.5

    return excel_save_path, json_save_path, json_head, json_total, image_id, hand_id, bbox_factor


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


if __name__ == '__main__':
    path = os.path.dirname(os.path.abspath(__file__))
    parser = ArgumentParser()
    parser.add_argument("--images_dir", default=os.path.join(path, r'/workspace/nas-data/test_data'))
    parser.add_argument("--save_dir", default=os.path.join(path, r'/workspace/nas-data/test_data_output/res'))
    parser.add_argument("--gt_dir", default=os.path.join(path, r'/workspace/nas-data/gt'))
    parser.add_argument("--model_name", default=r'Openpose')
    parser.add_argument("--start_id", default=1_400_000)
    opts = parser.parse_args()

    os.makedirs(opts.save_dir, exist_ok=True)

    main(opts)
