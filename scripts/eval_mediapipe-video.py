import json
import cv2
import os
from argparse import ArgumentParser
from copy import deepcopy
import numpy as np

from anaLib.models.mediapipe_hand import Mediapipe
from anaLib.EvalTools import EvaluationIntegration, WriteExcel
from anaLib.json_tool import make_json_head, convert_coco_format, load_json_data, get_ids
from anaLib.my_drawer import Mydrawer
from anaLib.video_writer import VideoWriter


def main(opts):
    json_head, json_total, image_id, hand_coco_id, bbox_factor = init(opts)
    total_gt_path = os.path.join(opts.gt_dir, 'gt.json')
    match_ids, annotations_dict = get_gt(total_gt_path)

    # make path with correct format
    json_save_dir = os.path.join(opts.save_dir, "annotations")  # json save path(all pic)
    os.makedirs(json_save_dir, exist_ok=True)
    excel_save_dir = os.path.join(opts.save_dir, 'excel')
    os.makedirs(excel_save_dir, exist_ok=True)
    video_save_dir = os.path.join(opts.save_dir, 'output-video')
    os.makedirs(video_save_dir, exist_ok=True)
    gt_save_dir = os.path.join(opts.save_dir, 'gt-video')
    os.makedirs(gt_save_dir, exist_ok=True)

    # total init
    mp = Mediapipe(0)
    json_save_path = os.path.join(json_save_dir, 'total.json')
    excel_save_path = os.path.join(excel_save_dir, 'total.xlsx')
    excel_writer = WriteExcel(excel_save_path, opts.model_name)
    my_eval = EvaluationIntegration()
    drawer = Mydrawer()

    # process
    video_names = os.listdir(opts.video_dir)
    for index, video_name in enumerate(video_names):
        if not video_name.endswith('.mp4'):
            continue

        # Video unit initialization
        json_file = deepcopy(json_head)
        video_id = video_name.split('.mp4')[0]

        video_path = os.path.join(opts.video_dir, video_name)
        json_unit_save_path = os.path.join(json_save_dir, video_id + '.json')
        video_save_path = os.path.join(video_save_dir, video_id + '.mp4')
        gt_save_path = os.path.join(gt_save_dir, video_id + '.mp4')

        cap = cv2.VideoCapture(video_path)
        width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # Instantiate video obj
        video_writer = VideoWriter(video_save_path, cap)
        gt_video_writer = VideoWriter(gt_save_path, cap) if opts.save_gt_flag else None

        # process video
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break

            input_image = cv2.cvtColor(deepcopy(image), cv2.COLOR_RGB2BGR)
            hands_list = mp(input_image, width, height)

            mp_image = deepcopy(image)
            for hand_index, keypoints in enumerate(hands_list):
                convert_coco_format(json_file, json_total, keypoints, image_id, hand_coco_id,
                                    hand_index, int(height), int(width), bbox_factor=bbox_factor)
                mp_image = drawer.draw_2d_points(keypoints, mp_image)
                hand_coco_id += 1

            # draw gt data
            gt_keypoints_list = annotations_dict[image_id]
            gt_image = drawer.draw_image(deepcopy(image), gt_keypoints_list)
            cv2.putText(gt_image, 'GT', (80, 80), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 4)

            # show output
            canves = np.hstack([gt_image, mp_image])
            cv2.imshow('test', canves)
            cv2.waitKey(1)

            if image_id not in match_ids:
                image_id += 1
                continue

            gt_video_writer.write(gt_image) if opts.save_gt_flag else None
            video_writer.write(mp_image)
            image_id += 1

        # release resources
        cap.release()
        video_writer.release()
        gt_video_writer.release() if opts.save_gt_flag else None

        # save dt json
        write_json(json_unit_save_path, json_file)

        # eval
        gt_path = os.path.join(opts.gt_dir, video_id+'--gt.json')
        res = my_eval.eval(gt_path, json_unit_save_path)

        # write excel
        write_info = [video_id]+res
        excel_writer.write(write_info, index)

    # total eval
    write_json(json_save_path, json_total)
    res = my_eval.eval(total_gt_path, json_save_path)

    write_info = ["total"] + res
    excel_writer.write(write_info, len(video_names))

    excel_writer.close()


def init(opts):
    cv2.namedWindow('test', cv2.WINDOW_NORMAL)
    json_head = make_json_head()
    json_total = deepcopy(json_head)

    image_id = opts.start_id
    hand_id = 1
    bbox_factor = 1.5

    return json_head, json_total, image_id, hand_id, bbox_factor


def write_json(save_path, json_file):
    print(f"writing {save_path}......")
    with open(save_path, 'w') as f:
        json.dump(json_file, f)


def get_gt(gt_path):
    images_dict, annotations_dict = load_json_data(gt_path)
    return get_ids(images_dict), annotations_dict


if __name__ == '__main__':
    path = os.path.dirname(os.path.abspath(__file__))
    parser = ArgumentParser()
    parser.add_argument("--video_dir", default=os.path.join(path, r'..\videos'))
    parser.add_argument("--save_dir", default=os.path.join(path, r'..\outputs\mediapipe-lite'))
    parser.add_argument("--gt_dir", default=os.path.join(path, r'..\anaLib\gt'))
    parser.add_argument("--model_name", default=r'Mediapipe')
    parser.add_argument("--start_id", default=1_400_000)
    parser.add_argument("--save_gt_flag", default=True)
    opts = parser.parse_args()

    os.makedirs(opts.save_dir, exist_ok=True)

    main(opts)
