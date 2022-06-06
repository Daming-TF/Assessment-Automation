import json
import cv2
import os
from argparse import ArgumentParser
from copy import deepcopy

from anaLib.models.mediapipe_hand import Mediapipe
from anaLib.EvalTools import EvaluationIntegration, WriteExcel
from anaLib.json_tool import make_json_head, convert_coco_format
from anaLib.my_drawer import Mydrawer


def main(opts):
    image_divide_ids = ['000001402094.jpg', '000001404081.jpg', '000001404682.jpg', '000001410139.jpg',
                        '000001410905.jpg', '000001412448.jpg', '000001414397.jpg', '000001416186.jpg',
                        '000001419204.jpg', '000001423669.jpg']
    video_ids = ['hand_test_01', 'hand_test_02', 'hand_test_03', 'hand_test_04', 'hand_test_05', 'hand_test_06',
                 'hand_test_07', 'hand_test_08', 'hand_test_09', 'hand_test_10']

    excel_save_path, json_save_path, json_head, json_total, image_id, hand_coco_id, bbox_factor = init(opts)

    # init
    mp = Mediapipe()
    my_eval = EvaluationIntegration()
    writer = WriteExcel(excel_save_path, opts.model_name)
    drawer = Mydrawer()

    # init
    json_file = deepcopy(json_head)

    # process
    video_index = 0
    images_names = os.listdir(opts.images_dir)
    for index, images_name in enumerate(images_names):
        if not images_name.endswith('.jpg'):
            continue

        image_path = os.path.join(opts.images_dir, images_name)
        image = cv2.imread(image_path)
        height, width = image.shape[:2]

        input_image = cv2.cvtColor(deepcopy(image), cv2.COLOR_RGB2BGR)
        hands_list = mp(input_image, width, height)

        for hand_index, keypoints in enumerate(hands_list):
            convert_coco_format(json_file, json_total, keypoints, image_id, hand_coco_id,
                                hand_index, int(height), int(width), bbox_factor=bbox_factor)
            image = drawer.draw_2d_points(keypoints, image)
            hand_coco_id += 1

        image_id += 1
        cv2.imshow('test', image)
        cv2.waitKey(1)

        if images_name in image_divide_ids:
            video_id = video_ids[video_index]
            # save dt json
            save_path = os.path.join(opts.save_dir, video_id + '.json')
            write_json(save_path, json_file)

            # eval
            gt_path = os.path.join(opts.gt_dir, video_id + '-gt.json')
            res = my_eval.eval(gt_path, save_path)

            # write excel
            write_info = [video_id] + res
            writer(write_info, index)

            video_index += 1

    # total eval
    write_json(json_save_path, json_total)
    gt_path = os.path.join(opts.gt_dir, 'gt.json')
    res = my_eval.eval(gt_path, json_save_path)
    write_info = ["total"] + res
    writer(write_info, index+1)

    writer.close()


def init(opts):
    cv2.namedWindow('test', cv2.WINDOW_NORMAL)
    json_save_path = os.path.join(opts.save_dir, "total.json")  # json save path(all pic)
    excel_save_path = os.path.join(opts.save_dir, 'eval.xlsx')
    json_head = make_json_head()
    json_total = deepcopy(json_head)

    image_id = opts.start_id
    hand_id = 1
    bbox_factor = 1.5

    return excel_save_path, json_save_path, json_head, json_total, image_id, hand_id, bbox_factor


def write_json(save_path, json_file):
    print(f"writing {save_path}......")
    with open(save_path, 'w') as f:
        json.dump(json_file, f)


if __name__ == '__main__':
    path = os.path.dirname(os.path.abspath(__file__))
    parser = ArgumentParser()
    parser.add_argument("--images_dir", default=os.path.join(path, r'E:\test_data\test_data_from_whole_body\images'))
    parser.add_argument("--save_dir", default=os.path.join(path, r'..\outputs\mediapipe-static_img'))
    parser.add_argument("--gt_dir", default=os.path.join(path, r'..\anaLib\gt'))
    parser.add_argument("--model_name", default=r'Mediapipe')
    parser.add_argument("--start_id", default=1_400_000)
    opts = parser.parse_args()

    os.makedirs(opts.save_dir, exist_ok=True)

    main(opts)
