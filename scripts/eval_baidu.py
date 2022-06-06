"""
评估百度API，由于百度API会有限流等反白嫖措施，常见错误包括：
1. 每分钟限流
2. 每个账号仅有5000个免费额度
3. 当一个主机ip申请次数过多，会强制切断远程连接

基于以上问题我做了以下设计：
1. 对于每分钟限流，服务返回的错误码是固定的，只要接收到该错误码，则休眠3s，然后重新跑处理失败的图片，当连续申请服务错误20次时错误退出程序
2. 在模型初始化是，把所有账号密匙以列表方式输入，当免费额度用完后，自动切换账号
3. 每张图片为期保存一个json，当图片指向一个视频最后一帧时，把所有帧对应的json合并转化成一个coco格式的json并且评估写入excel
"""
import json
import cv2
import os
from argparse import ArgumentParser
from copy import deepcopy
from collections import defaultdict

from anaLib.models.baidu_ai import BaiduAi
from anaLib.EvalTools import EvaluationIntegration, WriteExcel
from anaLib.json_tool import make_json_head, convert_coco_format
from anaLib.my_drawer import Mydrawer

app_id_list = [
    '26036066',
    '26036102',
    '26036284',
    '26036398',
    '26036455',
]

api_key_list = [
    'O86AagBPe4FHWM75nmscpzr2',
    'PBhYSDSNiyuwoXheHZ0tSvNM',
    'sHelwb5Cy6UbwhUxPeflitQs',
    'rH7LUt9l0wDASZBayxG042rY',
    'GbtASLvVgGnZxM29gLAap4KZ',
]

secret_key_list = [
    'D2FhOrgyLICf6ORDb2fL8p6vnyBSLEC5',
    'bPpI22KIegGXYidUriS4ZlzDLG5FUdYG',
    'WHPGEeQBVoZOqIXxBGnG45uDya1kRLpr',
    'eHEPhBsqo5k1i1GCxfy0fG7Agl7GjkrK',
    'XiiaFXOOWcQNOp86F2N4G8oTwK5mO23j',
]


def main(opts):
    image_divide_ids = ['000001402094.jpg', '000001404081.jpg', '000001404682.jpg', '000001410139.jpg',
                        '000001410905.jpg', '000001412448.jpg', '000001414397.jpg', '000001416186.jpg',
                        '000001419204.jpg', '000001423669.jpg']
    video_ids = ['hand_test_01', 'hand_test_02', 'hand_test_03', 'hand_test_04', 'hand_test_05', 'hand_test_06',
                 'hand_test_07', 'hand_test_08', 'hand_test_09', 'hand_test_10']

    excel_save_path, json_save_path, unit_json_save_dir, \
    json_head, json_total, image_id, hand_coco_id, bbox_factor = init(opts)
    remove_file(unit_json_save_dir, image_divide_ids)

    # init
    total_list = []
    model = BaiduAi(app_id_list, api_key_list, secret_key_list)
    my_eval = EvaluationIntegration()
    drawer = Mydrawer()

    # process
    start_image_id = 1400000
    last_image_id = 1400000
    video_index = 0
    images_names = os.listdir(opts.image_dir)
    for index, image_name in enumerate(images_names):
        if not image_name.endswith('.jpg'):
            continue

        # If the file exists, it means it has been processed before
        unit_json_save_path = os.path.join(unit_json_save_dir, image_name.replace('.jpg', '.json'))
        if os.path.exists(unit_json_save_path):
            hand_coco_id = (image_id-start_image_id) * 2 + 1
            image_id += 1
            continue

        # Mode run and show out
        print(f"image id:{image_name}")
        image_path = os.path.join(opts.image_dir, image_name)
        image = cv2.imread(image_path)
        height, width = image.shape[:2]

        input_image = deepcopy(image)
        hands_list, image_save_path = model.run(input_image)

        json_file = deepcopy(json_head)

        for hand_index, keypoints in enumerate(hands_list):
            convert_coco_format(json_file, json_total, keypoints, image_id, hand_coco_id,
                                hand_index, int(height), int(width), bbox_factor=bbox_factor)
            image = drawer.draw_2d_points(keypoints, image)
            hand_coco_id += 1

        write_unit_json(unit_json_save_path, json_file)

        image_id += 1
        cv2.imshow('test', image)
        cv2.waitKey(1)
        cv2.imwrite(image_save_path, image)

        # Determine whether the current picture points to the last frame of the video
        if image_name == image_divide_ids[video_index]:
            # save dt json
            video_id = video_ids[video_index]
            save_path = os.path.join(opts.save_dir, video_id + '.json')
            now_image_id = int(image_name.split('.jpg')[0])
            convert_json(save_path, unit_json_save_dir, last_image_id, now_image_id)

            # eval
            gt_path = os.path.join(opts.gt_dir, video_id+'--gt.json')
            res = my_eval.eval(gt_path, save_path)

            # write excel
            video_unit_excel_save_path = os.path.join(opts.save_dir, video_id+'.xlsx')
            writer = WriteExcel(video_unit_excel_save_path, opts.model_name)
            write_info = [video_id]+res
            writer.write(write_info, video_index)
            writer.close()
            total_list.append(write_info)

            # update info
            last_image_id = now_image_id
            video_index += 1

    # total eval
    # save dt json
    end_image_id = image_divide_ids[-1].split('.jpg')[0]
    convert_json(json_save_path, unit_json_save_dir, start_image_id, int(end_image_id))
    # eval
    gt_path = os.path.join(opts.gt_dir, 'gt.json')
    res = my_eval.eval(gt_path, json_save_path)
    # write excel
    writer = WriteExcel(excel_save_path, opts.model_name)
    write_info = ["total"] + res
    total_list.append(write_info)
    writer.write_total(total_list)
    writer.close()


def convert_json(save_path, unit_json_dir, start_id, end_id):
    annotations_dict = defaultdict(list)
    images_dict = {}

    for image_id in range(start_id, end_id+1):
        json_name = str(image_id).zfill(12)+'.json'
        json_path = os.path.join(unit_json_dir, json_name)

        with open(json_path, 'r') as f:
            dataset = json.load(f)

        for ann in dataset['annotations']:
            annotations_dict[ann['image_id']].append(ann)

        for img in dataset['images']:
            images_dict[img['id']] = img

    write_video_json(images_dict, annotations_dict, save_path)


def init(opts):
    cv2.namedWindow('test', cv2.WINDOW_NORMAL)
    json_save_path = os.path.join(opts.save_dir, "total.json")  # json save path(all pic)
    unit_json_save_dir = os.path.join(opts.save_dir, 'anno_res')
    os.makedirs(unit_json_save_dir, exist_ok=True)
    excel_save_path = os.path.join(opts.save_dir, 'total.xlsx')
    json_head = make_json_head()
    json_total = deepcopy(json_head)

    image_id = opts.start_id
    hand_id = 1
    bbox_factor = 1.5

    return excel_save_path, json_save_path, unit_json_save_dir, json_head, json_total, image_id, hand_id, bbox_factor


def write_video_json(images_dict, annotations_dict, save_path):
    json_head = make_json_head()
    images_ids = list(images_dict.keys())
    for image_id in images_ids:
        image_info = images_dict[image_id]
        json_head["images"].append(image_info)

        annotation_info_list = annotations_dict[image_id]
        for annotation_info in annotation_info_list:
            json_head["annotations"].append(annotation_info)

    with open(save_path, 'w')as f:
        json.dump(json_head, f)


def write_unit_json(save_path, json_file):
    with open(save_path, 'w')as f:
        json.dump(json_file, f)


def remove_file(unit_json_save_dir, image_divide_ids):
    for divide_id in image_divide_ids:
        file_path = os.path.join(unit_json_save_dir, divide_id.replace('.jpg', '.json'))
        if os.path.exists(file_path):
            os.remove(file_path)


if __name__ == '__main__':
    path = os.path.dirname(os.path.abspath(__file__))
    parser = ArgumentParser()
    parser.add_argument("--image_dir", default=os.path.join(path, r'E:\test_data\test_data_from_whole_body\images'))
    parser.add_argument("--save_dir", default=os.path.join(path, r'..\outputs\baidu'))
    parser.add_argument("--gt_dir", default=os.path.join(path, r'..\anaLib\gt'))
    parser.add_argument("--model_name", default=r'Baidu')
    parser.add_argument("--start_id", default=1_400_000)
    opts = parser.parse_args()

    os.makedirs(opts.save_dir, exist_ok=True)

    main(opts)
