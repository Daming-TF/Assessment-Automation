import cv2
from copy import deepcopy

from anaLib.models.baidu_ai import BaiduAi
from anaLib.my_drawer import Mydrawer

image_path = r'E:\test_data\test_data_from_whole_body\images\000001400730.jpg'


def main():
    app_id_list = ['26036455']
    api_key_list = [
        'GbtASLvVgGnZxM29gLAap4KZ',
    ]
    secret_key_list = [
        'XiiaFXOOWcQNOp86F2N4G8oTwK5mO23j',
    ]

    model = BaiduAi(app_id_list, api_key_list, secret_key_list)
    drawer = Mydrawer()
    image = cv2.imread(image_path)

    flag = 0

    for app_id, api_key, secret_key in zip(app_id_list, api_key_list, secret_key_list):
        print(f"---------{flag}----------------")
        print(f"app id:{app_id}, api_key:{api_key}, secret_key:{secret_key}")
        debug_image = deepcopy(image)
        hands_list, _ = model.run(debug_image)
        print(hands_list)

        flag += 1

        for hand_index, keypoints in enumerate(hands_list):
            debug_image = drawer.draw_2d_points(keypoints, debug_image)

        cv2.imshow('test', debug_image)
        cv2.waitKey(0)


if __name__ == '__main__':
    main()
