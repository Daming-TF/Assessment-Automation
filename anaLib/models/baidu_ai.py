#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import os.path
import time
import cv2
import numpy as np
import requests
import shutil
import base64

from anaLib.merge_images import MergeImages

image_save_dir = \
    r'D:\huya_AiBase\Project_hand\handpose\Assessment_automation\assessment_automation\outputs\baidu\image_res'

def convert_to_2d(res):
    keypoints_list = []
    hand_num = res['hand_num']
    hand_info_list = res['hand_info']
    assert hand_num==len(hand_info_list)
    for hand_info in hand_info_list:
        keypoints = np.zeros([21, 3])
        for keyponts_index in range(21):
            info = hand_info['hand_parts'][f'{keyponts_index}']
            keypoints[keyponts_index, 0] = info['x']
            keypoints[keyponts_index, 1] = info['y']
            keypoints[keyponts_index, 2] = 2

        keypoints_list.append(keypoints)

    return keypoints_list


class BaiduAi():
    def __init__(self, app_id_list, api_key_list, secret_key_lis):
        # Information obtained from the official website
        self.save_interval = 0
        self.count = 0
        self.index = -1
        self.change_account = 1

        self.app_id_list = app_id_list
        self.api_key_list = api_key_list
        self.secret_key_list = secret_key_lis

        # Request URL data format
        self.token_url = "https://aip.baidubce.com/oauth/2.0/token"
        self.host = ''
        self.access_token = ''

        # 调用手部关键点识别接口
        self.request_url_root = "https://aip.baidubce.com/rest/2.0/image-classify/v1/hand_analysis"
        self.headers = {"Content-Type": "application/x-www-form-urlencoded"}
        self.request_url = None

        self._init()

        # debug
        self.merger = MergeImages(image_size=256)

    def _init(self):
        if self.change_account:
            self.index += 1

            api_key = self.api_key_list[self.index]
            secret_key = self.secret_key_list[self.index]

            self.host = f'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={api_key}&client_secret={secret_key}'
            self.get_response()
            self.request_url = f"{self.request_url_root}?access_token={self.access_token}"

            self.change_account = 0

    def run(self, image):
        self.change_account = 0
        error_count = 0
        error_flag = 0
        while 1:
            self._init()
            if error_flag == 0:
                self.count += 1
            print(f"{self.count}\t{self.app_id_list[self.index]}")
            os.makedirs(image_save_dir, exist_ok=True)
            image_save_path = os.path.join(image_save_dir, f'{self.count}.jpg')
            cv2.imwrite(image_save_path, image)
            image_base64 = self.image2base64(image_save_path)
            data = {
                "image": image_base64,
            }
            response = requests.post(self.request_url, headers=self.headers, data=data)
            res = eval(response.content.decode("UTF-8"))

            if "error_code" not in res.keys():
                break
            else:
                if str(res['error_code']) == '17':
                    print(f"[!] error!\t ==> app id:{self.app_id_list[self.index]}\t"
                          f"{res['error_msg']}")
                    self.change_account = 1
                else:
                    print(res)

                error_flag = 1
                error_count += 1
                time.sleep(5)

            if error_count == 12:
                exit(1)

        # save intermediate process
        if self.count % 500 == 0:
            self.save_interval += 1
            if self.save_interval % 14 == 0:
                self.merger.merge_images(image_save_dir, self.count)
            shutil.rmtree(image_save_dir)

        return convert_to_2d(res), image_save_path

    def image2base64(self, image_path):
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read())

    def get_response(self):
        response = requests.get(self.host)
        if response:
            # print(response.json())
            self.access_token = response.json().get("access_token")
