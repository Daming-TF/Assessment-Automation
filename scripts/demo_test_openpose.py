# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse


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


try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/../../python/openpose/Release');
            os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
            import pyopenpose as op
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('../../python');
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            # sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="../../../examples/media/COCO_val2014_000000000241.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "../../../models/"
    params["hand"] = True
    params["hand_detector"] = 2
    params["body"] = 0

    # Add others in path?
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1])-1: next_item = args[1][i+1]
        else: next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-','')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-','')
            if key not in params: params[key] = next_item

    # Construct it from system arguments
    # op.init_argv(args[1])
    # oppython = op.OpenposePython()

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Read image and face rectangle locations
    imageToProcess = cv2.imread(args[0].image_path)
    handRectangles = [
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
    datum = op.Datum()
    datum.cvInputData = imageToProcess
    datum.handRectangles = handRectangles

    # Process and display image
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))
    print("Left hand keypoints: \n" + str(datum.handKeypoints[0]))
    print("Right hand keypoints: \n" + str(datum.handKeypoints[1]))

    image = datum.cvOutputData

    cv2.rectangle(image, (320, 377), (320 + 69, 377 + 69), (0,0,255), 2)
    cv2.rectangle(image, (80, 407), (80 + 80, 407 + 80), (0,0,255), 2)
    cv2.rectangle(image, (46, 404), (46 + 99, 404 + 99), (0,0,255), 2)
    cv2.rectangle(image, (185, 303), (185 + 157, 303 + 157), (0,0,255), 2)
    cv2.rectangle(image, (88, 268), (88 + 117, 268 + 117), (0,0,255), 2)

    # cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
    cv2.imwrite(r'/workspace/nas-data/test_data_output/test.jpg', image)
    print("Success to save pic")
    # cv2.waitKey(0)
except Exception as e:
    print(e)
    sys.exit(-1)
