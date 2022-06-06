from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
from collections import OrderedDict
import logging
import pylab,json
import matplotlib.pyplot as plt
import os

logger = logging.getLogger(__name__)
full_arch_name = 'pose_hrnet'


#  coco_results
def evaluate_predictions_on_coco(coco_gt, coco_dt, iou_type='keypoints'):
    # coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.params.useSegm = None     # params里面的imageid表示gt的image_id
    kpt_oks_sigmas = np.ones(21) * 0.35 / 10.0
    coco_eval.params.kpt_oks_sigmas = kpt_oks_sigmas

    # Run per image evaluation on given images and store results (a list of dict) in self.evalImgs.
    # 对给定图像运行每个图像评估并将结果（字典列表）存储在 self.evalImgs
    coco_eval.evaluate()
    # Accumulate per image evaluation results and store the result in self.eval.
    # 累积每个图像评估结果并将结果存储在 self.eval 中。
    coco_eval.accumulate()
    # Compute and display summary metrics for evaluation results.
    # 计算并显示评估结果的汇总指标。
    coco_eval.summarize()

    # [ 1.  1.  1. -1.  1.  1.  1.  1. -1.  1.]
    res = summarizeKps(coco_eval.params, coco_eval.eval)
    print(res)

    # precision - [TxRxKxAxM] precision for every evaluation setting
    # T: iouThrs - [.5:.05:.95] T=10 IoU thresholds for evaluation
    # R: recThrs - [0:.01:1] R=101 recall thresholds for evaluation
    # K:catIds - [all] K cat ids to use for evaluation
    # A: areaRng - [...] A=4 object area ranges for evaluation
    # M: maxDets - [1 10 100] M=3 thresholds on max detections per image
    pr_array1 = coco_eval.eval['precision'][0, :, 0, 0, 0]
    pr_array2 = coco_eval.eval['precision'][5, :, 0, 0, 0]
    pr_array3 = coco_eval.eval['precision'][9, :, 0, 0, 0]
    x = np.arange(0.0, 1.01, 0.01)

    plt.title('coco_eval')
    plt.xlabel('recall')
    plt.ylabel('precision')
    # 调整x,y坐标范围
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.01)
    # 显示背景的网格线
    plt.grid(True)

    # plot([x], y, [fmt], data=None, **kwargs)
    # 可选参数[fmt] = '[color][marker][line]'
    # 是一个字符串来定义图的基本属性如：颜色（color），点型（marker），线型（linestyle）,具体形式
    plt.plot(x, pr_array1, 'b-', label='IoU=0.5')
    plt.plot(x, pr_array2, 'c-', label='IoU=0.75')
    plt.plot(x, pr_array3, 'y-', label='IoU=0.95')

    # plt.legend()函数的作用是给图像加图例
    plt.legend(loc="lower right")
    # plt.savefig(r'D:\HuYa\hand_project\Assessment_automation\output\PR_eval.png')
    plt.show()


def summarize(params, coco_eval, ap=1, iouThr=None, areaRng='all', maxDets=100):
    p = params
    iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
    titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
    typeStr = '(AP)' if ap == 1 else '(AR)'
    iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
        if iouThr is None else '{:0.2f}'.format(iouThr)

    aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
    mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
    if ap == 1:
        # dimension of precision: [TxRxKxAxM]
        s = coco_eval['precision']
        # IoU
        if iouThr is not None:
            t = np.where(iouThr == p.iouThrs)[0]
            s = s[t]
        s = s[:, :, :, aind, mind]
    else:
        # dimension of recall: [TxKxAxM]
        s = coco_eval['recall']
        if iouThr is not None:
            t = np.where(iouThr == p.iouThrs)[0]
            s = s[t]
        s = s[:, :, aind, mind]
    if len(s[s > -1]) == 0:
        mean_s = -1
    else:
        mean_s = np.mean(s[s > -1])
    print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
    return mean_s


def summarizeKps(params, eval):
    stats = np.zeros((10,))
    stats[0] = summarize(params, eval, 1, maxDets=20)
    stats[1] = summarize(params, eval, 1, maxDets=20, iouThr=.5)
    stats[2] = summarize(params, eval, 1, maxDets=20, iouThr=.75)
    stats[3] = summarize(params, eval, 1, maxDets=20, areaRng='medium')
    stats[4] = summarize(params, eval, 1, maxDets=20, areaRng='large')
    stats[5] = summarize(params, eval, 0, maxDets=20)
    stats[6] = summarize(params, eval, 0, maxDets=20, iouThr=.5)
    stats[7] = summarize(params, eval, 0, maxDets=20, iouThr=.75)
    stats[8] = summarize(params, eval, 0, maxDets=20, areaRng='medium')
    stats[9] = summarize(params, eval, 0, maxDets=20, areaRng='large')
    return stats

if __name__ == "__main__":
    path = os.path.dirname(os.path.abspath(__file__))
    gt_path = os.path.join(path, r"..\anaLib\gt\hand_test_04--gt.json") # 存放真实标签的路径
    dt_path = os.path.join(path, r"..\outputs\mediapipe-lite\annotations\hand_test_04.json")    # 存放检测结果的路径

    cocoGt = COCO(gt_path)
    cocoDt = COCO(dt_path)
    evaluate_predictions_on_coco(cocoGt, cocoDt)
