import os

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import xlsxwriter as xw


class EvaluationIntegration:
    def __init__(self):
        self.coco_eval = None

    def eval(self, gt_path, dt_path, iou_type='keypoints'):
        coco_gt = COCO(gt_path)
        coco_dt = COCO(dt_path)

        coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
        coco_eval.params.useSegm = None
        kpt_oks_sigmas = np.ones(21) * 0.35 / 10.0
        coco_eval.params.kpt_oks_sigmas = kpt_oks_sigmas

        print(f"_____________________>>{os.path.basename(gt_path)}<<_____________________")
        # Run per image evaluation on given images and store results (a list of dict) in self.evalImgs.
        coco_eval.evaluate()
        # Accumulate per image evaluation results and store the result in self.eval.
        coco_eval.accumulate()
        # Compute and display summary metrics for evaluation results.
        coco_eval.summarize()

        self.coco_eval = coco_eval

        return self.summarize_kps()

    def summarize_kps(self):
        stats = np.zeros((10,))
        stats[0] = self.summarize(1, maxDets=20)
        stats[1] = self.summarize(1, maxDets=20, iouThr=.5)
        stats[2] = self.summarize(1, maxDets=20, iouThr=.75)
        stats[3] = self.summarize(1, maxDets=20, areaRng='medium')
        stats[4] = self.summarize(1, maxDets=20, areaRng='large')
        stats[5] = self.summarize(0, maxDets=20)
        stats[6] = self.summarize(0, maxDets=20, iouThr=.5)
        stats[7] = self.summarize(0, maxDets=20, iouThr=.75)
        stats[8] = self.summarize(0, maxDets=20, areaRng='medium')
        stats[9] = self.summarize(0, maxDets=20, areaRng='large')
        return stats.tolist()

    def summarize(self, ap=1, iouThr=None, areaRng='all', maxDets=100):
        p = self.coco_eval.params
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap == 1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = self.coco_eval.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:, :, :, aind, mind]
        else:
            # dimension of recall: [TxKxAxM]
            s = self.coco_eval.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:, :, aind, mind]
        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])
        return mean_s


class WriteExcel:
    def __init__(self, save_path, mode):
        self.mode = mode
        self.workbook = xw.Workbook(save_path)  # 创建工作簿
        self.worksheet = self.writer_title()
        self.bold = self.creat_bold()

    def write(self, data_list, i):
        print(f"writing")
        self.worksheet.write_row(f'B{4+i}', data_list, self.bold)  # 从A1单元格开始写入表头
        print("Success!")

    def write_total(self, info_list):
        for i, info in enumerate(info_list):
            self.worksheet.write_row(f'B{4 + i}', info, self.bold)  # 从A1单元格开始写入表头

    def close(self):
        self.workbook.close()

    def writer_title(self):
        worksheet1 = self.workbook.add_worksheet("sheet1")  # 创建子表
        worksheet1.activate()  # 激活表

        bold = self.workbook.add_format({
            'bold': True,  # 字体加粗
            'border': 3,  # 单元格边框宽度
            'align': 'center',  # 水平对齐方式
            'valign': 'vcenter',  # 垂直对齐方式
            'fg_color': '#F4B084',  # 单元格背景颜色
            'text_wrap': True,  # 是否自动换行
        })

        worksheet1.set_column('A:L', 15)
        iou_list = ['0.50:0.95', '0.50', '0.75']

        worksheet1.merge_range('A1:A3', 'Model', bold)
        worksheet1.merge_range('B1:B3', 'Video_name', bold)
        worksheet1.merge_range('C1:G1', 'AP', bold)
        worksheet1.merge_range('H1:L1', 'AR', bold)

        worksheet1.merge_range('C2:E2', 'Area=all', bold)
        worksheet1.write('F2', 'Area=medium', bold)
        worksheet1.write('G2', 'Area=large', bold)

        worksheet1.merge_range('H2:J2', 'Area=all', bold)
        worksheet1.write('K2', 'Area=medium', bold)
        worksheet1.write('L2', 'Area=large', bold)

        worksheet1.write_row('C3', iou_list, bold)
        worksheet1.write('F3', '0.50:0.95', bold)
        worksheet1.write('G3', '0.50:0.95', bold)
        worksheet1.write_row('H3', iou_list, bold)
        worksheet1.write('K3', '0.50:0.95', bold)
        worksheet1.write('L3', '0.50:0.95', bold)

        worksheet1.merge_range('A4:A14', self.mode, bold)

        return worksheet1

    def creat_bold(self):
        bold = self.workbook.add_format({
            'bold': True,  # 字体加粗
            'align': 'center',  # 水平对齐方式
        })
        return bold
