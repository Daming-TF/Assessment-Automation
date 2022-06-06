import xlsxwriter as xw


def main(data, fileName):  # xlsxwriter库储存数据到excel
    workbook = xw.Workbook(fileName)  # 创建工作簿

    worksheet = writer_title(workbook)

    bold = workbook.add_format({
        'bold': True,  # 字体加粗
        'align': 'center',  # 水平对齐方式
    })
    worksheet.write_row('A4', list(data.values()), bold)  # 从A1单元格开始写入表头

    workbook.close()


def writer_title(workbook):
    worksheet1 = workbook.add_worksheet("sheet1")  # 创建子表
    worksheet1.activate()  # 激活表

    bold = workbook.add_format({
        'bold': True,  # 字体加粗
        'border': 3,  # 单元格边框宽度
        'align': 'center',  # 水平对齐方式
        'valign': 'vcenter',  # 垂直对齐方式
        'fg_color': '#F4B084',  # 单元格背景颜色
        'text_wrap': True,  # 是否自动换行
    })
    worksheet1.set_column('A:L', 15)
    iou_list = ['0.50', '0.75', '0.50:0.95']
    worksheet1.merge_range('A1:A3', 'Index', bold)
    worksheet1.merge_range('B1:B3', 'Model', bold)
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

    return worksheet1


if __name__ == '__main__':
    # "-------------数据用例-------------"
    # testData = [
    #     {"id": 4, "name": "立智", "price": 100},
    #     {"id": 5, "name": "维纳", "price": 200},
    #     {"id": 3, "name": "如家", "price": 300},
    # ]
    testData = {
            "id": 1,
            "name": "Mediapipe",
            "AP-all-0.50": 0.9,
            "AP-all-0.75": 0.8,
            "AP-all-0.50:0.95": 0.78,
            "AP-medium-0.50:0.95": -1,
            "AP-large-0.50:0.95": 0.88,
            "AR-all-0.50": 0.91,
            "AR-all-0.75": 0.81,
            "AR-all-0.50:0.95": 0.79,
            "AR-medium-0.50:0.95": -1,
            "AR-large-0.50:0.95": 0.89,
    }
    fileName = r'D:\HuYa\hand_project\Assessment_automation\output\测试.xlsx'
    main(testData, fileName)
