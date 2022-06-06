import os
import PIL.Image as Image

class MergeImages():
    def __init__(self, image_size=128, image_colnum=10):
        self.image_size = image_size    # 每张小图片大小
        self.image_colnum = image_colnum    # 合并成一张图后， 一行有多少张小图
        self.image_rownum = None

    def merge_images(self, image_dir, save_name):
        # 获取图片集地址下的所有图片名称
        image_fullpath_list = self.get_image_list_fullpath(image_dir)
        print("image_fullpath_list", len(image_fullpath_list), image_fullpath_list)

        image_save_path = fr'{os.path.basename(image_dir)+str(save_name)}.jpg'  # 图片转换后的地址

        # image_rownum = 4  # 图片间隔，也就是合并成一张图后，一共有几行
        image_rownum_yu = len(image_fullpath_list) % self.image_colnum
        if image_rownum_yu == 0:
            self.image_rownum = len(image_fullpath_list) // self.image_colnum
        else:
            self.image_rownum = len(image_fullpath_list) // self.image_colnum + 1

        x_list = []
        y_list = []
        for img_file in image_fullpath_list:
            img_x, img_y = self.get_new_img_xy(img_file)
            x_list.append(img_x)
            y_list.append(img_y)

        print("x_list", sorted(x_list))
        print("y_list", sorted(y_list))
        x_new = int(x_list[len(x_list) // 5 * 4])
        y_new = int(x_list[len(y_list) // 5 * 4])
        self.image_compose(image_fullpath_list, image_save_path, x_new, y_new)

    def get_image_list_fullpath(self, dir_path):
        file_name_list = os.listdir(dir_path)
        image_fullpath_list = []
        for file_name_one in file_name_list:
            file_one_path = os.path.join(dir_path, file_name_one)
            if os.path.isfile(file_one_path):
                image_fullpath_list.append(file_one_path)
            else:
                img_path_list = self.get_image_list_fullpath(file_one_path)
                image_fullpath_list.extend(img_path_list)
        return image_fullpath_list

    def get_new_img_xy(self, infile):
        """返回一个图片的宽、高像素"""
        im = Image.open(infile)
        (x, y) = im.size
        lv = round(x / self.image_size, 2) + 0.01
        x_s = x // lv
        y_s = y // lv
        return x_s, y_s

    # 定义图像拼接函数
    def image_compose(self, image_names, image_save_path, x_new, y_new):
        to_image = Image.new('RGB', (self.image_colnum * x_new, self.image_rownum * y_new))  # 创建一个新图
        # 循环遍历，把每张图片按顺序粘贴到对应位置上
        total_num = 0
        for y in range(1, self.image_rownum + 1):
            for x in range(1, self.image_colnum + 1):
                from_image = self.resize_by_width(image_names[self.image_colnum * (y - 1) + x - 1])
                # from_image = Image.open(image_names[image_colnum * (y - 1) + x - 1]).resize((image_size,image_size ), Image.ANTIALIAS)
                to_image.paste(from_image, ((x - 1) * x_new, (y - 1) * y_new))
                total_num += 1
                if total_num == len(image_names):
                    break
        return to_image.save(image_save_path)  # 保存新图

    def resize_by_width(self, infile):
        """按照宽度进行所需比例缩放"""
        im = Image.open(infile)
        (x, y) = im.size
        lv = round(x / self.image_size, 2) + 0.01
        x_s = int(x // lv)
        y_s = int(y // lv)
        # print("x_s", x_s, y_s)
        out = im.resize((x_s, y_s), Image.ANTIALIAS)
        return out


def main():
    test_image_dir = r'G:\test_data\old_data\vedio_images\images'
    m = MergeImages()
    m.merge_images(test_image_dir)


if __name__ == "__main__":
    main()