# 导入 glob 模块，用于查找符合特定规则的文件路径名
import glob
import os

import numpy as np
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from skimage import io, transform
# 从 skimage.util 导入 img_as_ubyte，用于将图像转换为无符号 8 位整型
from skimage.util import img_as_ubyte


class Augmentation:
    """
    数据预处理类，包含数据增强、图像调整、数据生成和结果可视化等功能。
    提供生成训练和测试数据的生成器，以及对生成数据进行后处理的功能。
    """

    def __init__(self):
        """
        初始化方法，用于构造颜色字典，将各类别映射为固定颜色。
        """
        sky = [128, 128, 128]  # 灰色
        building = [128, 0, 0]  # 紫红色
        pole = [192, 192, 128]  # 浅红色
        road = [128, 64, 128]  # 浅紫色
        pavement = [60, 40, 222]  # 天蓝色
        tree = [128, 128, 0]  # 绿棕色
        sign_symbol = [192, 128, 128]  # 浅红色
        fence = [64, 64, 128]  # 午夜蓝
        car = [64, 0, 128]  # 深岩暗蓝灰色
        pedestrian = [64, 64, 0]  # 棕色
        bicyclist = [0, 128, 192]  # 道奇蓝
        unlabelled = [0, 0, 0]  # 纯黑色

        # 将所有颜色组合成一个 numpy 数组，存储到实例变量 color_dict_
        self.color_dict_ = np.array([
            sky, building, pole, road, pavement, tree,
            sign_symbol, fence, car, pedestrian, bicyclist, unlabelled
        ])

    @staticmethod
    def adjust(image, mask, flag_multi_class, num_class):
        """
        调整图像像素值范围和 mask 形式：
        - 对 image 归一化到 [0,1]
        - 对 mask 进行二值化（若只有两类）或 one-hot 编码（若多类）

        :param image: 原始输入图像
        :param mask: ground truth mask 图像
        :param flag_multi_class: 是否为多类别分割任务（True 表示多类，否则为二分类）
        :param num_class: 类别数量
        :return: 归一化后的 image 和处理后的 mask
        """
        # 如果为多类别情况
        if flag_multi_class:
            # 将图像归一化到 [0,1]
            image = image / 255
            # 如果 mask 是 4 维数据，则取第 4 维的第一个通道，否则取第一个通道
            mask = mask[:, :, :, 0] if (len(mask.shape) == 4) else mask[:, :, 0]
            # 创建一个全零的新 mask，其形状为原 mask 加上类别数的维度
            new_mask = np.zeros(mask.shape + (num_class,))
            # 对每个类别进行 one-hot 编码
            for i in range(num_class):
                new_mask[mask == i, i] = 1
            # 将 new_mask 重塑为二维形式，其中第二维为像素个数乘以类别数
            mask = np.reshape(new_mask, (new_mask.shape[0], new_mask.shape[1] * new_mask.shape[2], new_mask.shape[3]))
        # 如果为二分类情况且图像数值范围超过 1（即未归一化）
        elif np.max(image) > 1:
            # 将图像归一化到 [0,1]
            image = image / 255
            # 同样归一化 mask 到 [0,1]
            mask = mask / 255
            # 将大于 0.5 的 mask 像素设为 1
            mask[mask > 0.5] = 1
            # 将小于等于 0.5 的 mask 像素设为 0
            mask[mask <= 0.5] = 0

        # 返回归一化后的图像和 mask
        return image, mask

    def train_generator(self, batch_size, train_path, images_folder, masks_folder, aug_dict,
                        image_color_mode='grayscale', mask_color_mode='grayscale',
                        image_save_prefix='image', mask_save_prefix='mask', flag_multi_class=False,
                        num_class=2, save_to_dir=None, target_size=(256, 256), seed=1):
        """
        生成器方法，用于同时生成图像和 mask 数据，保证两者采用相同的数据增强参数。
        若指定 save_to_dir 参数，则生成的增强图像会被保存到该目录中（便于可视化）。

        :param batch_size: 每批次生成的样本数
        :param train_path: 训练数据根目录
        :param images_folder: 存放原始图像的文件夹名称
        :param masks_folder: 存放 mask 图像的文件夹名称
        :param aug_dict: 数据增强参数字典
        :param image_color_mode: 图像的颜色模式（'grayscale' 或 'rgb'）
        :param mask_color_mode: mask 图像的颜色模式
        :param image_save_prefix: 保存增强图像的前缀名称
        :param mask_save_prefix: 保存增强 mask 的前缀名称
        :param flag_multi_class: 是否为多类别分割任务
        :param num_class: 类别数量
        :param save_to_dir: 保存增强图像的目录（如果为 None，则不保存）
        :param target_size: 图像调整后的目标尺寸
        :param seed: 随机种子，保证图像和 mask 同步增强
        :return: 一个生成器，每次生成一批归一化后的图像和 mask
        """
        # 根据 aug_dict 参数构建图像数据增强生成器
        image_data_generator = ImageDataGenerator(**aug_dict)
        # 根据 aug_dict 参数构建 mask 数据增强生成器
        mask_data_generator = ImageDataGenerator(**aug_dict)
        # 从 train_path 目录中读取图像数据，并进行增强
        image_generator = image_data_generator.flow_from_directory(
            train_path,
            classes=[images_folder],  # 指定图像文件夹名称
            class_mode=None,  # 无类别标签
            color_mode=image_color_mode,  # 图像颜色模式
            target_size=target_size,  # 调整尺寸
            batch_size=batch_size,  # 批次大小
            save_to_dir=save_to_dir,  # 保存增强图像的目录
            save_prefix=image_save_prefix,  # 增强图像保存前缀
            seed=seed  # 随机种子
        )
        # 从 train_path 目录中读取 mask 数据，并进行增强
        mask_generator = mask_data_generator.flow_from_directory(
            train_path,  # 数据根目录
            classes=[masks_folder],  # 指定 mask 文件夹名称
            class_mode=None,  # 无类别标签
            color_mode=mask_color_mode,  # mask 颜色模式
            target_size=target_size,  # 调整尺寸
            batch_size=batch_size,  # 批次大小
            save_to_dir=save_to_dir,  # 保存增强 mask 的目录
            save_prefix=mask_save_prefix,  # 增强 mask 保存前缀
            seed=seed  # 随机种子
        )
        # 使用 zip 将 image_generator 和 mask_generator 同步迭代
        for image, mask in zip(image_generator, mask_generator):
            # 调用 adjust 方法处理图像和 mask，并返回处理结果
            yield self.adjust(image, mask, flag_multi_class, num_class)

    @staticmethod
    def test_generator(test_path, num_image=30, target_size=(256, 256), flag_multi_class=False, as_gray=True):
        """
        测试数据生成器，用于生成符合模型输入要求的测试图像。

        :param test_path: 测试图像所在的目录
        :param num_image: 需要生成的测试图像数量
        :param target_size: 图像调整后的目标尺寸
        :param flag_multi_class: 是否为多类别任务
        :param as_gray: 是否以灰度模式加载图像
        :return: 一个生成器，依次返回归一化且调整尺寸后的测试图像
        """
        # 遍历指定数量的测试图像
        for i in range(num_image):
            # 构造图像文件路径并读取图像，as_gray 指定是否以灰度图像加载
            image = io.imread(os.path.join(test_path, f'{i}.png'), as_gray=as_gray)
            # 将图像归一化到 [0,1]
            image = image / 255
            # 使用 transform.resize 将图像调整到目标尺寸
            image = transform.resize(image, target_size)
            # 如果不是多类任务，则扩展维度，使图像符合 (height, width, 1) 格式
            image = np.reshape(image, image.shape + (1,)) if not flag_multi_class else image
            # 为图像添加 batch 维度，变为 (1, height, width, channels)
            image = np.reshape(image, (1,) + image.shape)
            # 生成处理后的图像
            yield image

    @staticmethod
    def label_visualize(num_class, color_dict, image):
        """
        将 mask 图像转换为彩色图，用不同颜色表示不同类别。

        :param num_class: 类别数量
        :param color_dict: 颜色字典（numpy 数组）
        :param image: 待可视化的 mask 图像
        :return: 归一化后的彩色图像（像素值范围为 [0,1]）
        """
        # 如果图像为 3 维（height, width, channel），取第一个通道
        image = image[:, :, 0] if len(image.shape) == 3 else image
        # 创建一个全零数组，形状为 (height, width, 3)，用于存放彩色图像
        image_out = np.zeros(image.shape + (3,))
        # 遍历每个类别，赋予相应颜色
        for i in range(num_class):
            image_out[image == i] = color_dict[i]
        # 返回归一化后的彩色图像
        return image_out / 255

    def save_result(self, save_path, npy_file, flag_multi_class=False, num_class=2):
        """
        保存预测得到的 mask 结果到指定目录。

        :param save_path: 结果保存目录
        :param npy_file: 存放预测结果的 numpy 数组（每个元素为一个 mask）
        :param flag_multi_class: 是否为多类别任务
        :param num_class: 类别数量
        :return: None
        """
        # 遍历每个预测结果
        for i, item in enumerate(npy_file):
            # 如果为多类别任务，则先将 mask 进行颜色可视化，否则取 mask 的第一个通道
            image = self.label_visualize(num_class, self.color_dict_, item) if flag_multi_class else item[:, :, 0]
            # 保存处理后的图像到指定路径，文件名格式为 "{索引}_predict.png"
            io.imsave(os.path.join(save_path, f"{i}_predict.png"), img_as_ubyte(image))

    def generator_train_npy(self, image_path, mask_path, flag_multi_class=False, num_class=2, image_prefix='image',
                            mask_prefix='mask', image_as_gray=True, mask_as_gray=True):
        """
        从指定目录中加载图像和对应 mask，转换为 numpy 数组。
        当内存不足时不建议使用此方法。

        :param image_path: 图像存放目录
        :param mask_path: mask 存放目录
        :param flag_multi_class: 是否为多类别任务
        :param num_class: 类别数量
        :param image_prefix: 图像文件名前缀
        :param mask_prefix: mask 文件名前缀
        :param image_as_gray: 是否将图像转换为灰度图
        :param mask_as_gray: 是否将 mask 转换为灰度图
        :return: 两个 numpy 数组，分别包含所有图像和 mask 数据
        """
        # 根据前缀构造图像文件的搜索路径
        image_name_arr = glob.glob(os.path.join(image_path, f"{image_prefix}*.png"))
        # 初始化存储图像数据的列表
        image_arr = []
        # 初始化存储 mask 数据的列表
        mask_arr = []
        # 遍历所有找到的图像文件
        for index, item in enumerate(image_name_arr):
            # 读取图像，as_gray 参数决定是否转换为灰度图
            image = io.imread(item, as_gray=image_as_gray)
            # 如果图像为灰度图，则扩展通道维度
            image = np.reshape(image, image.shape + (1,)) if image_as_gray else image
            # 构造对应 mask 的文件路径，并读取 mask
            mask = io.imread(item.replace(image_path, mask_path).replace(image_prefix, mask_prefix),
                             as_gray=mask_as_gray)
            # 如果 mask 为灰度图，则扩展通道维度
            mask = np.reshape(mask, mask.shape + (1,)) if mask_as_gray else mask
            # 调用 adjust 方法对图像和 mask 进行归一化及处理
            image, mask = self.adjust(image, mask, flag_multi_class, num_class)
            # 将处理后的图像添加到列表中
            image_arr.append(image)
            # 将处理后的 mask 添加到列表中
            mask_arr.append(mask)
        # 将图像列表转换为 numpy 数组
        image_arr = np.array(image_arr)
        # 将 mask 列表转换为 numpy 数组
        mask_arr = np.array(mask_arr)
        # 返回生成的图像和 mask 数据
        return image_arr, mask_arr


# 以下为测试代码，仅在直接运行此脚本时执行
if __name__ == '__main__':
    # 创建 Augmentation 类的实例
    aug = Augmentation()
    # 定义训练数据所在路径
    path = '../dataset/train/'
    # 读取一张测试图像（灰度模式）
    image_ = io.imread(path + 'images/0.png', as_gray=True)
    # 读取对应的 mask 图像（灰度模式）
    mask_ = io.imread(path + 'ground_truth/0.png', as_gray=True)
    # 定义是否为多类别任务，此处为二分类，因此设为 False
    multi_class_ = False
    # 定义类别数量，此处为二分类
    num_class_ = 2
    # 调用 adjust 方法对读取的图像和 mask 进行处理
    aug.adjust(image=image_, mask=mask_, flag_multi_class=multi_class_, num_class=num_class_)
