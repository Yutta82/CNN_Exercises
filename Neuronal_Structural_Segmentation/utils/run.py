# 导入 argparse 模块，用于解析命令行参数
import argparse

from keras.callbacks import ModelCheckpoint
# 从 keras.models 模块导入 load_model，用于加载已训练的模型
from keras.models import load_model

from Neuronal_Structural_Segmentation.utils.augmentation import Augmentation
from Neuronal_Structural_Segmentation.utils.model import *


def build_args():
    """
    构造命令行参数解析器并返回参数字典。

    :return: 参数字典，包含训练数据集、测试数据集、steps 和 epochs 等参数
    """
    # 创建 ArgumentParser 对象，用于解析命令行参数
    ap = argparse.ArgumentParser()
    # 添加参数 -a / --train，指定训练数据集路径，默认值为 'dataset/train/'
    ap.add_argument('-a', '--train', required=False, default='dataset/train/', help='path to train data set')
    # 添加参数 -t / --test，指定测试数据集路径，默认值为 'dataset/test/'
    ap.add_argument('-t', '--test', required=False, default='dataset/test/', help='path to test data set')
    # 添加参数 -s / --steps，指定每个 epoch 的步数，类型为 int，默认值为 10
    ap.add_argument('-s', '--steps', required=False, type=int, default=10, help='steps per epoch for train')
    # 添加参数 -e / --epochs，指定训练轮数，类型为 int，默认值为 5
    ap.add_argument('-e', '--epochs', required=False, type=int, default=5, help='epochs for train model')
    # 解析命令行参数，并将解析结果转换为字典格式
    args = vars(ap.parse_args())
    # 返回参数字典
    return args


def train():
    """
    模型训练函数。
    使用数据增强生成器生成训练数据，并在 MirroredStrategy 下训练 U-Net 模型，
    训练过程中使用 ModelCheckpoint 回调保存训练损失最低的模型。

    :return: None
    """
    # 调用 build_args() 函数获取命令行参数字典
    args = build_args()
    # 定义数据增强参数字典
    data_gen_args = dict(
        rotation_range=0.2,  # 随机旋转范围
        width_shift_range=0.05,  # 随机水平平移范围
        height_shift_range=0.05,  # 随机垂直平移范围
        shear_range=0.05,  # 随机剪切变换范围
        zoom_range=0.05,  # 随机缩放范围
        horizontal_flip=True,  # 随机水平翻转
        fill_mode='nearest'  # 填充模式
    )
    # 创建 Augmentation 类实例，用于生成增强数据
    aug = Augmentation()
    # 生成训练数据生成器，批次大小为 2
    generator = aug.train_generator(
        batch_size=2,  # 每个批次样本数量为 2
        train_path=['train'],  # 训练数据根目录（注意：此处传入的是列表，需确保目录结构匹配）
        images_folder='images',  # 存放原始图像的文件夹名称
        masks_folder='ground_truth',  # 存放 mask 图像的文件夹名称
        aug_dict=data_gen_args,  # 数据增强参数字典
        save_to_dir=None  # 不保存增强后的图像
    )

    model = u_net()
    # 创建 ModelCheckpoint 回调，用于在训练过程中保存损失最低的模型
    model_checkpoint = ModelCheckpoint(
        filepath='../checkpoint/final_model.keras',  # 模型保存路径
        monitor='loss',  # 监控训练损失
        verbose=1,  # 输出详细信息
        save_best_only=True  # 仅保存最佳模型
    )
    # 使用生成器开始模型训练
    model.fit(
        generator,  # 训练数据生成器
        steps_per_epoch=args['steps'],  # 每个 epoch 的步数
        epochs=args['epochs'],  # 训练的总轮数
        callbacks=[model_checkpoint]  # 使用回调函数保存最佳模型
    )


def predict():
    """
    模型预测函数。
    加载训练好的模型，并使用测试生成器生成测试数据进行预测，
    最后调用 Augmentation.save_result() 将预测结果保存到指定目录。

    :return: None
    """
    # 调用 build_args() 函数获取命令行参数字典
    args = build_args()
    # 创建 Augmentation 类实例，用于生成测试数据和保存预测结果
    aug = Augmentation()
    # 加载训练过程中保存的最佳模型
    model = load_model('../checkpoint/final_model.keras')
    # 生成测试数据生成器，默认生成 30 个测试样本
    test_generator_ = aug.test_generator(args['test'])
    # 使用模型对测试数据进行预测，verbose=1 输出详细预测进度
    results = model.predict(test_generator_, 30, verbose=1)
    # 保存预测结果到指定目录
    aug.save_result(args['test'], results)
