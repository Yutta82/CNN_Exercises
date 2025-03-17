# 导入 argparse 模块，用于解析命令行参数
import argparse
import random

from matplotlib import pyplot as plt
from skimage import io
import os

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
    # 添加参数 -a / --train，指定训练数据集路径
    ap.add_argument('-a', '--train', required=False, default='dataset/train/', help='path to train data set')
    # 添加参数 -t / --test，指定测试数据集路径
    ap.add_argument('-t', '--test', required=False, default='dataset/test/', help='path to test data set')
    # 添加参数 -v / --val，指定验证数据集路径
    ap.add_argument('-v', '--val', required=False, default='dataset/val/', help='path to val data set')
    # 添加参数 -s / --steps，指定每个 epoch 的步数，类型为 int，默认值为 30
    ap.add_argument('-s', '--steps', required=False, type=int, default=30, help='steps per epoch for train')
    # 添加参数 -e / --epochs，指定训练轮数，类型为 int，默认值为 5
    ap.add_argument('-e', '--epochs', required=False, type=int, default=60, help='epochs for train model')
    # 添加参数 -m / --mode，指定运行方式
    ap.add_argument('-m', '--mode', required=False, choices=['train', 'predict'], default='train', help='model type')
    # 添加参数 -n / --number，指定预测图片的数目，仅在预测时使用
    ap.add_argument('-n', '--number', required=False, type=int, default=5, help='number of images')
    # 解析命令行参数，并将解析结果转换为字典格式
    args = vars(ap.parse_args())
    # 返回参数字典
    return args


def train(args):
    """
    模型训练函数。
    使用数据增强生成器生成训练数据，并在 MirroredStrategy 下训练 U-Net 模型，
    训练过程中使用 ModelCheckpoint 回调保存训练损失最低的模型。

    :return: None
    """
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
        batch_size=1,  # 每个批次样本数量为 2
        train_path=args['train'],  # 训练数据根目录（注意：此处传入的是列表，需确保目录结构匹配）
        images_folder='images',  # 存放原始图像的文件夹名称
        masks_folder='labels',  # 存放 mask 图像的文件夹名称
        aug_dict=data_gen_args,  # 数据增强参数字典
        save_to_dir=None  # 不保存增强后的图像
    )

    # 验证数据生成器
    val_generator = aug.train_generator(
        batch_size=1,
        train_path=args['val'],
        images_folder='images',
        masks_folder='labels',
        aug_dict={},  # 验证集不进行数据增强
        save_to_dir=None
    )

    model = u_net()
    # 创建 ModelCheckpoint 回调，用于在训练过程中保存损失最低的模型
    model_checkpoint = ModelCheckpoint(
        filepath='checkpoint/final_model.keras',  # 模型保存路径
        monitor='loss',  # 监控训练损失
        verbose=1,  # 输出详细信息
        save_best_only=True  # 仅保存最佳模型
    )
    # 使用生成器开始模型训练
    history = model.fit(
        generator,  # 训练数据生成器
        steps_per_epoch=args['steps'],  # 每个 epoch 的步数
        epochs=args['epochs'],  # 训练的总轮数
        validation_data=val_generator,
        validation_steps=6,
        callbacks=[model_checkpoint]  # 使用回调函数保存最佳模型
    )

    # 绘制 train loss、val loss、train accuracy 和 val accuracy 曲线
    # 获取训练过程中的数据
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs_range = range(1, args['epochs'] + 1)
    # 绘制损失和准确率曲线
    plt.figure(figsize=(8, 6))
    # 绘制损失曲线
    plt.plot(epochs_range, train_loss, 'r-', label='Train Loss')  # 红色实线
    plt.plot(epochs_range, val_loss, 'r--', label='Validation Loss')  # 红色虚线
    # 绘制准确率曲线
    plt.plot(epochs_range, train_acc, 'b-', label='Train Accuracy')  # 蓝色实线
    plt.plot(epochs_range, val_acc, 'b--', label='Validation Accuracy')  # 蓝色虚线
    plt.title('Loss and Accuracy per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(r"checkpoint/", 'img.png')
    plt.savefig(plot_path)
    plt.show()


def predict(args):
    """
    模型预测函数。
    加载训练好的模型，并使用测试生成器生成测试数据进行预测，
    最后调用 Augmentation.save_result() 将预测结果保存到指定目录。

    :return: None
    """
    aug = Augmentation()
    model = load_model('checkpoint/final_model.keras')
    # 获取测试目录中的图像文件列表
    test_path = args['test']
    num_images_to_show = args['number']
    num_images_to_show = 5 if num_images_to_show > 30 else num_images_to_show
    num = sorted(random.sample(range(0, 30), num_images_to_show))
    test_generator_ = aug.test_generator(test_path, num=num)
    results = model.predict(test_generator_, num_images_to_show, verbose=1)
    # 保存预测结果到指定目录
    aug.save_result(r"dataset/test/results", results, num=num)

    # 展示指定数量的原始图像和分割结果
    for i in num:
        # 加载原始图像
        original_image = io.imread(os.path.join(test_path, f"{i}.png"))
        # 加载预测的分割结果
        predicted_mask = io.imread(os.path.join(r"dataset/test/results", f"{i}_predict.png"))

        # 创建一个新的图像窗口
        plt.figure(figsize=(10, 5))

        # 展示原始图像
        plt.subplot(1, 2, 1)
        plt.imshow(original_image, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')

        # 展示分割结果
        plt.subplot(1, 2, 2)
        plt.imshow(predicted_mask, cmap='gray')
        plt.title('Predicted Mask')
        plt.axis('off')

        # 显示图像
        plt.show()