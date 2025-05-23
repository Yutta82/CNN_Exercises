import argparse
import json
import os
import pickle
import random

import numpy as np
import tensorflow as tf
from PIL import Image
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten
from keras.models import Model, load_model
from matplotlib import pyplot as plt


def load_config():
    """从 JSON 文件加载配置参数，并与命令行参数合并

    Returns:
        argparse.Namespace: 包含所有配置参数的对象
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="中文字符识别参数配置")
    # 添加配置文件路径参数
    parser.add_argument('--config', type=str, default='config.json', help='配置文件路径')
    # 添加运行模式参数
    parser.add_argument('--mode', type=str, choices=['train', 'validation', 'predict'], default='train',
                        help='运行模式，覆盖配置文件中的 mode')
    # 解析命令行参数
    args = parser.parse_args()

    # 读取并加载JSON配置文件
    with open(args.config, 'r') as f:
        config = json.load(f)

    # 将嵌套的配置字典展平为一级结构
    flat_config = {}
    # 遍历配置组的每个部分（如data_params, model_params等）
    for group in config.values():
        flat_config.update(group)  # 合并配置项

    # 将字典转换为命名空间对象方便属性访问
    config_args = argparse.Namespace(**flat_config)

    # 命令行参数优先级高于配置文件
    if args.mode:
        config_args.mode = args.mode

    return config_args


# 加载合并后的配置参数
ARGS = load_config()


class DataIterator:
    """数据迭代器类，用于加载和处理图像数据"""

    def __init__(self, data_dir):
        """初始化数据迭代器

        Args:
            data_dir (str): 数据目录路径
        """
        # 生成字符集截断路径（例如：data_dir + "03755"）
        truncate_path = data_dir + ('%05d' % ARGS.charset_size)
        print(truncate_path)

        # 初始化图像路径列表
        self.image_names = []
        # 遍历数据目录
        for root, _, file_list in os.walk(data_dir):
            # 过滤超出字符集数量的目录
            if root < truncate_path:
                # 拼接完整文件路径
                self.image_names += [os.path.join(root, file_path) for file_path in file_list]

        # 打乱数据顺序
        random.shuffle(self.image_names)

        # 从文件路径提取标签（目录名即为标签）
        self.labels = [int(file_name[len(data_dir):].split(os.sep)[0]) for file_name in self.image_names]

    @property
    def size(self):
        """获取数据集大小

        Returns:
            int: 数据集中的样本数量
        """
        return len(self.labels)

    @staticmethod
    def data_augmentation(images):
        """数据增强处理

        Args:
            images (tf.Tensor): 输入图像张量

        Returns:
            tf.Tensor: 增强后的图像张量
        """
        # 随机上下翻转
        if ARGS.random_flip_up_down:
            images = tf.image.random_flip_up_down(images)
        # 随机亮度调整
        if ARGS.random_brightness:
            images = tf.image.random_brightness(images, max_delta=0.3)
        # 随机对比度调整
        if ARGS.random_contrast:
            images = tf.image.random_contrast(images, 0.8, 1.2)
        return images

    def input_pipeline(self, batch_size, num_epochs=None, aug=False):
        """创建输入数据管道

        Args:
            batch_size (int): 批量大小
            num_epochs (int, optional): 迭代次数
            aug (bool, optional): 是否启用数据增强

        Returns:
            tf.data.Dataset: 配置好的数据集对象
        """

        # 定义数据解析函数
        def parse_function(image_path, label):
            # 读取图像文件
            image_content = tf.io.read_file(image_path)
            # 解码PNG图像
            image = tf.image.decode_png(image_content, channels=1)
            # 转换像素值到[0,1]范围
            image = tf.image.convert_image_dtype(image, tf.float32)
            # 执行数据增强
            if aug:
                image = self.data_augmentation(image)
            # 调整图像尺寸
            image = tf.image.resize(image, [ARGS.image_size, ARGS.image_size])
            return image, label

        # 创建基础数据集
        dataset = tf.data.Dataset.from_tensor_slices((self.image_names, self.labels))
        # 数据预处理（并行）
        dataset = dataset.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
        # 数据混洗
        dataset = dataset.shuffle(buffer_size=10000)
        # 批次生成
        dataset = dataset.batch(batch_size)
        # 重复次数
        dataset = dataset.repeat(num_epochs)
        return dataset


def build_model():
    """构建卷积神经网络模型

    Returns:
        tf.keras.Model: 配置好的Keras模型
    """
    # 输入层（64x64灰度图像）
    inputs = Input(shape=(64, 64, 1), name='image_batch')
    # 第一卷积块
    x = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv1')(inputs)
    x = MaxPooling2D((2, 2), padding='same')(x)
    # 第二卷积块
    x = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv2')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    # 第三卷积块
    x = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv3')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    # 展平层
    x = Flatten()(x)
    # 全连接层
    x = Dense(1024, activation='tanh', name='fc1')(x)
    # Dropout层防止过拟合
    x = Dropout(0.5)(x)
    # 输出层（使用softmax激活）
    outputs = Dense(ARGS.charset_size, activation='softmax', name='fc2')(x)
    return Model(inputs=inputs, outputs=outputs)


def train():
    """训练模型主函数"""
    print('Begin training')
    # 初始化数据迭代器
    train_feeder = DataIterator(data_dir=ARGS.train_data_dir)
    test_feeder = DataIterator(data_dir=ARGS.test_data_dir)

    # 构建并编译模型
    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # 创建数据管道
    train_dataset = train_feeder.input_pipeline(batch_size=128, aug=True)
    test_dataset = test_feeder.input_pipeline(batch_size=128)

    # 训练回调函数配置
    callbacks = [
        # 模型检查点保存
        ModelCheckpoint(
            filepath=os.path.join(ARGS.checkpoint_dir, 'final_model.keras'),
            monitor='accuracy',
            mode='max',  # 选择最高的准确率
            verbose=1,  # 训练过程中输出日志
            save_freq=ARGS.save_steps,
            save_best_only=True,  # 仅保存最优模型
        ),
        # 早停机制
        EarlyStopping(patience=10, monitor='val_loss')
    ]

    # 开始模型训练
    history = model.fit(
        train_dataset,
        epochs=ARGS.epoch,
        # 计算每epoch步数
        steps_per_epoch=train_feeder.size // 128,
        validation_data=test_dataset,
        validation_steps=test_feeder.size // 128,
        callbacks=callbacks  # 在回调时保存模型
    )

    # 获取训练过程中的数据
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs_range = range(1, ARGS.epoch + 1)

    # 绘制损失和准确率曲线
    plt.figure(figsize=(8, 6))

    # 绘制损失曲线
    plt.plot(epochs_range, train_loss, 'r-', label='Train Loss')  # 红色实线
    plt.plot(epochs_range, val_loss, 'r--', label='Validation Loss')  # 红色虚线

    # 绘制准确率曲线
    plt.plot(epochs_range, train_acc, 'b-', label='Train Accuracy')  # 蓝色实线
    plt.plot(epochs_range, val_acc, 'b--', label='Validation Accuracy')  # 蓝色虚线

    # 设置图例、坐标轴标签和标题
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.title('Training & Validation Loss and Accuracy')
    plt.grid(True)
    # 保存 loss accuracy 变化图
    plot_path = os.path.join(ARGS.checkpoint_dir, 'img.png')
    plt.savefig(plot_path)
    print(f"Training curve saved to {plot_path}")
    plt.show()


def validation():
    """模型验证函数"""
    print('Begin validation')
    # 初始化测试数据
    test_feeder = DataIterator(data_dir=ARGS.test_data_dir)
    test_dataset = test_feeder.input_pipeline(batch_size=128)

    # 加载训练好的模型
    model = load_model(os.path.join(ARGS.checkpoint_dir, 'final_model.keras'))
    # 评估模型性能
    results = model.evaluate(test_dataset)
    # 记录评估结果
    print(f'Validation loss: {results[0]}, accuracy: {results[1]}')


def predict():
    """执行推理预测

    Args:
        input_list (list): 图像路径列表

    Returns:
        list: 预测结果列表
    """
    print('Inference')
    # 加载标签字典
    label_dict = get_label_dict()

    # 获取待预测文件列表
    input_list = get_file_list('./input')

    image_set = []
    # 预处理每个图像
    for image in input_list:
        # 打开并转换图像为灰度
        temp_image = Image.open(image).convert('L')
        # 调整图像尺寸
        temp_image = temp_image.resize((ARGS.image_size, ARGS.image_size), Image.Resampling.LANCZOS)
        # 转换为numpy数组并归一化
        temp_image = np.asarray(temp_image) / 255.0
        # 调整维度匹配模型输入
        temp_image = temp_image.reshape([-1, 64, 64, 1])
        image_set.append(temp_image)

    # 加载训练好的模型
    model = load_model(os.path.join(ARGS.checkpoint_dir, 'final_model.keras'))
    # 批量预测
    predictions = model.predict(np.vstack(image_set))
    # 输出预测结果
    for i, pred in enumerate(predictions):
        # 获取top3预测结果
        top3_indices = np.argsort(pred)[-3:][::-1]
        top3_values = pred[top3_indices]
        # 记录原始预测信息
        print(f'Image: {input_list[i]}, '
              f'Top 3 predictions: {top3_indices}, '
              f'Values: {top3_values}')
        # 解析预测结果
        candidate1 = top3_indices[0]
        candidate2 = top3_indices[1]
        candidate3 = top3_indices[2]
        # 输出可读结果
        print(f'[Result] Image: {input_list[i]}, '
              f'Predict: {label_dict[candidate1]} {label_dict[candidate2]} {label_dict[candidate3]}, '
              f'Most Likely: {label_dict[candidate1]}')


def get_label_dict():
    """获取标签字典

    Returns:
        dict: 标签到汉字的映射字典
    """
    with open('./char_dict', 'rb') as f:
        label_dict = pickle.load(f)
    return label_dict


def get_file_list(path):
    """获取目录文件列表

    Args:
        path (str): 目录路径

    Returns:
        list: 排序后的文件路径列表
    """
    list_name = []
    # 遍历目录并排序文件
    files = os.listdir(path)
    files.sort()
    # 生成完整路径
    for file in files:
        file_path = os.path.join(path, file)
        list_name.append(file_path)
    return list_name


def main():
    """程序入口函数"""
    # 根据运行模式调度不同功能
    if ARGS.mode == "train":
        train()
    elif ARGS.mode == 'validation':
        validation()
    elif ARGS.mode == 'inference':
        # 执行推理
        predict()


if __name__ == "__main__":
    main()
