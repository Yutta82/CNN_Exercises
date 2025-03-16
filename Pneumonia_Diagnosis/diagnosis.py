# 导入必要的库
import os
from pathlib import Path

import imgaug as aug
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from PIL import Image
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, SeparableConv2D, BatchNormalization
from keras.models import Model, load_model  # 用于构建神经网络模型
from keras.optimizers import Adam  # 用于优化器
from keras.utils import to_categorical  # 用于独热编码
from mlxtend.plotting import plot_confusion_matrix  # 用于绘制混淆矩阵
from skimage.io import imread  # 用于读取图像
from sklearn.metrics import confusion_matrix  # 用于计算混淆矩阵

np.bool = bool  # 为兼容性定义 np.bool

# 设置 matplotlib 的颜色
color = sns.color_palette()

# 打印输入目录中的文件
print(os.listdir("./input"))


def set_random_seed():
    """设置随机种子以确保结果可重复性。

    Args:

    Returns:
        无返回值。
    """
    # 设置 Python 哈希种子
    os.environ['PYTHONHASHSEED'] = '0'
    # 设置 NumPy 随机种子
    np.random.seed(111)
    # 设置 TensorFlow 2.X 随机种子
    tf.random.set_seed(111)
    # 设置图像增强的随机种子
    aug.seed(111)


def load_data(data_dir):
    """加载训练、验证和测试数据。

    Args:
        data_dir (Path): 数据目录路径。

    Returns:
        train_data (pd.DataFrame): 训练数据 DataFrame。
        valid_data (np.ndarray): 验证数据图像数组。
        valid_labels (np.ndarray): 验证数据标签数组。
        test_data (np.ndarray): 测试数据图像数组。
        test_labels (np.ndarray): 测试数据标签数组。
    """
    # 训练数据目录
    train_dir = data_dir / 'train'
    # 验证数据目录
    val_dir = data_dir / 'val'
    # 测试数据目录
    test_dir = data_dir / 'test'

    # 加载训练数据
    normal_cases_dir = train_dir / 'NORMAL'
    pneumonia_cases_dir = train_dir / 'PNEUMONIA'
    normal_cases = normal_cases_dir.glob('*.jpeg')
    pneumonia_cases = pneumonia_cases_dir.glob('*.jpeg')
    train_data = [(img, 0) for img in normal_cases] + [(img, 1) for img in pneumonia_cases]
    train_data = pd.DataFrame(train_data, columns=['image', 'label'], index=None)
    train_data = train_data.sample(frac=1.).reset_index(drop=True)

    # 加载验证数据
    normal_cases_dir = val_dir / 'NORMAL'
    pneumonia_cases_dir = val_dir / 'PNEUMONIA'
    normal_cases = normal_cases_dir.glob('*.jpeg')
    pneumonia_cases = pneumonia_cases_dir.glob('*.jpeg')
    valid_data, valid_labels = process_images(normal_cases, 0)
    valid_data_p, valid_labels_p = process_images(pneumonia_cases, 1)
    valid_data = np.concatenate((valid_data, valid_data_p), axis=0)
    valid_labels = np.concatenate((valid_labels, valid_labels_p), axis=0)

    # 加载测试数据
    normal_cases_dir = test_dir / 'NORMAL'
    pneumonia_cases_dir = test_dir / 'PNEUMONIA'
    normal_cases = normal_cases_dir.glob('*.jpeg')
    pneumonia_cases = pneumonia_cases_dir.glob('*.jpeg')
    test_data, test_labels = process_images(normal_cases, 0)
    test_data_p, test_labels_p = process_images(pneumonia_cases, 1)
    test_data = np.concatenate((test_data, test_data_p), axis=0)
    test_labels = np.concatenate((test_labels, test_labels_p), axis=0)

    return train_data, valid_data, valid_labels, test_data, test_labels


def process_images(image_paths, label):
    """处理图像，包括调整大小、归一化和独热编码标签。

    Args:
        image_paths (generator): 图像文件路径生成器。
        label (int): 图像标签（0: 正常, 1: 肺炎）。

    Returns:
        data (np.ndarray): 处理后的图像数组。
        labels (np.ndarray): 独热编码的标签数组。
    """
    data = []
    labels = []
    for img_path in image_paths:
        img = Image.open(str(img_path)).convert('RGB')
        img = img.resize((224, 224))
        # 转换为 numpy 数组并归一化
        img = np.asarray(img, dtype=np.float32) / 255.0
        encoded_label = to_categorical(label, num_classes=2)
        data.append(img)
        labels.append(encoded_label)
    return np.array(data), np.array(labels)


def visualize_data(train_data):
    """可视化训练数据的分布和样本。

    Args:
        train_data (pd.DataFrame): 训练数据 DataFrame。

    Returns:
        无返回值。
    """
    # 统计每个类别的样本数量
    cases_count = train_data['label'].value_counts()
    print(cases_count)

    # 绘制条形图
    plt.figure(figsize=(10, 8))
    sns.barplot(x=cases_count.index, y=cases_count.values)
    plt.title('Number of cases', fontsize=14)
    plt.xlabel('Case type', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(range(len(cases_count.index)), ['Normal(0)', 'Pneumonia(1)'])
    plt.show()

    # 获取样本图像
    pneumonia_samples = (train_data[train_data['label'] == 1]['image'].iloc[:5]).tolist()
    normal_samples = (train_data[train_data['label'] == 0]['image'].iloc[:5]).tolist()
    samples = pneumonia_samples + normal_samples

    # 绘制样本图像
    f, ax = plt.subplots(2, 5, figsize=(30, 10))
    for i in range(10):
        img = imread(samples[i])
        ax[i // 5, i % 5].imshow(img, cmap='gray')
        if i < 5:
            ax[i // 5, i % 5].set_title("Pneumonia")
        else:
            ax[i // 5, i % 5].set_title("Normal")
        ax[i // 5, i % 5].axis('off')
        ax[i // 5, i % 5].set_aspect('auto')
    plt.show()


def get_augmentation_sequence():
    """定义数据增强序列。

    Returns:
        seq (iaa.OneOf): 图像增强序列。
    """
    return iaa.OneOf([
        iaa.Fliplr(),  # 水平翻转
        iaa.Affine(rotate=20),  # 旋转 20 度
        iaa.Multiply((1.2, 1.5))  # 随机调整亮度
    ])


def data_generator(data, batch_size):
    """数据生成器，用于训练时的批量数据生成和增强。

    Args:
        data (pd.DataFrame): 训练数据 DataFrame。
        batch_size (int): 批量大小。

    Yields:
        batch_data (np.ndarray): 批量图像数据。
        batch_labels (np.ndarray): 批量标签数据。
    """
    n = len(data)  # 数据总数
    steps = n // batch_size  # 每 epoch 的步数
    batch_data = np.zeros((batch_size, 224, 224, 3), dtype=np.float32)  # 批量数据数组
    batch_labels = np.zeros((batch_size, 2), dtype=np.float32)  # 批量标签数组
    indices = np.arange(n)  # 获取输入数据所有索引的 numpy 数组
    i = 0
    seq = get_augmentation_sequence()  # 获取增强序列
    while True:
        np.random.shuffle(indices)  # 打乱索引
        next_batch = indices[(i * batch_size):(i + 1) * batch_size]
        count = 0
        for j, idx in enumerate(next_batch):
            img_name = data.iloc[idx]['image']
            label = data.iloc[idx]['label']
            encoded_label = to_categorical(label, num_classes=2)
            # 读取图像并调整大小
            img = Image.open(img_name).convert('RGB')
            img = img.resize((224, 224))
            # 转换为 numpy 数组并归一化
            orig_img = np.asarray(img, dtype=np.float32) / 255.0
            batch_data[count] = orig_img
            batch_labels[count] = encoded_label
            # 对正常样本进行增强
            if label == 0 and count < batch_size - 2:
                # 增强需要 numpy 数组
                aug_img1 = seq.augment_image(np.asarray(img))
                aug_img2 = seq.augment_image(np.asarray(img))
                aug_img1 = aug_img1.astype(np.float32) / 255.
                aug_img2 = aug_img2.astype(np.float32) / 255.
                batch_data[count + 1] = aug_img1
                batch_labels[count + 1] = encoded_label
                batch_data[count + 2] = aug_img2
                batch_labels[count + 2] = encoded_label
                count += 2
            else:
                count += 1
            if count == batch_size - 1:
                break
        i += 1
        yield batch_data, batch_labels
        if i >= steps:
            i = 0


def build_model():
    """构建 CNN 模型。

    Returns:
        model (Model): Keras 模型。
    """
    input_img = Input(shape=(224, 224, 3), name='ImageInput')
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='Conv1_1')(input_img)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='Conv1_2')(x)
    x = MaxPooling2D((2, 2), name='pool1')(x)
    x = SeparableConv2D(128, (3, 3), activation='relu', padding='same', name='Conv2_1')(x)
    x = SeparableConv2D(128, (3, 3), activation='relu', padding='same', name='Conv2_2')(x)
    x = MaxPooling2D((2, 2), name='pool2')(x)
    x = SeparableConv2D(256, (3, 3), activation='relu', padding='same', name='Conv3_1')(x)
    x = BatchNormalization(name='bn1')(x)
    x = SeparableConv2D(256, (3, 3), activation='relu', padding='same', name='Conv3_2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = SeparableConv2D(256, (3, 3), activation='relu', padding='same', name='Conv3_3')(x)
    x = MaxPooling2D((2, 2), name='pool3')(x)
    x = SeparableConv2D(512, (3, 3), activation='relu', padding='same', name='Conv4_1')(x)
    x = BatchNormalization(name='bn3')(x)
    x = SeparableConv2D(512, (3, 3), activation='relu', padding='same', name='Conv4_2')(x)
    x = BatchNormalization(name='bn4')(x)
    x = SeparableConv2D(512, (3, 3), activation='relu', padding='same', name='Conv4_3')(x)
    x = MaxPooling2D((2, 2), name='pool4')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(1024, activation='relu', name='fc1')(x)
    x = Dropout(0.7, name='dropout1')(x)
    x = Dense(512, activation='relu', name='fc2')(x)
    x = Dropout(0.5, name='dropout2')(x)
    x = Dense(2, activation='softmax', name='fc3')(x)
    model = Model(inputs=input_img, outputs=x)
    return model


def compile_and_train(model, train_data, valid_data, valid_labels, batch_size, epochs):
    """编译和训练模型。

    Args:
        model (Model): Keras 模型。
        train_data (pd.DataFrame): 训练数据 DataFrame。
        valid_data (np.ndarray): 验证数据图像数组。
        valid_labels (np.ndarray): 验证数据标签数组。
        batch_size (int): 批量大小。
        epochs (int): 训练轮数。

    Returns:
        无返回值。
    """
    print("train")
    # Adam 优化器
    opt = Adam(learning_rate=0.0001, decay=1e-5)
    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=opt)
    train_gen = data_generator(train_data, batch_size)
    model_checkpoint = ModelCheckpoint(
        filepath='./checkpoint/final_model.keras',
        monitor='accuracy',
        mode='max',  # 选择最高的准确率
        verbose=1,
        save_best_only=True
    )
    history = model.fit(
        train_gen,
        steps_per_epoch=len(train_data) // batch_size,
        epochs=epochs,
        validation_data=(valid_data, valid_labels),  # 训练时不验证
        callbacks=[model_checkpoint]  # 在回调时保存模型
    )

    # 获取 loss 和 accuracy 数据
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs_range = range(1, epochs + 1)

    # 绘制 loss 和 accuracy 曲线
    plt.figure(figsize=(8, 6))

    # 绘制 loss 相关曲线
    plt.plot(epochs_range, train_loss, 'r-', label='Train Loss')  # 红色实线
    plt.plot(epochs_range, val_loss, 'r--', label='Val Loss')  # 红色虚线

    # 绘制 accuracy 相关曲线
    plt.plot(epochs_range, train_acc, 'b-', label='Train Accuracy')  # 蓝色实线
    plt.plot(epochs_range, val_acc, 'b--', label='Val Accuracy')  # 蓝色虚线

    # 设置图例、标题和坐标轴标签
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.title("Training & Validation Loss and Accuracy")
    plt.grid(True)
    # 保存 loss accuracy 变化图
    plot_path = os.path.join(r"./checkpoint", 'img.png')
    plt.savefig(plot_path)
    print(f"Training curve saved to {plot_path}")
    plt.show()


def validate_model(model, test_data, test_labels):
    """评估模型并可视化结果。

    Args:
        model (Model): Keras 模型。
        test_data (np.ndarray): 测试数据图像数组。
        test_labels (np.ndarray): 测试数据标签数组。

    Returns:
        无返回值。
    """
    print("validate_model")
    test_loss, test_score = model.evaluate(test_data, test_labels, batch_size=16)
    print("测试集损失: ", test_loss)
    print("测试集准确率: ", test_score)

    predicts = model.predict(test_data, batch_size=16)
    predicts = np.argmax(predicts, axis=-1)
    orig_test_labels = np.argmax(test_labels, axis=-1)

    cm = confusion_matrix(orig_test_labels, predicts)
    plt.figure()
    plot_confusion_matrix(
        cm,
        figsize=(12, 8),
        hide_ticks=True,
        fontcolor_threshold=0.7,
        cmap=plt.colormaps.get_cmap("Blues")
    )
    plt.xticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)
    plt.yticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)
    plt.show()

    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print("模型的召回率: {:.2f}".format(recall))
    print("模型的精确率: {:.2f}".format(precision))


def predict():
    """
    使用训练好的模型对输入的胸部X光图像进行预测，并显示结果。

    Args:

    Returns:
        np.ndarray: 预测结果数组，每个预测结果为一个包含各类别预测概率的数组
    """
    print("predict")
    # 定义标签字典
    label_dict = {0: 'Normal', 1: 'Pneumonia'}
    image_set = []
    image_size = 224  # 模型输入尺寸
    input_dir = r"./input"
    input_list = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.jpeg')]

    # 对每张图像进行预处理
    for image_path in input_list:
        # 打开图像并转换为 RGB 模式
        temp_image = Image.open(image_path).convert('RGB')
        # 调整图像尺寸
        temp_image = temp_image.resize((image_size, image_size), Image.Resampling.LANCZOS)
        # 转换为 numpy 数组并归一化到 [0, 1]
        temp_image = np.asarray(temp_image, dtype=np.float32) / 255.0
        # 调整维度以匹配模型输入 (1, 224, 224, 3)
        temp_image = temp_image.reshape((1, image_size, image_size, 3))
        image_set.append(temp_image)

    # 加载训练好的模型
    model = load_model(r'./checkpoint/final_model.keras')
    # 批量预测
    predictions = model.predict(np.vstack(image_set))
    # 输出预测结果
    for i, pred in enumerate(predictions):
        prob_normal = pred[0]
        prob_pneumonia = pred[1]
        most_likely = label_dict[0] if prob_normal >= prob_pneumonia else label_dict[1]
        # 输出可读结果
        print(f'[Result] Image: {input_list[i]}, '
              f'Normal: {prob_normal:.4f}, Pneumonia: {prob_pneumonia:.4f}, '
              f'Most Likely: {most_likely}')


def main():
    """程序主入口，执行数据加载、模型构建、训练和评估。

    Args:
        无参数。

    Returns:
        无返回值。
    """
    mode = 'predict'
    # 设置随机种子
    set_random_seed()
    # 定义数据目录路径
    data_dir = Path('./dataset')
    # 加载数据
    train_data, valid_data, valid_labels, test_data, test_labels = load_data(data_dir)
    # print("验证样本总数: ", valid_data.shape)
    # print("标签总数: ", valid_labels.shape)
    # print("测试样本总数: ", test_data.shape)
    # print("标签总数: ", test_labels.shape)
    # 可视化训练数据
    # visualize_data(train_data)

    if mode == 'train':
        # 构建并显示模型
        model = build_model()
        # model.summary()
        # 编译和训练模型
        compile_and_train(model, train_data, valid_data, valid_labels, batch_size=16, epochs=10)
    elif mode == 'validate':
        model = load_model(r'./checkpoint/final_model.keras')
        # 评估模型
        validate_model(model, test_data, test_labels)
    elif mode == 'predict':
        predict()


if __name__ == '__main__':
    main()
