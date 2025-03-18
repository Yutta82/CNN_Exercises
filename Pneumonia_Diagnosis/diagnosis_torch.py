import argparse
import os
import pickle
import random
from pathlib import Path

import imgaug as aug
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# PyTorch 相关导入
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from mlxtend.plotting import plot_confusion_matrix
from skimage.io import imread
from sklearn.metrics import confusion_matrix

np.bool = bool  # 为兼容性定义 np.bool

# 设置 matplotlib 的颜色
color = sns.color_palette()


def to_categorical(y, num_classes):
    """
    将类别索引转换为独热编码向量。

    Args:
        y (int): 类别索引。
        num_classes (int): 类别总数。

    Returns:
        np.ndarray: 独热编码向量，形状 (num_classes,)。
    """
    return np.eye(num_classes, dtype=np.float32)[y]


def load_config():
    """
    加载命令行参数配置。

    Returns:
        argparse.Namespace: 包含所有命令行参数的对象。
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="中文字符识别参数配置")
    # 添加运行模式参数
    parser.add_argument('--mode', type=str, choices=['train', 'validation', 'inference'],
                        default='train',
                        help='运行模式，覆盖配置文件中的 mode')
    # 解析命令行参数
    args = parser.parse_args()
    return args


def set_random_seed():
    """
    设置随机种子，确保结果的可重复性。

    设置 Python、NumPy、TensorFlow、PyTorch 以及图像增强的随机种子。
    """
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(111)
    random.seed(111)
    torch.manual_seed(111)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(111)
    aug.seed(111)


def load_data(data_dir):
    """
    加载训练、验证和测试数据。

    Args:
        data_dir (Path): 数据目录路径。

    Returns:
        tuple: 包含训练数据 DataFrame、验证图像数组、验证标签数组、
               测试图像数组和测试标签数组。
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
    """
    处理图像：调整大小、归一化，并对标签进行独热编码。

    Args:
        image_paths (generator): 图像文件路径生成器。
        label (int): 图像标签（0: 正常, 1: 肺炎）。

    Returns:
        tuple: (data, labels)，分别为处理后的图像数组和独热编码标签数组。
    """
    data = []
    labels = []
    for img_path in image_paths:
        img = Image.open(str(img_path)).convert('RGB')
        img = img.resize((224, 224))
        img = np.asarray(img, dtype=np.float32) / 255.0
        encoded_label = to_categorical(label, num_classes=2)
        data.append(img)
        labels.append(encoded_label)
    return np.array(data), np.array(labels)


def visualize_data(train_data):
    """
    可视化训练数据的分布和样本。

    Args:
        train_data (pd.DataFrame): 训练数据 DataFrame。
    """
    # 统计每个类别的样本数量
    cases_count = train_data['label'].value_counts()
    print(cases_count)
    plt.figure(figsize=(10, 8))
    sns.barplot(x=cases_count.index, y=cases_count.values)
    plt.title('Number of cases', fontsize=14)
    plt.xlabel('Case type', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(range(len(cases_count.index)), ['Normal(0)', 'Pneumonia(1)'])
    plt.show()

    # 绘制部分样本图像
    pneumonia_samples = (train_data[train_data['label'] == 1]['image'].iloc[:5]).tolist()
    normal_samples = (train_data[train_data['label'] == 0]['image'].iloc[:5]).tolist()
    samples = pneumonia_samples + normal_samples

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
    """
    定义并返回图像增强序列。

    Returns:
        iaa.OneOf: 图像增强序列。
    """
    return iaa.OneOf([
        iaa.Fliplr(),  # 水平翻转
        iaa.Affine(rotate=20),  # 旋转 20 度
        iaa.Multiply((1.2, 1.5))  # 随机调整亮度
    ])


def data_generator(data, batch_size):
    """
    数据生成器，用于训练时批量生成和增强数据。

    Args:
        data (pd.DataFrame): 训练数据 DataFrame。
        batch_size (int): 批量大小。

    Yields:
        tuple: (batch_data, batch_labels)，分别为批量图像数据和批量标签数据（均为 numpy 数组）。
    """
    n = len(data)
    steps = n // batch_size
    batch_data = np.zeros((batch_size, 224, 224, 3), dtype=np.float32)
    batch_labels = np.zeros((batch_size, 2), dtype=np.float32)
    indices = np.arange(n)
    i = 0
    seq = get_augmentation_sequence()
    while True:
        np.random.shuffle(indices)
        next_batch = indices[(i * batch_size):(i + 1) * batch_size]
        count = 0
        for j, idx in enumerate(next_batch):
            img_name = data.iloc[idx]['image']
            label = data.iloc[idx]['label']
            encoded_label = to_categorical(label, num_classes=2)
            img = Image.open(img_name).convert('RGB')
            img = img.resize((224, 224))
            orig_img = np.asarray(img, dtype=np.float32) / 255.0
            batch_data[count] = orig_img
            batch_labels[count] = encoded_label
            if label == 0 and count < batch_size - 2:
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


# ---------------------------
# PyTorch 实现：SeparableConv2d，用于模拟 Keras 中的 SeparableConv2D
class SeparableConv2d(nn.Module):
    """
    实现深度可分离卷积。

    Args:
        in_channels (int): 输入通道数。
        out_channels (int): 输出通道数。
        kernel_size (int): 卷积核大小。
        padding (int, optional): 填充大小，默认 1。
        bias (bool, optional): 是否使用偏置，默认 True.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding=1, bias=True):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        """
        前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 输出张量。
        """
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


# ---------------------------


def build_model():
    """
    构建 CNN 模型，使用 PyTorch 实现，与原 Keras 模型结构保持一致。

    Returns:
        torch.nn.Module: 构建好的模型。
    """

    class CNNModel(nn.Module):
        """
        内部 CNN 模型类，与原 Keras 模型结构对应。
        """

        def __init__(self):
            super(CNNModel, self).__init__()
            self.Conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
            self.Conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            self.pool1 = nn.MaxPool2d(2, stride=2)

            self.Conv2_1 = SeparableConv2d(64, 128, kernel_size=3, padding=1)
            self.Conv2_2 = SeparableConv2d(128, 128, kernel_size=3, padding=1)
            self.pool2 = nn.MaxPool2d(2, stride=2)

            self.Conv3_1 = SeparableConv2d(128, 256, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(256)
            self.Conv3_2 = SeparableConv2d(256, 256, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(256)
            self.Conv3_3 = SeparableConv2d(256, 256, kernel_size=3, padding=1)
            self.pool3 = nn.MaxPool2d(2, stride=2)

            self.Conv4_1 = SeparableConv2d(256, 512, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(512)
            self.Conv4_2 = SeparableConv2d(512, 512, kernel_size=3, padding=1)
            self.bn4 = nn.BatchNorm2d(512)
            self.Conv4_3 = SeparableConv2d(512, 512, kernel_size=3, padding=1)
            self.pool4 = nn.MaxPool2d(2, stride=2)

            self.flatten = nn.Flatten()
            # 输入尺寸224经过4次池化变为14x14，通道数为512
            self.fc1 = nn.Linear(512 * 14 * 14, 1024)
            self.dropout1 = nn.Dropout(0.7)
            self.fc2 = nn.Linear(1024, 512)
            self.dropout2 = nn.Dropout(0.5)
            self.fc3 = nn.Linear(512, 2)

        def forward(self, x):
            """
            前向传播。

            Args:
                x (torch.Tensor): 输入张量，形状 (batch, 3, 224, 224)。

            Returns:
                torch.Tensor: 输出 logits，形状 (batch, 2)。
            """
            x = torch.relu(self.Conv1_1(x))
            x = torch.relu(self.Conv1_2(x))
            x = self.pool1(x)

            x = torch.relu(self.Conv2_1(x))
            x = torch.relu(self.Conv2_2(x))
            x = self.pool2(x)

            x = torch.relu(self.Conv3_1(x))
            x = self.bn1(x)
            x = torch.relu(self.Conv3_2(x))
            x = self.bn2(x)
            x = torch.relu(self.Conv3_3(x))
            x = self.pool3(x)

            x = torch.relu(self.Conv4_1(x))
            x = self.bn3(x)
            x = torch.relu(self.Conv4_2(x))
            x = self.bn4(x)
            x = torch.relu(self.Conv4_3(x))
            x = self.pool4(x)

            x = self.flatten(x)
            x = torch.relu(self.fc1(x))
            x = self.dropout1(x)
            x = torch.relu(self.fc2(x))
            x = self.dropout2(x)
            x = self.fc3(x)
            # 注意：不在此处使用 softmax，因为 CrossEntropyLoss 要求原始 logits
            return x

    return CNNModel()


def compile_and_train(model, train_data, valid_data, valid_labels, batch_size, epochs):
    """
    使用 PyTorch 编译和训练模型，并绘制训练过程中 loss 与 accuracy 的变化曲线。

    Args:
        model (Model): 原 Keras 模型，此处为 PyTorch 模型实例。
        train_data (pd.DataFrame): 训练数据 DataFrame。
        valid_data (np.ndarray): 验证图像数组，形状 (N, 224, 224, 3)。
        valid_labels (np.ndarray): 验证标签，one-hot 编码，形状 (N, 2)。
        batch_size (int): 批量大小。
        epochs (int): 训练轮数。

    Returns:
        None
    """
    print("train")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # to()将模型加载到指定设备上
    model.to(device)
    # 将验证标签从 one-hot 转换为类别索引
    valid_labels_idx = np.argmax(valid_labels, axis=1)
    valid_data_tensor = torch.tensor(valid_data.transpose(0, 3, 1, 2), dtype=torch.float32).to(device)
    valid_labels_tensor = torch.tensor(valid_labels_idx, dtype=torch.long).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

    train_gen = data_generator(train_data, batch_size)
    steps_per_epoch = len(train_data) // batch_size

    loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for step in range(steps_per_epoch):
            batch_data, batch_labels = next(train_gen)
            # 转换 batch_data 为 tensor，调整维度为 (batch, 3, 224, 224)
            batch_data_tensor = torch.tensor(batch_data.transpose(0, 3, 1, 2), dtype=torch.float32).to(device)
            batch_labels_idx = np.argmax(batch_labels, axis=1)
            batch_labels_tensor = torch.tensor(batch_labels_idx, dtype=torch.long).to(device)

            optimizer.zero_grad()
            outputs = model(batch_data_tensor)
            loss = criterion(outputs, batch_labels_tensor)
            loss.backward()
            optimizer.step()
            # 将一个Tensor变量转换为python标量，常用于用于深度学习训练时，将loss值转换为标量并加
            running_loss += loss.item() * batch_data_tensor.size(0)
            _, predicted = torch.max(outputs, 1)
            total += batch_labels_tensor.size(0)
            correct += (predicted == batch_labels_tensor).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc)

        # 验证阶段
        model.eval()
        # with torch.no_grad()或者@torch.no_grad()中的数据不需要计算梯度，也不会进行反向传播
        with torch.no_grad():
            val_outputs = model(valid_data_tensor)
            val_loss = criterion(val_outputs, valid_labels_tensor)
            _, val_predicted = torch.max(val_outputs, 1)
            val_correct = (val_predicted == valid_labels_tensor).sum().item()
            val_total = valid_labels_tensor.size(0)
            epoch_val_loss = val_loss.item()
            epoch_val_acc = val_correct / val_total
        val_loss_history.append(epoch_val_loss)
        val_acc_history.append(epoch_val_acc)

        print(f"Epoch {epoch + 1}/{epochs}: Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

        # 保存表现最佳的模型（按验证准确率）
        if epoch_val_acc >= max(val_acc_history):
            torch.save(model.state_dict(), os.path.join(r'./checkpoint', 'final_model.pth'))
            print("Save the better model to " + os.path.join(r'./checkpoint', 'final_model.pth'))

    epochs_range = range(1, len(loss_history) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, loss_history, 'r-', label='Train Loss')
    plt.plot(epochs_range, train_acc_history, 'b-', label='Train Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.title('Training & Validation Loss and Accuracy')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(r"./checkpoint", 'img_torch.png')
    plt.savefig(plot_path)
    print(f"Training curve saved to {plot_path}")
    plt.show()


def validate_model(model, test_data, test_labels):
    """
    评估模型并可视化测试结果。

    Args:
        model (Model): PyTorch 模型实例。
        test_data (np.ndarray): 测试图像数组，形状 (N, 224, 224, 3)。
        test_labels (np.ndarray): 测试标签，one-hot 编码，形状 (N, 2)。

    Returns:
        None
    """
    print("validate_model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(r"./checkpoint", 'final_model.pth'), map_location=device))
    model.eval()

    test_data_tensor = torch.tensor(test_data.transpose(0, 3, 1, 2), dtype=torch.float32).to(device)
    test_labels_idx = np.argmax(test_labels, axis=1)
    test_labels_tensor = torch.tensor(test_labels_idx, dtype=torch.long).to(device)

    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        outputs = model(test_data_tensor)
        loss = criterion(outputs, test_labels_tensor)
        _, predicted = torch.max(outputs, 1)
        total = test_labels_tensor.size(0)
        correct = (predicted == test_labels_tensor).sum().item()
        test_acc = correct / total

    print("测试集损失: {:.4f}, 测试集准确率: {:.4f}".format(loss.item(), test_acc))
    # cpu() 将数据的处理设备从其他设备（如.cuda()）拿到cpu上
    # numpy() 将Tensor转化为ndarray
    predicts = predicted.cpu().numpy()
    orig_test_labels = test_labels_idx
    cm = confusion_matrix(orig_test_labels, predicts)
    plt.figure()
    plot_confusion_matrix(
        cm,
        figsize=(12, 8),
        hide_ticks=True,
        fontcolor_threshold=0.7,
        cmap=plt.get_cmap("Blues")
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
    使用训练好的模型对输入图像进行推理预测，并显示结果。

    Returns:
        None
    """
    print("predict")
    label_dict = {0: 'Normal', 1: 'Pneumonia'}
    image_set = []
    image_size = 224  # 模型输入尺寸
    input_dir = r"./input"
    input_list = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.jpeg')]

    for image_path in input_list:
        temp_image = Image.open(image_path).convert('RGB')
        temp_image = temp_image.resize((image_size, image_size), Image.Resampling.LANCZOS)
        temp_image = np.asarray(temp_image, dtype=np.float32) / 255.0
        temp_image = temp_image.reshape((1, image_size, image_size, 3))
        image_set.append(temp_image)

    images_np = np.vstack(image_set)
    images_tensor = torch.tensor(images_np.transpose(0, 3, 1, 2), dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    model.load_state_dict(torch.load(os.path.join(r"./checkpoint", 'final_model.pth'), map_location=device))
    model.eval()

    with torch.no_grad():
        outputs = model(images_tensor.to(device))
        outputs_prob = torch.softmax(outputs, dim=1).cpu().numpy()
        for i, pred in enumerate(outputs_prob):
            top3_indices = np.argsort(pred)[-3:][::-1]
            top3_values = pred[top3_indices]
            print(f'Image: {input_list[i]}, Top 3 predictions: {top3_indices}, Values: {top3_values}')
            candidate1 = top3_indices[0]
            candidate2 = top3_indices[1]
            candidate3 = top3_indices[2]
            print(f'[Result] Image: {input_list[i]}, '
                  f'Predict: {label_dict[candidate1]} {label_dict[candidate2]} {label_dict[candidate3]}, '
                  f'Most Likely: {label_dict[candidate1]}')


def get_label_dict():
    """
    获取标签字典，将标签映射为汉字。

    Returns:
        dict: 标签映射字典。
    """
    with open('./char_dict', 'rb') as f:
        label_dict = pickle.load(f)
    return label_dict


def get_file_list(path):
    """
    获取指定目录下的文件列表（排序后）。

    Args:
        path (str): 目录路径。

    Returns:
        list: 文件路径列表。
    """
    list_name = []
    files = os.listdir(path)
    files.sort()
    for file in files:
        file_path = os.path.join(path, file)
        list_name.append(file_path)
    return list_name


def main():
    """
    程序主入口，根据运行模式执行数据加载、模型构建、训练、验证或推理任务。

    Returns:
        None
    """
    args = load_config()
    mode = args.mode
    set_random_seed()
    data_dir = Path('./dataset')
    train_data, valid_data, valid_labels, test_data, test_labels = load_data(data_dir)

    if mode == 'train':
        model = build_model()
        compile_and_train(model, train_data, valid_data, valid_labels, batch_size=16, epochs=5)
    elif mode == 'validate':
        model = build_model()
        model.load_state_dict(
            torch.load(os.path.join('./checkpoint', 'final_model.pth'), map_location=torch.device("cpu")))
        validate_model(model, test_data, test_labels)
    elif mode == 'inference':
        predict()


if __name__ == '__main__':
    main()
