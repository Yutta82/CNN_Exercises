import argparse
import json
import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset


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


class DataIterator(Dataset):
    """数据迭代器类，用于加载和处理图像数据"""

    def __init__(self, data_dir):
        """初始化数据迭代器

        Args:
            data_dir (str): 数据目录路径
        """
        # 生成字符集截断路径（例如：data_dir + "03755"）
        truncate_path = data_dir + ('%05d' % ARGS.charset_size)
        # print(truncate_path)

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

        # 默认不进行数据增强
        self.aug_flag = False

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]
        # 读取图像并转换为灰度
        image = Image.open(self.image_names[index]).convert('L')
        # 调整图像尺寸
        image = image.resize((ARGS.image_size, ARGS.image_size), Image.BILINEAR)
        # 转换为 numpy 数组并归一化到 [0,1]
        image = np.array(image, dtype=np.float32) / 255.0
        # 增加通道维度，转换为 (1, H, W)
        image = np.expand_dims(image, axis=0)
        image = torch.tensor(image, dtype=torch.float32)
        # 如果启用数据增强，则调用 data_augmentation
        if self.aug_flag:
            image = DataIterator.data_augmentation(image)
        return image, label

    @staticmethod
    def data_augmentation(image):
        """数据增强处理

        Args:
            image (torch.Tensor): 输入图像张量 (1, H, W)

        Returns:
            torch.Tensor: 增强后的图像张量
        """
        # 随机上下翻转
        if ARGS.random_flip_up_down:
            if random.random() < 0.5:
                image = torch.flip(image, dims=[1])
        # 随机亮度调整
        if ARGS.random_brightness:
            factor = 1.0 + random.uniform(-0.3, 0.3)
            image = image * factor
            image = torch.clamp(image, 0.0, 1.0)
        # 随机对比度调整
        if ARGS.random_contrast:
            mean = torch.mean(image)
            factor = random.uniform(0.8, 1.2)
            image = (image - mean) * factor + mean
            image = torch.clamp(image, 0.0, 1.0)
        return image

    def input_pipeline(self, batch_size, num_epochs=None, aug=False):
        """创建输入数据管道

        Args:
            batch_size (int): 批量大小
            num_epochs (int, optional): 迭代次数（PyTorch中由训练循环控制）
            aug (bool, optional): 是否启用数据增强

        Returns:
            DataLoader: 配置好的数据集加载器
        """
        self.aug_flag = aug
        return DataLoader(self, batch_size=batch_size, shuffle=True, num_workers=4)


def build_model():
    """构建卷积神经网络模型

    Returns:
        torch.nn.Module: 配置好的PyTorch模型
    """

    class CNNModel(nn.Module):
        def __init__(self):
            super(CNNModel, self).__init__()
            # 输入: (batch, 1, 64, 64)
            self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 此处不使用 'same' 填充
            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
            self.flatten = nn.Flatten()
            # 经过三次池化，64->32->16->8
            self.fc1 = nn.Linear(256 * 8 * 8, 1024)
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(1024, ARGS.charset_size)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = self.pool(x)
            x = torch.relu(self.conv2(x))
            x = self.pool(x)
            x = torch.relu(self.conv3(x))
            x = self.pool(x)
            # x = x.view(x.size(0), -1)  # 疑似展平层
            x = self.flatten(x)
            x = torch.tanh(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            # 原代码使用 softmax 激活，注意：CrossEntropyLoss 在 PyTorch 中要求未经过 softmax，
            # 因此这里我们在训练和验证时使用 log-softmax 与 NLLLoss。
            x = torch.softmax(x, dim=1)
            return x

    return CNNModel()


def train():
    """训练模型主函数"""
    print('Begin training')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化数据迭代器
    train_feeder = DataIterator(data_dir=ARGS.train_data_dir)
    test_feeder = DataIterator(data_dir=ARGS.test_data_dir)

    # 构建并编译模型
    model = build_model().to(device)
    # 定义损失函数（使用 NLLLoss 结合 log-softmax）和优化器
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-4)

    # 创建数据管道（DataLoader）
    train_loader = train_feeder.input_pipeline(batch_size=128, aug=True)
    test_loader = test_feeder.input_pipeline(batch_size=128)

    best_acc = 0.0
    patience = 5  # 早停忍耐度，若连续patience轮best_acc未改变，则直接停止训练
    wait = 0
    loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    epochs_range = range(1, ARGS.epoch + 1)

    for epoch in range(ARGS.epoch):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = torch.as_tensor(labels, dtype=torch.long, device=device)  # 避免用户警告
            optimizer.zero_grad()
            outputs = model(images)
            # 由于模型输出已经过 softmax，因此取 log 再计算 NLLLoss
            loss = criterion(torch.log(outputs), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = torch.as_tensor(labels, dtype=torch.long, device=device)  # 避免用户警告
                outputs = model(images)
                loss_val = criterion(torch.log(outputs), labels)
                val_loss += loss_val.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        epoch_val_loss = val_loss / total_val
        epoch_val_acc = correct_val / total_val
        val_loss_history.append(epoch_val_loss)
        val_acc_history.append(epoch_val_acc)

        print(f"Epoch {epoch + 1}/{ARGS.epoch}: Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), os.path.join(ARGS.checkpoint_dir, 'final_model.pth'))
            print("Save the better model to " + os.path.join(ARGS.checkpoint_dir, 'final_model.pth'))
            wait = 0  # 早停等待次数归零
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping")
                break

    # 绘制损失和准确率曲线（在同一张图中）
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, val_loss_history, 'r-', label='Val Loss')
    plt.plot(epochs_range, train_acc_history, 'b-', label='Train Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.title('Training & Validation Loss and Accuracy')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(ARGS.checkpoint_dir, 'img_torch.png')
    plt.savefig(plot_path)
    print(f"Training curve saved to {plot_path}")
    plt.show()


def validation():
    """模型验证函数"""
    print('Begin validation')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_feeder = DataIterator(data_dir=ARGS.test_data_dir)
    test_loader = test_feeder.input_pipeline(batch_size=128)
    model = build_model().to(device)
    model.load_state_dict(torch.load(os.path.join(ARGS.checkpoint_dir, 'final_model.pth'), map_location=device))
    model.eval()

    correct = 0
    total = 0
    total_loss = 0.0

    criterion = nn.NLLLoss()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = torch.tensor(labels, dtype=torch.long).to(device)
            outputs = model(images)
            loss = criterion(torch.log(outputs), labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = total_loss / total
    val_acc = correct / total
    print(f'Validation loss: {val_loss:.4f}, accuracy: {val_acc:.4f}')


def predict():
    """执行推理预测

    Args:

    Returns:
        list: 预测结果列表
    """
    print('Inference')
    label_dict = get_label_dict()
    input_list = get_file_list('./input')

    image_set = []
    for image_path in input_list:
        temp_image = Image.open(image_path).convert('L')
        temp_image = temp_image.resize((ARGS.image_size, ARGS.image_size), Image.LANCZOS)
        temp_image = np.array(temp_image, dtype=np.float32) / 255.0
        temp_image = np.expand_dims(temp_image, axis=0)  # (1, H, W)
        temp_image = torch.tensor(temp_image, dtype=torch.float32)
        temp_image = temp_image.unsqueeze(0)  # (1, 1, H, W)
        image_set.append(temp_image)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    model.load_state_dict(torch.load(os.path.join(ARGS.checkpoint_dir, 'final_model.pth'), map_location=device))
    model.eval()

    with torch.no_grad():
        for i, image in enumerate(image_set):
            image = image.to(device)
            output = model(image)
            prob = output.squeeze(0).cpu().numpy()
            top3_indices = np.argsort(prob)[-3:][::-1]
            top3_values = prob[top3_indices]
            print(f'Image: {input_list[i]}, Top 3 predictions: {top3_indices}, Values: {top3_values}')
            candidate1 = top3_indices[0]
            candidate2 = top3_indices[1]
            candidate3 = top3_indices[2]
            print(
                f'[Result] Image: {input_list[i]}, Predict: {label_dict[candidate1]} {label_dict[candidate2]} {label_dict[candidate3]}, Most Likely: {label_dict[candidate1]}')


def predict1():
    """
    在手写汉字数据集中查找特定汉字
    Return:
         None
    """
    print('Inference')
    label_dict = get_label_dict()

    # 指定数据集路径
    dataset_path = "./dataset/test"
    image_set = []
    # 遍历 dataset/test 下的所有文件夹
    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)

        # 确保是文件夹
        if os.path.isdir(folder_path):
            files = os.listdir(folder_path)  # 获取文件夹下的所有文件
            if files:  # 确保文件夹不为空
                first_file = files[0]  # 取第一个文件
                image_path = os.path.join(folder_path, first_file)
                temp_image = Image.open(image_path).convert('L')
                temp_image = temp_image.resize((ARGS.image_size, ARGS.image_size), Image.LANCZOS)
                temp_image = np.array(temp_image, dtype=np.float32) / 255.0
                temp_image = np.expand_dims(temp_image, axis=0)  # (1, H, W)
                temp_image = torch.tensor(temp_image, dtype=torch.float32)
                temp_image = temp_image.unsqueeze(0)  # (1, 1, H, W)
                image_set.append(temp_image)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    model.load_state_dict(torch.load(os.path.join(ARGS.checkpoint_dir, 'final_model.pth'), map_location=device))
    model.eval()

    with torch.no_grad():
        for i, image in enumerate(image_set):
            image = image.to(device)
            output = model(image)
            prob = output.squeeze(0).cpu().numpy()
            top1_indices = np.argsort(prob)[-1:][0]
            char = label_dict[top1_indices]
            print(f"{i}:{char}")
            if char == '一':
                break


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
    files = os.listdir(path)
    files.sort()
    for file in files:
        file_path = os.path.join(path, file)
        list_name.append(file_path)
    return list_name


def main():
    """程序入口函数"""
    if ARGS.mode == "train":
        train()
    elif ARGS.mode == 'validation':
        validation()
    elif ARGS.mode == 'predict':
        predict()


if __name__ == "__main__":
    main()
