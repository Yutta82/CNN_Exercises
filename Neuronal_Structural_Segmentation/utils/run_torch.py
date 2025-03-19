import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from skimage import io

from Neuronal_Structural_Segmentation.utils.augmentation import Augmentation
from Neuronal_Structural_Segmentation.utils.model_torch import u_net


def build_args():
    """
    构造命令行参数解析器并返回参数字典。
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('-a', '--train', required=False, default='dataset/train/', help='path to train data set')
    ap.add_argument('-t', '--test', required=False, default='dataset/test/', help='path to test data set')
    ap.add_argument('-v', '--val', required=False, default='dataset/val/', help='path to val data set')
    ap.add_argument('-s', '--steps', required=False, type=int, default=30, help='steps per epoch for train')
    ap.add_argument('-e', '--epochs', required=False, type=int, default=60, help='epochs for train model')
    ap.add_argument('-m', '--mode', required=False, choices=['train', 'predict'], default='train', help='model type')
    ap.add_argument('-n', '--number', required=False, type=int, default=5, help='number of images')
    args = vars(ap.parse_args())
    return args


def train(args):
    """
    模型训练函数。
    """
    data_gen_args = dict(
        rotation_range=0.2,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.05,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    aug = Augmentation()
    train_gen = aug.train_generator(
        batch_size=1,
        train_path=args['train'],
        images_folder='images',
        masks_folder='labels',
        aug_dict=data_gen_args,
        save_to_dir=None
    )
    val_gen = aug.train_generator(
        batch_size=1,
        train_path=args['val'],
        images_folder='images',
        masks_folder='labels',
        aug_dict={},
        save_to_dir=None
    )

    model = u_net()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()

    num_epochs = args['epochs']
    steps_per_epoch = args['steps']
    val_steps = 6

    best_loss = float('inf')
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        for step in range(steps_per_epoch):
            batch = next(train_gen)
            images, masks = batch
            images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
            masks = torch.tensor(masks, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct += (preds == masks).sum().item()
            total += masks.numel()
        train_loss = epoch_loss / steps_per_epoch
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        model.eval()
        val_epoch_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for step in range(val_steps):
                batch = next(val_gen)
                images, masks = batch
                images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
                masks = torch.tensor(masks, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_epoch_loss += loss.item()
                preds = (outputs > 0.5).float()
                correct_val += (preds == masks).sum().item()
                total_val += masks.numel()
        val_loss = val_epoch_loss / val_steps
        val_acc = correct_val / total_val
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"Train Loss: {train_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_acc:.4f}")

        if train_loss < best_loss:
            best_loss = train_loss
            if not os.path.exists('checkpoint'):
                os.makedirs('checkpoint')
            torch.save(model.state_dict(), r'checkpoint/final_model.pth')
            print("Save the better model to checkpoint/final_model.pth")

    epochs_range = range(1, num_epochs + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, train_losses, 'r-', label='Train Loss')
    plt.plot(epochs_range, val_losses, 'r--', label='Validation Loss')
    plt.plot(epochs_range, train_accuracies, 'b-', label='Train Accuracy')
    plt.plot(epochs_range, val_accuracies, 'b--', label='Validation Accuracy')
    plt.title('Neuronal_Structural_Segmentation Loss and Accuracy per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join("checkpoint", "img.png")
    plt.savefig(plot_path)
    print(f"Training curve saved to {plot_path}")
    plt.show()


def predict(args):
    """
    模型预测函数。
    """
    aug = Augmentation()
    model = u_net()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(r'checkpoint/final_model.pth', map_location=device))
    model.to(device)
    model.eval()

    test_path = args['test']
    num_images_to_show = args['number']
    num_images_to_show = 5 if num_images_to_show > 30 else num_images_to_show
    num = sorted(random.sample(range(0, 30), num_images_to_show))
    test_generator_ = aug.test_generator(test_path, num=num)
    results = []
    with torch.no_grad():
        for batch in test_generator_:
            image = batch[0]  # shape: (1, H, W, C)
            image_tensor = torch.tensor(image, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
            output = model(image_tensor)
            output_np = output.cpu().numpy().transpose(0, 2, 3, 1)
            results.append(output_np[0])
    results = np.array(results)
    aug.save_result(r"dataset/test/results", results, num=num)

    for i in num:
        original_image = io.imread(os.path.join(test_path, f"{i}.png"))
        predicted_mask = io.imread(os.path.join(r"dataset/test/results", f"{i}_predict.png"))
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(original_image, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(predicted_mask, cmap='gray')
        plt.title('Predicted Mask')
        plt.axis('off')
        plt.show()
