import os
import csv
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为SimHei
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

import cv2
import torch
import itertools
import numpy as np
import pandas as pd
from matplotlib import rc
from pathlib import Path
from PIL import Image
from itertools import cycle

from torchvision import transforms
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# 定义CLAHE转换函数
def apply_clahe(img):
    # 将PIL图像转换为OpenCV图像a
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # 转换到LAB颜色空间
    lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)

    # 分离L通道
    l, a, b = cv2.split(lab)

    # 创建CLAHE对象
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    # 应用CLAHE于L通道
    cl = clahe.apply(l)

    # 合并回LAB图像
    limg = cv2.merge((cl, a, b))

    # 转换回RGB颜色空间
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # 将OpenCV图像转换回PIL图像
    return Image.fromarray(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))


# Function to calculate class weights for the loss function
def calculate_weights(root_dir):
    total_count = 0
    class_counts = []

    # 获取每个类别的样本数
    for class_dir in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_dir)
        if os.path.isdir(class_path):
            count = len(os.listdir(class_path))
            class_counts.append(count)
            total_count += count

    # 计算每个类别的权重
    weights = [total_count / count for count in class_counts]
    # weights=weights[::-1]

    return torch.tensor(weights, dtype=torch.float32)


# Data transformation for training

import torch

import torch
from torchvision import transforms
import random

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., probability=0.5):
        self.mean = mean
        self.std = std
        self.probability = probability

    def __call__(self, tensor):
        if random.random() < self.probability:
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        return tensor

    def __repr__(self):
        return f'{self.__class__.__name__}(mean={self.mean}, std={self.std}, probability={self.probability})'

class AddPoissonNoise(object):
    def __init__(self, scale=1.0):
        self.scale = scale

    def __call__(self, tensor):
        if not torch.is_tensor(tensor):
            raise TypeError(f'Expected torch.Tensor but got {type(tensor)}')
        # Apply Poisson noise only to tensor
        noise = torch.poisson(tensor * self.scale) / self.scale
        return tensor + noise - tensor.mean()

    def __repr__(self):
        return f'{self.__class__.__name__}(scale={self.scale})'

size=[230,230]
center_size=[224,224]
TrainTransform = transforms.Compose([
    transforms.Resize(size),
   transforms.CenterCrop(center_size),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.02),

    transforms.ToTensor(),
    # AddPoissonNoise(scale=5.0),
    #AddGaussianNoise(0., 0.001, 0.5),  # 添加高斯噪声，mean和std根据需要调整，概率设为0.5

    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 在你的数据变换流程中加入高斯噪声层


# Data transformation for validation
ValidTransform = transforms.Compose([
    transforms.Resize(size),
    transforms.CenterCrop(center_size),

    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to save ROC data to a CSV file
def save_roc_data_to_csv(fpr_dict, tpr_dict, roc_auc_dict, class_names, file_path):
    # CSV file operations
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Class', 'FPR', 'TPR', 'AUC'])

        # Save micro-average ROC curve data
        for fpr, tpr in zip(fpr_dict["micro"], tpr_dict["micro"]):
            writer.writerow(['Micro-average', fpr, tpr, roc_auc_dict["micro"]])

        # Save ROC curve data for other classes
        for i, class_name in enumerate(class_names):
            for fpr, tpr in zip(fpr_dict[i], tpr_dict[i]):
                writer.writerow([class_name, fpr, tpr, roc_auc_dict[i]])

# Function to plot ROC curves
def plot_roc_curve2(fpr_dict, tpr_dict, roc_auc_dict, class_names, save_path, title):
    # Plotting settings
    plt.figure(figsize=(11, 11))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为SimHei
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    lw = 3
    title=''
    class_names=class_names[::-1]
    save_roc_data_to_csv(fpr_dict, tpr_dict, roc_auc_dict, class_names, save_path + 'roc_data.csv')

    # Plot micro-average ROC curve
    plt.plot(fpr_dict["micro"] * 100, tpr_dict["micro"] * 100,
             label=f'Micro-average ROC Curve (AUC = {roc_auc_dict["micro"]:0.4f})',
             color='deeppink', linestyle=':', linewidth=4)

    # Plot ROC curves for other classes
    colors = cycle([ 'aqua', 'darkorange','cornflowerblue','green', 'yellow', 'lime', 'blue'])
    for i, color in zip(range(len(class_names)), colors):
        plt.plot(fpr_dict[i] * 100, tpr_dict[i] * 100, color=color, lw=lw,
                 label=f'{class_names[i]} ROC Curve (AUC = {roc_auc_dict[i]:0.4f})')

    # Additional plot settings
    # plt.plot([0, 1] * 100, [0, 1] * 100, 'k--', lw=lw)
    plt.xlim([0.0, 100.0])
    plt.ylim([0.0, 100.0])
    plt.xlabel('False Positive Rate (%)', fontsize=18)
    plt.ylabel('True Positive Rate (%)', fontsize=18)

    plt.plot([0, 100], [0, 100], color='black', linestyle='--',lw=lw)

    plt.title(title)
    plt.legend(loc="lower right", fontsize=18)
    plt.grid(False)
    plt.gca().spines['top'].set_visible(True)
    plt.gca().spines['right'].set_visible(True)
    plt.savefig(save_path, format='pdf')
    plt.show()
def plot_roc_curve(fpr_dict, tpr_dict, roc_auc_dict, class_names, save_path, title):
    # Plotting settings
    plt.figure(figsize=(11, 11))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为SimHei
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    lw = 3
    title=''
    class_names=class_names[::-1]
    save_roc_data_to_csv(fpr_dict, tpr_dict, roc_auc_dict, class_names, save_path + 'roc_data.csv')

    # Plot micro-average ROC curve
    plt.plot(fpr_dict["micro"] , tpr_dict["micro"] ,
             label=f'Micro-average ROC曲线 (AUC = {roc_auc_dict["micro"]:0.4f})',
             color='deeppink', linestyle=':', linewidth=4)

    # Plot ROC curves for other classes
    colors = cycle([ 'aqua', 'darkorange','cornflowerblue','green', 'yellow', 'lime', 'blue'])
    for i, color in zip(range(len(class_names)), colors):
        plt.plot(fpr_dict[i] , tpr_dict[i] , color=color, lw=lw,
                 label=f'{class_names[i]} ROC曲线 (AUC = {roc_auc_dict[i]:0.4f})')

    # Additional plot settings
    # plt.plot([0, 1] * 100, [0, 1] * 100, 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate (%)', fontsize=18)
    plt.ylabel('True Positive Rate (%)', fontsize=18)

    plt.plot([0, 1], [0, 1], color='black', linestyle='--',lw=lw)

    plt.title(title)
    plt.legend(loc="lower right", fontsize=18)
    plt.grid(False)
    plt.gca().spines['top'].set_visible(True)
    plt.gca().spines['right'].set_visible(True)
    plt.savefig(save_path, format='pdf')
    plt.show()

# Function to plot confusion matrix
def plot_confusion_matrix2(cm, target_names, save_dir, title='Confusion Matrix', cmap=plt.cm.Greens, normalize=True):
    # Plotting confusion matrix
    plt.figure(figsize=(15, 12))
    plt.title(title, fontsize=36)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为SimHei
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # Setting axis labels
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45, fontsize=32)
        plt.yticks(tick_marks, target_names, fontsize=32)

    # Normalization option
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1 if normalize else None)

    # Color bar settings
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=20)

    # Text settings inside the matrix
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:.2f}%".format(cm[i, j] * 100) if normalize else "{:,}".format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=32)

    # Additional plot settings
    plt.tight_layout()
    save_path = Path(os.path.join(save_dir, 'confusion_matrix.pdf'))
    print(f"正在保存到: {save_path}")
    plt.savefig(save_path)

    plt.show()


def plot_confusion_matrix(cm, target_names, save_dir, title='Confusion Matrix', cmap=plt.cm.Greens, normalize=True):
    # Plotting confusion matrix
    plt.figure(figsize=(15, 12))
    plt.title(title, fontsize=36)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为SimHei
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    cm=np.flip(cm)
    target_names=target_names[::-1]

    # Setting axis labels
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45, fontsize=32)
        plt.yticks(tick_marks, target_names, fontsize=32)

    # Normalization option
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1 if normalize else None)

    # Color bar settings
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=20)

    # Text settings inside the matrix
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:.0f}".format(cm[i, j] * 100) if normalize else "{:,}".format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=32)

    # Additional plot settings
    plt.tight_layout()
    save_path = Path(os.path.join(save_dir, 'confusion_matrix.pdf'))
    print(f"正在保存到: {save_path}")
    plt.savefig(save_path)

    plt.show()


# Rest of your functions...
def save_micro_curves_to_excel(pr_precision_micro, pr_recall_micro, roc_fpr_micro, roc_tpr_micro, roc_auc_micro, excel_path):
    """
    Save micro-average PR and ROC curve data to an Excel file.
    :param pr_precision_micro: Precision data for micro-average PR curve.
    :param pr_recall_micro: Recall data for micro-average PR curve.
    :param roc_fpr_micro: FPR data for micro-average ROC curve.
    :param roc_tpr_micro: TPR data for micro-average ROC curve.
    :param roc_auc_micro: AUC data for micro-average ROC curve.
    :param excel_path: Path to save the Excel file.
    """
    # Create DataFrame for PR curve
    pr_data = {'Precision': pr_precision_micro, 'Recall': pr_recall_micro}
    pr_df = pd.DataFrame(pr_data)

    # Create DataFrame for ROC curve
    roc_data = {'FPR': roc_fpr_micro, 'TPR': roc_tpr_micro, 'AUC': [roc_auc_micro] * len(roc_fpr_micro)}
    roc_df = pd.DataFrame(roc_data)

    # Save data to Excel file
    with pd.ExcelWriter(excel_path) as writer:
        pr_df.to_excel(writer, sheet_name='Micro-Average PR Curve', index=False)
        roc_df.to_excel(writer, sheet_name='Micro-Average ROC Curve', index=False)

# Function to calculate ROC metrics
def calculate_roc_metrics(label_onehot, score_array, num_classes):
    """
    Calculate ROC metrics for each class and micro/macro averages.
    :param label_onehot: One-hot encoded labels.
    :param score_array: Prediction scores for each class.
    :param num_classes: Number of classes.
    :return: Dictionaries of FPR, TPR, and AUC for each class and micro/macro averages.
    """
    fpr_dict = dict()
    tpr_dict = dict()
    roc_auc_dict = dict()

    # Calculate ROC metrics for each class
    for i in range(num_classes):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(label_onehot[:, i], score_array[:, i])
        roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])

    # Calculate micro-average ROC metrics
    fpr_dict["micro"], tpr_dict["micro"], _ = roc_curve(label_onehot.ravel(), score_array.ravel())
    roc_auc_dict["micro"] = auc(fpr_dict["micro"], tpr_dict["micro"])

    # Calculate macro-average ROC metrics
    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
    mean_tpr /= num_classes
    fpr_dict["macro"] = all_fpr
    tpr_dict["macro"] = mean_tpr
    roc_auc_dict["macro"] = auc(fpr_dict["macro"], tpr_dict["macro"])

    return fpr_dict, tpr_dict, roc_auc_dict

# Function to plot PR curve
def plot_pr_curve(precision, recall, average_precision, class_names, save_path):
    """
    Plot PR curve for each class and micro-average.
    :param precision: Precision values for each class.
    :param recall: Recall values for each class.
    :param average_precision: Average precision values for each class.
    :param class_names: List of class names.
    :param save_path: Path to save the plot.
    """
    plt.figure()
    num_classes = len(class_names)
    # Plot PR curve for each class
    for i in range(num_classes):
        plt.step(recall[i] * 100, precision[i] * 100, where='post',
                 label=f'{class_names[i]} AP={average_precision[i]:0.2f}')

    # Plot micro-average PR curve
    plt.step(recall['micro'] * 100, precision['micro'] * 100, where='post', color='red', linestyle=':',
             label=f'Micro-average AP={average_precision["micro"]:0.2f}')

    # Additional plot settings
    plt.xlabel('Recall (%)')
    plt.ylabel('Precision (%)')
    plt.ylim([0.0, 100.0])
    plt.xlim([0.0, 100.0])
    plt.title('PR Curve')
    plt.legend(loc="lower right")
    plt.savefig(save_path, format='pdf')
    plt.show()

# Function to calculate precision and recall metrics
def calculate_precision_recall_metrics(label_list, score_array, num_classes):
    """
    Calculate precision and recall metrics for each class and micro-average.
    :param label_list: List of labels.
    :param score_array: Prediction scores for each class.
    :param num_classes: Number of classes.
    :return: Dictionaries of precision, recall, and average precision for each class and micro-average.
    """
    precision = dict()
    recall = dict()
    average_precision = dict()

    # Convert labels to one-hot encoding
    label_tensor = torch.tensor(label_list)
    label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
    label_onehot = torch.zeros(label_tensor.shape[0], num_classes)
    label_onehot.scatter_(dim=1, index=label_tensor, value=1)
    label_onehot = np.array(label_onehot)

    # Calculate metrics for each class
    for i in range(num_classes):
        precision[i], recall[i], _ = precision_recall_curve(label_onehot[:, i], score_array[:, i])
        average_precision[i] = average_precision_score(label_onehot[:, i], score_array[:, i])

    # Calculate micro-average metrics
    precision["micro"], recall["micro"], _ = precision_recall_curve(label_onehot.ravel(), score_array.ravel())
    average_precision["micro"] = average_precision_score(label_onehot, score_array, average="micro")

    return precision, recall, average_precision