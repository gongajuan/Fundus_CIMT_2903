import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve, average_precision_score

from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report  # 导入 classification_report
import matplotlib.pyplot as plt
import itertools
from ModelAndEyeDataset import NewEyeDataset, SiameseSeResNeXtdropout
from utils import ValidTransform
from sklearn.preprocessing import label_binarize
from scipy import interp
from itertools import cycle
import csv
from pathlib import Path

# 检测GPU并选择设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def validate_model(model, valid_loader, criterion):
    model.eval()  # 确保在验证时关闭dropout等
    running_loss = 0.0
    running_corrects = 0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for i, (combined_image, labels, id, age_gender) in enumerate(valid_loader):
            combined_image = combined_image.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            age_gender = age_gender.to(device, non_blocking=True)

            outputs = model(combined_image, age_gender)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * combined_image.size(0)
            _, preds = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1)
            running_corrects += torch.sum(preds == labels.data)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    total_loss = running_loss / len(valid_loader.dataset)
    total_acc = running_corrects.double() / len(valid_loader.dataset)

    all_probs = np.concatenate(all_probs, axis=0)

    return total_loss, total_acc, np.array(all_labels), np.array(all_preds), all_probs

def plot_confusion_matrix(labels, preds, save_dir):
    plt.figure(figsize=(15, 12))

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为SimHei
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    cm = confusion_matrix(labels, preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_percent = cm_normalized * 100

    # 绘制混淆矩阵并设置颜色范围
    im = plt.imshow(cm_percent, interpolation='nearest', cmap=plt.cm.Greens, vmin=0, vmax=100)
    plt.title('Valid Group Confusion Matrix', fontsize=56)

    # 添加颜色条
    cbar = plt.colorbar(im)
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label('%', fontsize=20)  # 可选，为颜色条添加标签
    cbar.set_ticks(np.linspace(0, 100, 6))  # 设置刻度从 0 到 100

    # 设置坐标轴刻度
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Normal', 'Thickened'], rotation=45, fontsize=48)
    plt.yticks(tick_marks, ['Normal', 'Thickened'], fontsize=48)

    # 添加百分比文本
    fmt = '.2f'
    thresh = cm_percent.max() / 2.
    for i, j in itertools.product(range(cm_percent.shape[0]), range(cm_percent.shape[1])):
        plt.text(j, i, format(cm_percent[i, j], fmt) + '%',
                 horizontalalignment="center",
                 color="white" if cm_percent[i, j] > thresh else "black", fontsize=56)

    plt.tight_layout()

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    save_path = Path(os.path.join(save_dir, 'confusion_matrix.pdf'))
    print(f"正在保存混淆矩阵图像到: {save_path}")
    plt.savefig(save_path)
    plt.close()

    # 保存混淆矩阵到txt文件
    cm_txt_path = Path(os.path.join(save_dir, 'confusion_matrix.txt'))
    cm_percent_txt_path = Path(os.path.join(save_dir, 'confusion_matrix_percent.txt'))
    print(f"正在保存混淆矩阵数据到: {cm_txt_path} 和 {cm_percent_txt_path}")
    np.savetxt(cm_txt_path, cm, fmt='%d')
    np.savetxt(cm_percent_txt_path, cm_percent, fmt='%.2f')


def save_roc_data_to_csv(fpr_dict, tpr_dict, roc_auc_dict, class_names, file_path):
    # CSV file operations
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Class', 'FPR', 'TPR', 'AUC'])

        # Save macro-average ROC curve data
        for fpr, tpr in zip(fpr_dict["macro"], tpr_dict["macro"]):
            writer.writerow(['Macro-average', fpr, tpr, roc_auc_dict["macro"]])

        # Save ROC curve data for other classes
        for i, class_name in enumerate(class_names):
            for fpr, tpr in zip(fpr_dict[i], tpr_dict[i]):
                writer.writerow([class_name, fpr, tpr, roc_auc_dict[i]])


def plot_roc_curve(labels, probs, num_classes=2, save_dir='./'):
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 10))
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为SimHei
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    labels_binarized = label_binarize(labels, classes=[i for i in range(num_classes)])
    if labels_binarized.shape[1] == 1:
        # 如果是二分类问题，手动添加另一列
        labels_binarized = np.hstack([1 - labels_binarized, labels_binarized])
    n_classes = labels_binarized.shape[1]
    print(f"labels_binarized.shape: {labels_binarized.shape}")
    print(f"probs.shape: {probs.shape}")

    # 计算每个类别的ROC曲线和AUC值
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    class_names = ['Normal', 'Thickened']

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(labels_binarized[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 计算宏平均ROC曲线和AUC值
    # 首先汇总所有的FPR
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # 在这些点上对ROC曲线进行插值
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # 计算平均的TPR
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # 保存ROC数据到CSV文件
    save_path_csv = Path(os.path.join(save_dir, 'roc_data.csv'))
    print(f"正在保存ROC数据到: {save_path_csv}")
    save_roc_data_to_csv(fpr, tpr, roc_auc, class_names, save_path_csv)

    # 绘制ROC曲线
    plt.plot(fpr["macro"], tpr["macro"],
             label='Macro-average ROC curve (area = {0:0.2f}%)'
                   ''.format(roc_auc["macro"] * 100),
             color='navy', linestyle=':', linewidth=2)

    colors = ['aqua', 'darkorange']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='{0} ROC curve (area = {1:0.2f}%)'
                       ''.format(class_names[i], roc_auc[i] * 100))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.gca().set_xticks(np.linspace(0.1, 1.0, 10))
    plt.gca().set_yticks(np.linspace(0, 1.0, 11))
    plt.xlabel('False Positive Rate (%)', fontsize=18)
    plt.ylabel('True Positive Rate (%)', fontsize=18)
    plt.title('ROC Curve for Each Class', fontsize=20)
    plt.plot([0, 1], [0, 1], color='black', linestyle='--', lw=2)

    plt.legend(loc="lower right", fontsize=18)
    plt.grid(False)
    plt.gca().spines['top'].set_visible(True)
    plt.gca().spines['right'].set_visible(True)

    plt.tight_layout()
    save_path_fig = Path(os.path.join(save_dir, 'ROC.pdf'))
    print(f"正在保存ROC曲线到: {save_path_fig}")
    plt.savefig(save_path_fig)
    plt.close()

def plot_pr_curve(labels, probs, num_classes=2, save_dir='./'):
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(8, 8))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为 SimHei
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    labels_binarized = label_binarize(labels, classes=[i for i in range(num_classes)])
    if labels_binarized.shape[1] == 1:
        # 如果是二分类问题，手动添加另一列
        labels_binarized = np.hstack([1 - labels_binarized, labels_binarized])
    n_classes = labels_binarized.shape[1]
    print(f"labels_binarized.shape: {labels_binarized.shape}")
    print(f"probs.shape: {probs.shape}")

    # 计算每个类别的 PR 曲线和平均精确度
    precision = dict()
    recall = dict()
    average_precision = dict()
    class_names = ['Normal', 'Thickened']

    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(labels_binarized[:, i], probs[:, i])
        average_precision[i] = average_precision_score(labels_binarized[:, i], probs[:, i])

    # 计算微平均 PR 曲线和平均精确度
    precision["micro"], recall["micro"], _ = precision_recall_curve(labels_binarized.ravel(), probs.ravel())
    average_precision["micro"] = average_precision_score(labels_binarized, probs, average="micro")

    # 绘制 PR 曲线
    plt.plot(recall["micro"], precision["micro"],
             label='Micro-average PR curve (area = {0:0.2f}%)'
                   ''.format(average_precision["micro"] * 100),
             color='gold', linestyle=':', linewidth=2)

    colors = ['aqua', 'darkorange']
    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label='{0} PR curve (area = {1:0.2f}%)'
                       ''.format(class_names[i], average_precision[i] * 100))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall', fontsize=20)
    plt.ylabel('Precision', fontsize=20)
    plt.title('Precision-Recall Curve for Each Class', fontsize=20)

    plt.legend(loc="lower left", fontsize=18)
    plt.grid(False)
    plt.tight_layout()
    save_path_fig = Path(os.path.join(save_dir, 'PR_curve.pdf'))
    print(f"正在保存 PR 曲线到: {save_path_fig}")
    plt.savefig(save_path_fig)
    plt.close()


def main(PATH, weight_path, model, save_path='./', include_0_9mm=True, batch_size=14,group=1):
    json_file = r"D:\data\15.json"
    if group == 1:
        group_name="train"
    elif group == 2:
        group_name = 'valid'
    elif group == 3:
        group_name = 'test'
    else:
        print(f"未知的 group 值：{group}")
        return

        # 更新保存路径
    save_path = os.path.join(save_path, group_name)

    # 确保保存目录存在
    os.makedirs(save_path, exist_ok=True)

    valid_dataset = NewEyeDataset.from_json(json_file=json_file,
                                            root_dir=PATH,
                                            group_value=group,
                                            include_0_9mm=include_0_9mm,
                                            transform=ValidTransform)

    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=4, pin_memory=True,
                              prefetch_factor=2)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()

    # 加载模型权重
    if weight_path and os.path.isfile(weight_path):
        checkpoint = torch.load(weight_path, map_location=device)
        if 'module.' in list(checkpoint.keys())[0]:
            # 如果模型是使用DataParallel训练的，需要去掉'module.'前缀
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k[7:]  # 去掉'module.'前缀
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"权重文件不存在：{weight_path}")
        return

    model.to(device)

    valid_loss, valid_acc, all_labels, all_preds, all_probs = validate_model(model, valid_loader, criterion)

    print(f'Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_acc:.4f}')
    # 打印标签分布
    print(f"验证集中标签的类别：{np.unique(all_labels)}")
    print(f"每个类别的样本数：{np.bincount(all_labels)}")
    print(f"labels.shape: {all_labels.shape}")
    print(f"probs.shape: {all_probs.shape}")

    # 生成并保存混淆矩阵
    plot_confusion_matrix(all_labels, all_preds, save_dir=save_path)

    # 生成并保存分类报告
    class_names = ['Normal', 'Thickened']
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    report_path = Path(os.path.join(save_path, 'classification_report.txt'))
    print(f"正在保存分类报告到: {report_path}")
    with open(report_path, 'w') as f:
        f.write(report)
    print("分类报告内容：")
    print(report)

    # 生成并保存ROC曲线
    plot_roc_curve(all_labels, all_probs, num_classes=2, save_dir=save_path)

    # 生成并保存 PR 曲线
    plot_pr_curve(all_labels, all_probs, num_classes=2, save_dir=save_path)


if __name__ == '__main__':
    PATH = r'D:\data\1'
    save_path = r'./2'
    weight_path = r'./SiameseSeResNeXtdropoutmodel_state_dict_0.8200_0.8400.pth'
    model = SiameseSeResNeXtdropout(dropout_p=0.05, spatial_dropout_p=0.05, out=2).to(device)
    batch_size = 60
    include_0_9mm = True
    groups = [1, 2, 3]
    for group in groups:
        if group==1:
            continue
        main(save_path=save_path, PATH=PATH, batch_size=batch_size, include_0_9mm=include_0_9mm,
             weight_path=weight_path, model=model, group=group)
