import os
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
from sklearn.preprocessing import label_binarize
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns

from my_dataset import MyDataSet
from model import resnet34 as create_model
from utils import create_lr_scheduler, train_one_epoch
from sklearn.metrics import classification_report
import numpy as np
import sys
from tqdm import tqdm
from sklearn.exceptions import UndefinedMetricWarning
import warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
# 设置全局字体
plt.rcParams['font.family'] = 'SimSun'  # 宋体的英文名称

# 设置结果文件夹路径
result_folder = "result"

def get_patient_data_paths(data_path):
    """
    获取患者的图像路径和标签
    """
    patients_paths = []
    patients_labels = []
    patient_ids = []

    for class_index, class_name in enumerate(os.listdir(data_path)):
        class_path = os.path.join(data_path, class_name)
        if not os.path.isdir(class_path):
            continue

        for patient_folder in os.listdir(class_path):
            patient_path = os.path.join(class_path, patient_folder)
            if not os.path.isdir(patient_path):
                continue

            # 获取图像路径
            image_paths = [os.path.join(patient_path, img) for img in os.listdir(patient_path) if img.endswith(('.jpg', '.png'))]
            patients_paths.append(image_paths)
            patients_labels.append(class_index)
            patient_ids.append(patient_folder)

    return patients_paths, patients_labels, patient_ids


def leave_one_out_split(patients_paths, patients_labels):
    """
    留一法的交叉验证划分
    """
    for i in range(len(patients_paths)):
        train_idx = list(range(len(patients_paths)))
        train_idx.remove(i)
        val_idx = [i]
        yield train_idx, val_idx


from sklearn.metrics import roc_curve, precision_recall_curve, auc

def plot_pr_curve(y_true, y_scores, num_classes, class_names, title="PR曲线", title1="PR曲线"):
    """
    绘制并显示 PR (Precision-Recall) 曲线，同时绘制宏平均和微平均曲线
    :param y_true: 真实标签，二进制形式
    :param y_scores: 预测概率
    :param num_classes: 类别数
    :param class_names: 类别名称列表
    :param title: 图像标题
    :param title1: 用于生成文件名的部分
    """
    precision = {}
    recall = {}
    pr_auc = {}

    # 逐类绘制 PR 曲线
    for i in range(num_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_scores[:, i])
        pr_auc[i] = auc(recall[i], precision[i])

        # 为每个类别创建单独的图像
        plt.figure()
        plt.plot(recall[i], precision[i], label=f'{class_names[i]} (AUC = {pr_auc[i]:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f"{title} - {class_names[i]}")
        plt.legend(loc='lower left')

        # 保存单个类别的图像
        save_path = os.path.join(result_folder, f"{title1.replace(' ', '_')}_{class_names[i]}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()

    # 计算微平均 PR 曲线
    precision_micro, recall_micro, _ = precision_recall_curve(y_true.ravel(), y_scores.ravel())
    pr_auc_micro = auc(recall_micro, precision_micro)

    # 为微平均创建单独的图像
    plt.figure()
    plt.plot(recall_micro, precision_micro, label=f'Micro-average (AUC = {pr_auc_micro:.2f})', linestyle='--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f"{title} - Micro-average")
    plt.legend(loc='lower left')

    # 保存微平均的图像
    save_path = os.path.join(result_folder, f"{title1.replace(' ', '_')}_micro_average.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

    # 计算宏平均 PR 曲线
    all_recalls = np.unique(np.concatenate([recall[i] for i in range(num_classes)]))
    mean_precision = np.zeros_like(all_recalls)
    for i in range(num_classes):
        mean_precision += np.interp(all_recalls, np.flip(recall[i]), np.flip(precision[i]))
    mean_precision /= num_classes
    pr_auc_macro = auc(all_recalls, mean_precision)

    # 为宏平均创建单独的图像
    plt.figure()
    plt.plot(all_recalls, mean_precision, label=f'Macro-average (AUC = {pr_auc_macro:.2f})', linestyle='--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f"{title} - Macro-average")
    plt.legend(loc='lower left')

    # 保存宏平均的图像
    save_path = os.path.join(result_folder, f"{title1.replace(' ', '_')}_macro_average.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

    # 创建包含所有类别的 PR 曲线的图像（不包括微平均和宏平均）
    plt.figure()
    for i in range(num_classes):
        plt.plot(recall[i], precision[i], label=f'{class_names[i]} (AUC = {pr_auc[i]:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f"{title} - All Classes (No Averages)")
    plt.legend(loc='lower left')

    # 保存所有类别的图像（不包括微平均和宏平均）
    save_path = os.path.join(result_folder, f"{title1.replace(' ', '_')}_all_classes_no_averages.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

    # 创建包含所有类别的 PR 曲线以及微平均和宏平均的图像
    plt.figure()
    for i in range(num_classes):
        plt.plot(recall[i], precision[i], label=f'{class_names[i]} (AUC = {pr_auc[i]:.2f})')
    plt.plot(recall_micro, precision_micro, label=f'Micro-average (AUC = {pr_auc_micro:.2f})', linestyle='--')
    plt.plot(all_recalls, mean_precision, label=f'Macro-average (AUC = {pr_auc_macro:.2f})', linestyle='--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc='lower left')

    # 保存所有类别的图像（包括微平均和宏平均）
    save_path = os.path.join(result_folder, f"{title1.replace(' ', '_')}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()  # 关闭图像以释放内存

def plot_roc_curve(y_true, y_scores, num_classes, class_names, title="ROC曲线", title1="ROC曲线"):
    """
    绘制并显示 ROC 曲线，同时绘制宏平均和微平均曲线
    :param y_true: 真实标签，二进制形式
    :param y_scores: 预测概率
    :param num_classes: 类别数
    :param class_names: 类别名称列表
    :param title: 图像标题
    :param title1: 用于生成文件名的部分
    """
    fpr = {}
    tpr = {}
    roc_auc = {}

    # 逐类绘制 ROC 曲线
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        # 为每个类别创建单独的图像
        plt.figure()
        plt.plot(fpr[i], tpr[i], label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f"{title} - {class_names[i]}")
        plt.legend(loc='lower right')

        # 保存单个类别的图像
        save_path = os.path.join(result_folder, f"{title1.replace(' ', '_')}_{class_names[i]}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()

    # 计算微平均 ROC 曲线
    fpr_micro, tpr_micro, _ = roc_curve(y_true.ravel(), y_scores.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)

    # 为微平均创建单独的图像
    plt.figure()
    plt.plot(fpr_micro, tpr_micro, label=f'Micro-average (AUC = {roc_auc_micro:.2f})', linestyle='--')
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"{title} - Micro-average")
    plt.legend(loc='lower right')

    # 保存微平均的图像
    save_path = os.path.join(result_folder, f"{title1.replace(' ', '_')}_micro_average.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

    # 计算宏平均 ROC 曲线
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= num_classes
    roc_auc_macro = auc(all_fpr, mean_tpr)

    # 为宏平均创建单独的图像
    plt.figure()
    plt.plot(all_fpr, mean_tpr, label=f'Macro-average (AUC = {roc_auc_macro:.2f})', linestyle='--')
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"{title} - Macro-average")
    plt.legend(loc='lower right')

    # 保存宏平均的图像
    save_path = os.path.join(result_folder, f"{title1.replace(' ', '_')}_macro_average.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

    # 创建包含所有类别的 ROC 曲线的图像（不包括微平均和宏平均）
    plt.figure()
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"{title} - All Classes (No Averages)")
    plt.legend(loc='lower right')

    # 保存所有类别的图像（不包括微平均和宏平均）
    save_path = os.path.join(result_folder, f"{title1.replace(' ', '_')}_all_classes_no_averages.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

    # 创建包含所有类别的 ROC 曲线以及微平均和宏平均的图像
    plt.figure()
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    plt.plot(fpr_micro, tpr_micro, label=f'Micro-average (AUC = {roc_auc_micro:.2f})', linestyle='--')
    plt.plot(all_fpr, mean_tpr, label=f'Macro-average (AUC = {roc_auc_macro:.2f})', linestyle='--')
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')

    # 保存所有类别的图像（包括微平均和宏平均）
    save_path = os.path.join(result_folder, f"{title1.replace(' ', '_')}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()  # 关闭图像以释放内存

# def plot_roc_curve(y_true, y_scores, num_classes,class_names, title="ROC曲线",title1="ROC曲线"):
#     """
#     绘制并显示 ROC 曲线，同时绘制宏平均和微平均曲线
#     :param y_true: 真实标签，二进制形式
#     :param y_scores: 预测概率
#     :param num_classes: 类别数
#     :param title: 图像标题
#     """
#     fpr = {}
#     tpr = {}
#     roc_auc = {}
#
#     plt.figure()
#
#     # 逐类绘制 ROC 曲线
#     for i in range(num_classes):
#         fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_scores[:, i])
#         roc_auc[i] = auc(fpr[i], tpr[i])
#         plt.plot(fpr[i], tpr[i], label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
#
#     # 计算微平均 ROC 曲线
#     fpr_micro, tpr_micro, _ = roc_curve(y_true.ravel(), y_scores.ravel())
#     roc_auc_micro = auc(fpr_micro, tpr_micro)
#     plt.plot(fpr_micro, tpr_micro, label=f'Micro-average (AUC = {roc_auc_micro:.2f})', linestyle='--')
#
#     # 计算宏平均 ROC 曲线
#     all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
#     mean_tpr = np.zeros_like(all_fpr)
#     for i in range(num_classes):
#         mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
#     mean_tpr /= num_classes
#     roc_auc_macro = auc(all_fpr, mean_tpr)
#     plt.plot(all_fpr, mean_tpr, label=f'Macro-average (AUC = {roc_auc_macro:.2f})', linestyle='--')
#
#     plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title(title)
#     plt.legend(loc='lower right')
#     # plt.show()
#     # 生成保存路径
#     save_path = os.path.join(result_folder, f"{title1.replace(' ', '_')}.png")
#
#     # 确保保存路径的目录存在
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#
#     # 保存图像
#     plt.savefig(save_path)
#     plt.close()  # 关闭图像以释放内存




def calculate_class_weights(labels, num_classes):
    """
    计算类别权重，处理类别不平衡问题
    """
    class_counts = np.bincount(labels, minlength=num_classes)
    total_samples = len(labels)
    class_weights = total_samples / (num_classes * class_counts)
    return torch.tensor(class_weights, dtype=torch.float)


def evaluate_image_level(model, data_loader, device, loss_function):
    """
    评估模型并返回图片级别的验证损失、准确率、预测概率和真实标签
    """
    model.eval()
    val_loss = 0.0
    correct = 0.0
    total = 0.0
    image_probs = []
    image_labels = []

    with torch.no_grad():
        data_loader = tqdm(data_loader, file=sys.stdout)
        for step, data in enumerate(data_loader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            # 获取模型输出
            outputs = model(images)
            loss = loss_function(outputs, labels)
            val_loss += loss.item()

            # 获取每个样本的预测概率
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            image_probs.extend(probs)

            # 获取预测类别
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            image_labels.extend(labels.cpu().numpy())

            # 更新平均损失和准确率
            avg_val_loss = val_loss / (step + 1)
            avg_val_acc = correct / total

            # 更新进度条描述
            data_loader.set_description(f"[validation] loss: {avg_val_loss:.3f}, acc: {avg_val_acc:.3f}")

    val_acc = correct / total
    val_loss = val_loss / len(data_loader)

    return val_loss, val_acc, np.array(image_probs), np.array(image_labels)

import logging

# # 配置日志记录
# logging.basicConfig(filename='individual_results.log', level=logging.INFO,
#                     format='%(asctime)s - %(levelname)s - %(message)s')

# 创建日志记录器
logger1 = logging.getLogger('logger1')
logger2 = logging.getLogger('logger2')

# 设置日志级别
logger1.setLevel(logging.INFO)
logger2.setLevel(logging.INFO)

# 创建文件处理器
file_handler1 = logging.FileHandler('log_file1.log')
file_handler2 = logging.FileHandler('log_file2.log')

# 设置日志格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler1.setFormatter(formatter)
file_handler2.setFormatter(formatter)

# 将文件处理器添加到日志记录器
logger1.addHandler(file_handler1)
logger2.addHandler(file_handler2)

def select_best_individual_results1(val_probs, val_labels, patient_ids, num_classes, class_names):
    """
    从验证结果中按人头计算每个个体的最佳验证结果（即：每个个体的预测概率和真实标签）
    :param val_probs: 验证集中所有图片的预测概率
    :param val_labels: 验证集中所有图片的真实标签
    :param patient_ids: 每个图片对应的个体ID
    :return: 个体级别的预测结果（按人头）
    """
    individual_probs = {}
    individual_labels = {}

    for i, patient_id in enumerate(patient_ids):
        if patient_id not in individual_probs:
            individual_probs[patient_id] = []
            individual_labels[patient_id] = []

        individual_probs[patient_id].append(val_probs[i])
        individual_labels[patient_id].append(val_labels[i])

    # 计算每个个体的平均预测概率
    avg_individual_probs = []
    avg_individual_labels = []
    # 假设 individual_probs 和 individual_labels 已经定义并填充了数据
    for patient_id in individual_probs:
        # 计算每个类别的平均预测概率
        avg_prob = np.mean(individual_probs[patient_id], axis=0)

        # 显示每个类别的预测概率
        print(f"Patient ID {patient_id} Average Prediction Probabilities:")
        for class_idx, prob in enumerate(avg_prob):
            print(f"  {class_names[class_idx]}: {prob:.4f}")

        # # 记录每个类别的预测概率
        # logging.info(f"Patient ID {patient_id} Average Prediction Probabilities:")
        # for class_idx, prob in enumerate(avg_prob):
        #     logging.info(f"  {class_names[class_idx]}: {prob:.4f}")
        # 记录每个类别的预测概率
        logger1.info(f"Patient ID {patient_id} Average Prediction Probabilities:")
        for class_idx, prob in enumerate(avg_prob):
            logger1.info(f"  {class_names[class_idx]}: {prob:.4f}")


        # 将平均概率添加到最终结果列表中
        avg_individual_probs.append(avg_prob)

        # 假设所有图片的真实标签相同，获取并添加真实标签
        true_label = individual_labels[patient_id][0]
        avg_individual_labels.append(true_label)
        print(f"true label : {true_label}")
        logger1.info(f"  True label: {true_label}")

    return np.array(avg_individual_probs), np.array(avg_individual_labels)
# # 配置日志记录
# logging.basicConfig(filename='individual_results2.log', level=logging.INFO,
#                     format='%(asctime)s - %(levelname)s - %(message)s')

def select_best_individual_results2(val_probs, val_labels, patient_ids, num_classes, class_names):
    """
    从验证结果中按人头计算每个个体的最佳验证结果（即：每个个体的类别）
    :param val_probs: 验证集中所有图片的预测概率
    :param val_labels: 验证集中所有图片的真实标签
    :param patient_ids: 每个图片对应的个体ID
    :return: 个体级别的预测结果（按人头）
    """
    individual_predictions = {}
    individual_labels = {}

    for i, patient_id in enumerate(patient_ids):
        if patient_id not in individual_predictions:
            individual_predictions[patient_id] = []
            individual_labels[patient_id] = []

        # 获取每张图片的预测类别
        preds = np.argmax(val_probs[i])
        individual_predictions[patient_id].append(preds)
        individual_labels[patient_id].append(val_labels[i])

    # 计算每个个体的最终类别（最多类别）

    final_individual_preds = []
    final_individual_labels = []
    individual_class_frequencies = []  # 存储每个类别的频率

    # 假设 individual_predictions 和 individual_labels 已经定义并填充了数据
    for patient_id in individual_predictions:
        # 初始化一个计数器
        class_counts = np.zeros(num_classes)

        # 遍历预测结果，更新计数器
        for prediction in individual_predictions[patient_id]:
            class_counts[prediction] += 1

        # 计算每个类别的频率
        total_count = np.sum(class_counts)
        if total_count > 0:
            class_frequencies = class_counts / total_count
        else:
            # 如果没有预测，则可以设置默认频率或者处理这种情况
            class_frequencies = np.ones(num_classes) / num_classes

        # 输出每个类别的频率
        print(f"Patient ID {patient_id} Individual Prediction Probabilities:")
        for class_idx in range(num_classes):
            print(f"  {class_names[class_idx]}: {class_frequencies[class_idx]:.4f}")

        # 记录每个类别的预测概率
        logger2.info(f"Patient ID {patient_id} Individual Prediction Probabilities:")
        for class_idx in range(num_classes):
            logger2.info(f"  {class_names[class_idx]}: {class_frequencies[class_idx]:.4f}")

        # 找出最常见的预测值
        most_common_pred = np.argmax(class_counts)

        # 将最常见的预测值添加到最终结果列表中
        final_individual_preds.append(most_common_pred)

        # 保存每个类别的频率
        individual_class_frequencies.append(class_frequencies)

        # 假设所有图片的真实标签相同，获取并添加真实标签
        true_label = individual_labels[patient_id][0]
        final_individual_labels.append(true_label)
        print(f"true label : {true_label}")
        logger2.info(f"  True label: {true_label}")

    return np.array(individual_class_frequencies), np.array(final_individual_labels)

def plot_learning_rate(lr_history):
    """
    绘制学习率变化曲线
    :param lr_history: 每个训练周期的学习率历史记录
    """
    plt.figure(figsize=(10, 6))
    plt.plot(lr_history, label='Learning Rate', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Change Curve')
    plt.legend()
    plt.grid()
    # plt.show()
    save_path = "result/lr_curve.png"
    plt.savefig(save_path)


import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, class_names, title,title1):
    """
    绘制并显示混淆矩阵的热力图
    :param cm: 混淆矩阵
    :param class_names: 类别名称列表
    :param title: 标题
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predict Label")
    plt.ylabel("Real Label")
    plt.title(title)
    # plt.show()
    # 生成保存路径
    save_path = os.path.join(result_folder, f"{title1.replace(' ', '_')}.png")

    # 确保保存路径的目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 保存图像
    plt.savefig(save_path)
    plt.close()  # 关闭图像以释放内存

def main(args):

    # class_names = ['Anencephaly', 'Encephalocele_meningocele', 'Holoprosencephaly', 'Rachischisis']
    # class_names = ['Anencephaly','Cerebellum', 'Encephalocele_meningocele', 'Holoprosencephaly',
    #                'Lateral_brain', 'Rachischisis','Spine','Thalamus']
    # class_names = ['abnormal', 'normal']
    class_names = ['Anencephaly', 'Encephalocele_meningocele', 'Holoprosencephaly', 'normal','Rachischisis']

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}.")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    patients_paths, patients_labels, patient_ids = get_patient_data_paths(args.data_path)

    img_size = 224
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 存储所有折的图片级别和个体级别结果
    all_image_probs = []
    all_image_labels = []
    all_individual_probs = []
    all_individual_labels = []
    all_individual_probs2 = []
    all_individual_labels2 = []

    for fold, (train_idx, val_idx) in enumerate(leave_one_out_split(patients_paths, patients_labels)):
        print(f"留一法 第 {fold + 1}/{len(patients_paths)} 折")

        train_image_paths = [img_path for idx in train_idx for img_path in patients_paths[idx]]
        train_labels = [patients_labels[idx] for idx in train_idx for _ in patients_paths[idx]]
        val_image_paths = [img_path for idx in val_idx for img_path in patients_paths[idx]]
        val_labels = [patients_labels[idx] for idx in val_idx for _ in patients_paths[idx]]
        val_patient_ids = [patient_ids[idx] for idx in val_idx for _ in patients_paths[idx]]

        train_dataset = MyDataSet(images_path=train_image_paths,
                                  images_class=train_labels,
                                  transform=data_transform["train"])
        val_dataset = MyDataSet(images_path=val_image_paths,
                                images_class=val_labels,
                                transform=data_transform["val"])

        batch_size = args.batch_size
        nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 4])
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   num_workers=nw,
                                                   collate_fn=train_dataset.collate_fn)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 num_workers=nw,
                                                 collate_fn=val_dataset.collate_fn)

        model = create_model(num_classes=args.num_classes).to(device)

        if args.weights != "":
            assert os.path.exists(args.weights), "权重文件: '{}' 不存在.".format(args.weights)
            weights_dict = torch.load(args.weights, map_location=device)
            if "model" in weights_dict:
                weights_dict = weights_dict["model"]
            weights_dict = {k: v for k, v in weights_dict.items() if not k.startswith('fc')}
            model.load_state_dict(weights_dict, strict=False)

        in_channel = model.fc.in_features
        model.fc = torch.nn.Linear(in_channel, args.num_classes)
        model.to(device)

        if args.freeze_layers:
            for name, para in model.named_parameters():
                if "head" not in name:
                    para.requires_grad_(False)

        pg = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)
        lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True, warmup_epochs=1)

        class_weights = calculate_class_weights(np.array(train_labels), args.num_classes).to(device)
        loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)

        patience = 5
        no_improvement_epochs = 0

        best_val_acc = 0.0
        best_epoch_val_probs = []
        best_epoch_val_labels = []
        best_epoch_patient_ids = []
        lr_history = []  # 用于存储每个训练周期的学习率
        for epoch in range(args.epochs):
            train_loss, train_acc = train_one_epoch(model=model,
                                                    lossfunction=loss_function,
                                                    optimizer=optimizer,
                                                    data_loader=train_loader,
                                                    device=device,
                                                    epoch=epoch,
                                                    lr_scheduler=lr_scheduler)

            # 图片级别验证
            val_loss, val_acc, val_probs, val_labels = evaluate_image_level(model=model,
                                                                           data_loader=val_loader,
                                                                           device=device,
                                                                           loss_function=loss_function)

            # 如果当前轮次验证准确率最高，则保存这一轮的验证结果
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_epoch_val_probs = val_probs  # 保存图片级别的预测概率
                best_epoch_val_labels = val_labels  # 保存图片级别的真实标签
                best_epoch_patient_ids = val_patient_ids  # 保存个体ID
            #
            # if val_acc > best_val_acc:
            #     best_val_acc = val_acc
                no_improvement_epochs = 0
                torch.save(model.state_dict(), f"weights/best_model_fold{fold+1}.pth")
                print(f"保存最佳模型: epoch {epoch}, val_acc: {val_acc:.4f}")
            else:
                no_improvement_epochs += 1
                if no_improvement_epochs >= patience:
                    print(f"早停: 在 epoch {epoch} 验证集准确率未提升")
                    break
                    
            # 记录当前学习率
            lr_history.append(optimizer.param_groups[0]['lr'])
        # # 绘制学习率变化
        # plot_learning_rate(lr_history)

        # 记录每折的图片级别最优结果
        all_image_probs.extend(best_epoch_val_probs)
        all_image_labels.extend(best_epoch_val_labels)

        # 计算每折的个体级别最优结果
        individual_probs, individual_labels = select_best_individual_results1(best_epoch_val_probs,
                                                                             best_epoch_val_labels,
                                                                             best_epoch_patient_ids,
                                                                             args.num_classes,
                                                                             class_names)
        individual_probs2, individual_labels2 = select_best_individual_results2(best_epoch_val_probs,
                                                                              best_epoch_val_labels,
                                                                              best_epoch_patient_ids,
                                                                              args.num_classes,
                                                                              class_names)


        all_individual_probs.extend(individual_probs)
        all_individual_labels.extend(individual_labels)

        all_individual_probs2.extend(individual_probs2)
        all_individual_labels2.extend(individual_labels2)
    # # 绘制学习率变化
    # plot_learning_rate(lr_history)

    # 图片级别的最终指标
    all_image_labels = np.array(all_image_labels)
    all_image_probs = np.array(all_image_probs)

    accuracy = accuracy_score(all_image_labels, np.argmax(all_image_probs, axis=1))
    precision = precision_score(all_image_labels, np.argmax(all_image_probs, axis=1), average='macro', zero_division=0)
    recall = recall_score(all_image_labels, np.argmax(all_image_probs, axis=1), average='macro', zero_division=0)
    f1 = f1_score(all_image_labels, np.argmax(all_image_probs, axis=1), average='macro', zero_division=0)

    # 计算并绘制图片级别的混淆矩阵
    cm_image = confusion_matrix(all_image_labels, np.argmax(all_image_probs, axis=1))
    plot_confusion_matrix(cm_image, class_names=[f"  {class_names[i]}" for i in range(args.num_classes)], title="Confusion Matrix",title1="tp Confusion Matrix")

    # 检查是否有多个类别以避免 ROC AUC 错误
    if len(np.unique(all_image_labels)) > 1:

        auroc = roc_auc_score(label_binarize(all_image_labels, classes=np.arange(args.num_classes)),
                              all_image_probs, average='macro', multi_class='ovr')
        print(f"图片级别的AUROC: {auroc:.4f}")
        logger1.info(f"Image-level_AUROC: {auroc:.4f}")
    else:
        print("图片级别的AUROC无法计算：只有一个类存在。")
        logger1.info("Image-level_AUROC无法计算：只有一个类存在。")

    print(f"图片级别的准确率: {accuracy:.4f}")
    print(f"图片级别的精确率: {precision:.4f}")
    print(f"图片级别的召回率: {recall:.4f}")
    print(f"图片级别的F1值: {f1:.4f}")
    logger1.info(f"Image-level_accuracy: {accuracy:.4f}")
    logger1.info(f"Image-level_precision: {precision:.4f}")
    logger1.info(f"Image-level_recall: {recall:.4f}")
    logger1.info(f"Image-level_F1: {f1:.4f}")

    # 绘制图片级别的 ROC 和 PR 曲线（包括微平均和宏平均）
    if len(np.unique(all_image_labels)) > 1:
        y_true_bin = label_binarize(all_image_labels, classes=np.arange(args.num_classes))
        plot_roc_curve(y_true_bin, all_image_probs, args.num_classes,class_names, title="ROC curve",title1="tp ROC curve")
        plot_pr_curve(y_true_bin, all_image_probs, args.num_classes,class_names, title="PR curve",title1="tp PR curve")

    # 个体级别的最终指标
    all_individual_labels = np.array(all_individual_labels)
    all_individual_probs = np.array(all_individual_probs)

    accuracy = accuracy_score(all_individual_labels, np.argmax(all_individual_probs, axis=1))
    precision = precision_score(all_individual_labels, np.argmax(all_individual_probs, axis=1), average='macro', zero_division=0)
    recall = recall_score(all_individual_labels, np.argmax(all_individual_probs, axis=1), average='macro', zero_division=0)
    f1 = f1_score(all_individual_labels, np.argmax(all_individual_probs, axis=1), average='macro', zero_division=0)

    # 计算并绘制个体级别的混淆矩阵
    cm_individual = confusion_matrix(all_individual_labels, np.argmax(all_individual_probs, axis=1))
    plot_confusion_matrix(cm_individual, class_names=[f"{class_names[i]}" for i in range(args.num_classes)], title="Confusion Matrix",title1="gt1 Confusion Matrix")

    # 检查是否有多个类别以避免 ROC AUC 错误
    if len(np.unique(all_individual_labels)) > 1:
        auroc = roc_auc_score(label_binarize(all_individual_labels, classes=np.arange(args.num_classes)),
                              all_individual_probs, average='macro', multi_class='ovr')
        print(f"个体级别的AUROC: {auroc:.4f}")
        logger1.info(f"Individual-level_AUROC: {auroc:.4f}")
    else:
        print("个体级别的AUROC无法计算：只有一个类存在。")
        logger1.info(f"Individual-level_AUROC: {auroc:.4f}")

    print(f"个体级别的准确率: {accuracy:.4f}")
    print(f"个体级别的精确率: {precision:.4f}")
    print(f"个体级别的召回率: {recall:.4f}")
    print(f"个体级别的F1值: {f1:.4f}")
    logger1.info(f"Individual-level_accuracy: {accuracy:.4f}")
    logger1.info(f"Individual-level_precision: {precision:.4f}")
    logger1.info(f"Individual-level_recall: {recall:.4f}")
    logger1.info(f"Individual-level_F1: {f1:.4f}")

    # 绘制个体级别的 ROC 和 PR 曲线（包括微平均和宏平均）
    if len(np.unique(all_individual_labels)) > 1:
        y_true_bin = label_binarize(all_individual_labels, classes=np.arange(args.num_classes))
        plot_roc_curve(y_true_bin, all_individual_probs, args.num_classes,class_names, title="ROC curve",title1="gt1 ROC curve")
        plot_pr_curve(y_true_bin, all_individual_probs, args.num_classes,class_names, title="PR curve",title1="gt1 PR curve")

# 个体级别2的最终指标
    all_individual_labels2 = np.array(all_individual_labels2)
    all_individual_probs2 = np.array(all_individual_probs2)

    accuracy2 = accuracy_score(all_individual_labels2, np.argmax(all_individual_probs2, axis=1))
    precision2 = precision_score(all_individual_labels2, np.argmax(all_individual_probs2, axis=1), average='macro', zero_division=0)
    recall2 = recall_score(all_individual_labels2, np.argmax(all_individual_probs2, axis=1), average='macro', zero_division=0)
    f12 = f1_score(all_individual_labels2, np.argmax(all_individual_probs2, axis=1), average='macro', zero_division=0)

    # 计算并绘制个体级别的混淆矩阵
    cm_individual2 = confusion_matrix(all_individual_labels2, np.argmax(all_individual_probs2, axis=1))
    plot_confusion_matrix(cm_individual2, class_names=[f"{class_names[i]}" for i in range(args.num_classes)], title="Confusion Matrix",title1="gt2 Confusion Matrix")

    # 检查是否有多个类别以避免 ROC AUC 错误
    if len(np.unique(all_individual_labels2)) > 1:
        auroc2 = roc_auc_score(label_binarize(all_individual_labels2, classes=np.arange(args.num_classes)),
                              all_individual_probs2, average='macro', multi_class='ovr')
        print(f"个体级别2的AUROC: {auroc2:.4f}")
        logger2.info(f"Individual-level2_AUROC: {auroc2:.4f}")
    else:
        print("个体级别2的AUROC无法计算：只有一个类存在。")
        logger2.info("Individual-level2_AUROC无法计算：只有一个类存在。")

    print(f"个体级别2的准确率: {accuracy2:.4f}")
    print(f"个体级别2的精确率: {precision2:.4f}")
    print(f"个体级别2的召回率: {recall2:.4f}")
    print(f"个体级别2的F1值: {f12:.4f}")
    logger2.info(f"Individual-level2_accuracy2: {accuracy2:.4f}")
    logger2.info(f"Individual-level2_precision2: {precision2:.4f}")
    logger2.info(f"Individual-level2_recall2: {recall2:.4f}")
    logger2.info(f"Individual-level2_F1: {f12:.4f}")

    # 绘制个体级别的 ROC 和 PR 曲线（包括微平均和宏平均）
    if len(np.unique(all_individual_labels)) > 1:
        y_true_bin = label_binarize(all_individual_labels, classes=np.arange(args.num_classes))
        plot_roc_curve(y_true_bin, all_individual_probs, args.num_classes,class_names, title="ROC curve",title1="gt2 ROC curve")
        plot_pr_curve(y_true_bin, all_individual_probs, args.num_classes,class_names, title="PR curve",title1="gt2 PR curve")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--wd', type=float, default=5e-2)
    parser.add_argument('--data-path', type=str,
                        default=r"E:\qy-chaoshe\dataset\train_set_yuan_liu_5")
    parser.add_argument('--weights', type=str, default='./resnet34-b627a593.pth',
                        help='预训练权重路径')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='设备ID (如 0 或 0,1 或 cpu)')

    opt = parser.parse_args()
    main(opt)