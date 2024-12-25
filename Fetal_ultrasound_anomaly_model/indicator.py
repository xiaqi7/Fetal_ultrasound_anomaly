import os
import json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, \
    confusion_matrix, precision_recall_curve
# from model import resnet34 as create_model
from model import resnext101_32x8d as create_model
from tqdm import tqdm
import seaborn as sns  # 导入seaborn用于绘制混淆矩阵
from sklearn.preprocessing import label_binarize

plt.rcParams['font.family'] = 'SimSun'  # 例如设置为宋体


# 计算并显示混淆矩阵、TP、FP、FN、TN
def display_confusion_matrix(true_labels, pred_labels, num_classes):
    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(num_classes)))
    print(f"混淆矩阵:\n{cm}")

    # 使用seaborn绘制热力图
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(num_classes), yticklabels=range(num_classes))
    plt.xlabel('预测类别')
    plt.ylabel('实际类别')
    plt.title('混淆矩阵')
    plt.show()

    tp = np.diag(cm)
    fn = cm.sum(axis=1) - tp
    fp = cm.sum(axis=0) - tp
    tn = cm.sum() - (fp + fn + tp)

    specificity = tn / (tn + fp)

    # 输出每个类别的TP、FN、FP、TN和Specificity
    for i in range(num_classes):
        print(f"类别 {i}:")
        print(f"  真阳性 (TP): {tp[i]}")
        print(f"  假阴性 (FN): {fn[i]}")
        print(f"  假阳性 (FP): {fp[i]}")
        print(f"  真阴性 (TN): {tn[i]}")
        print(f"  特异性 (Specificity): {specificity[i]:.4f}")
        print()



# 评估图像级别的性能指标，并绘制ROC曲线和P-R曲线
def evaluate_image_level(true_labels, pred_labels, pred_probs, num_classes):
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average='weighted')
    recall = recall_score(true_labels, pred_labels, average='weighted')
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    roc_auc = roc_auc_score(true_labels, pred_probs, multi_class='ovr')
    print(f"图像级别准确率: {accuracy:.4f}")
    print(f"图像级别精确率: {precision:.4f}")
    print(f"图像级别召回率: {recall:.4f}")
    print(f"图像级别F1分数: {f1:.4f}")
    print(f"图像级别ROC AUC: {roc_auc:.4f}")

    # 将标签进行One-vs-Rest二值化
    true_labels_bin = label_binarize(true_labels, classes=list(range(num_classes)))

    # 计算每个类别的 Precision-Recall 曲线
    precision_dict = dict()
    recall_dict = dict()
    for i in range(num_classes):
        precision_dict[i], recall_dict[i], _ = precision_recall_curve(true_labels_bin[:, i], pred_probs[:, i])

    # 计算 micro-average P-R 曲线
    precision_micro, recall_micro, _ = precision_recall_curve(true_labels_bin.ravel(), pred_probs.ravel())

    # 计算 macro-average P-R 曲线
    all_precision = np.unique(np.concatenate([precision_dict[i] for i in range(num_classes)]))
    mean_recall = np.zeros_like(all_precision)
    for i in range(num_classes):
        mean_recall += np.interp(all_precision, precision_dict[i][::-1], recall_dict[i][::-1])
    mean_recall /= num_classes

    # 绘制图像级别 P-R 曲线
    plt.figure()
    for i in range(num_classes):
        plt.plot(recall_dict[i], precision_dict[i], lw=2, label=f'类别{i} 图像级 P-R曲线')
    plt.plot(recall_micro, precision_micro, lw=2, linestyle='--', label='micro-average 图像级 P-R曲线')
    plt.plot(mean_recall, all_precision, lw=2, linestyle='--', label='macro-average 图像级 P-R曲线')
    plt.xlabel('召回率')
    plt.ylabel('精确率')
    plt.title('图像级 Precision-Recall 曲线')
    plt.legend(loc="lower left")
    plt.show()

    # 显示混淆矩阵和TP、FP、FN、TN
    display_confusion_matrix(true_labels, pred_labels, num_classes)


# 评估个体级别的性能指标，并绘制ROC曲线和P-R曲线
def evaluate_person_level(folder_probabilities, folder_labels, num_classes):
    pred_labels = []
    true_labels = []

    # 计算每个文件夹（个体级别预测）的平均概率
    for folder, probs in folder_probabilities.items():
        avg_prob = np.mean(probs, axis=0)  # 平均所有图片的概率
        pred_label = np.argmax(avg_prob)  # 使用平均概率预测类别
        pred_labels.append(pred_label)
        true_labels.append(folder_labels[folder])

    # 转换为 numpy 数组
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    pred_probs = np.array([np.mean(probs, axis=0) for probs in folder_probabilities.values()])

    # 打印调试信息
    print(f"true_labels: {true_labels}")
    print(f"pred_labels: {pred_labels}")

    # 计算准确率、精确率、召回率、F1 分数
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average='weighted',zero_division=0)
    recall = recall_score(true_labels, pred_labels, average='weighted')
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    roc_auc = roc_auc_score(true_labels, pred_probs, multi_class='ovr')

    print(f"个体级别准确率: {accuracy:.4f}")
    print(f"个体级别精确率: {precision:.4f}")
    print(f"个体级别召回率: {recall:.4f}")
    print(f"个体级别F1分数: {f1:.4f}")
    print(f"个体级别ROC AUC: {roc_auc:.4f}")

    # 将标签进行One-vs-Rest二值化
    true_labels_bin = label_binarize(true_labels, classes=list(range(num_classes)))

    # 计算每个类别的 Precision-Recall 曲线
    precision_dict = dict()
    recall_dict = dict()
    for i in range(num_classes):
        precision_dict[i], recall_dict[i], _ = precision_recall_curve(true_labels_bin[:, i], pred_probs[:, i])

    # 计算 micro-average P-R 曲线
    precision_micro, recall_micro, _ = precision_recall_curve(true_labels_bin.ravel(), pred_probs.ravel())

    # 计算 macro-average P-R 曲线
    all_precision = np.unique(np.concatenate([precision_dict[i] for i in range(num_classes)]))
    mean_recall = np.zeros_like(all_precision)
    for i in range(num_classes):
        mean_recall += np.interp(all_precision, precision_dict[i][::-1], recall_dict[i][::-1])
    mean_recall /= num_classes

    # 在同一张图上绘制个体级别 P-R 曲线
    plt.figure()
    for i in range(num_classes):
        plt.plot(recall_dict[i], precision_dict[i], lw=2, label=f'类别{i} 个体级 P-R曲线')
    plt.plot(recall_micro, precision_micro, lw=2, linestyle='--', label='micro-average 个体级 P-R曲线')
    plt.plot(mean_recall, all_precision, lw=2, linestyle='--', label='macro-average 个体级 P-R曲线')
    plt.xlabel('召回率')
    plt.ylabel('精确率')
    plt.title('个体级 Precision-Recall 曲线')
    plt.legend(loc="lower left")
    plt.show()

    # 显示混淆矩阵和TP、FP、FN、TN
    display_confusion_matrix(true_labels, pred_labels, num_classes)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用 {device} 设备.")

    num_classes = 4  # 四分类任务
    img_size = 224
    data_transform = transforms.Compose(
        [transforms.Resize(int(img_size * 1.14)),
         transforms.CenterCrop(img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # 创建模型
    model = create_model(num_classes=num_classes).to(device)
    model_weight_path = "weights/fold_resnext101_32x8d/best_model_fold2_0.5351.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    # 加载类别索引
    json_path = 'class_indices_4.json'
    assert os.path.exists(json_path), f"文件: '{json_path}' 不存在."
    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # 测试集目录结构: 类别 -> 个体（文件夹）-> 图片
    test_dir = r"E:\qy-chaoshe\dataset\test_set"
    assert os.path.exists(test_dir), f"测试数据集路径 '{test_dir}' 不存在."

    true_labels = []
    pred_labels = []
    pred_probs = []
    folder_probabilities = {}
    folder_labels = {}

    # 遍历所有类别（类）
    for class_name in os.listdir(test_dir):
        class_path = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        class_index = list(class_indict.values()).index(class_name)
        print(f"处理类别 {class_name}...")

        # 遍历类内的每个个体（文件夹）
        for person_folder in os.listdir(class_path):
            person_path = os.path.join(class_path, person_folder)
            if not os.path.isdir(person_path):
                continue

            folder_probabilities[person_folder] = []
            folder_labels[person_folder] = class_index
            print(f"  处理个体 {person_folder}...")

            # 遍历个体文件夹内的每张图片
            for img_file in tqdm(os.listdir(person_path), desc=f"处理图片 ({person_folder})"):
                img_path = os.path.join(person_path, img_file)
                img = Image.open(img_path)
                img = data_transform(img)
                img = torch.unsqueeze(img, dim=0).to(device)

                with torch.no_grad():
                    output = torch.squeeze(model(img)).cpu()
                    prob = torch.softmax(output, dim=0).numpy()
                    pred_label = np.argmax(prob)

                # 存储图像级别的结果
                true_labels.append(class_index)
                pred_labels.append(pred_label)
                pred_probs.append(prob)

                # 存储个体级别的概率
                folder_probabilities[person_folder].append(prob)

    # 转换为numpy数组以便计算指标
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    pred_probs = np.array(pred_probs)

    # 图像级别评估
    evaluate_image_level(true_labels, pred_labels, pred_probs, num_classes)

    # 个体级别评估
    evaluate_person_level(folder_probabilities, folder_labels, num_classes)

if __name__ == '__main__':
    main()

