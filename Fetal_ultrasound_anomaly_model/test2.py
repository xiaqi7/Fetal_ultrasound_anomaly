import time
import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from model import resnet34 as create_model

# 定义类别名称
# class_names = ['Anencephaly', 'Encephalocele_meningocele', 'Holoprosencephaly', 'normal', 'Rachischisis']
class_names = ['Anencephaly', 'Encephalocele_meningocele', 'Holoprosencephaly', 'Rachischisis']
# 加载和预处理输入图像
def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    img_tensor = transform(image).to(device)
    return img_tensor

# 对单张图片进行预测
def predict_single_image(model, image_path):
    img_tensor = load_image(image_path)

    # 获取预测结果
    model.eval()
    with torch.no_grad():
        output = model(img_tensor.unsqueeze(0))
        predicted_class = torch.argmax(output, dim=1).item()
        predicted_probabilities = torch.nn.functional.softmax(output, dim=1).squeeze().cpu().numpy()

    # 返回预测类别和概率
    return predicted_class, predicted_probabilities

# 对测试集文件夹进行预测
def predict_image_folder(folder_path, weights_base_path):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))  # 按数字顺序排序

    start_time = time.time()  # 记录开始时间

    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(folder_path, image_file)

        # 动态生成权重文件名
        weight_number = idx + 1
        weights_path = weights_base_path.replace("1.pth", f"{weight_number}.pth")

        # 加载权重
        model = create_model(num_classes=4)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.to(device)
        model.eval()

        # 预测单张图片
        predicted_class, predicted_probabilities = predict_single_image(model, image_path)

        # 打印每张图片的预测结果
        print(f"Image: {image_file}")
        print(f"  Predicted class: {class_names[predicted_class]}")
        print("  Class probabilities:")
        for i, prob in enumerate(predicted_probabilities):
            print(f"    {class_names[i]}: {prob:.4f}")
        print(f"  Using weights: {weights_path}\n")

    end_time = time.time()  # 记录结束时间

    # 打印总测试时间
    print(f"Test completed for {len(image_files)} images in {end_time - start_time:.2f} seconds.")

# 示例调用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

folder_path = r'E:\qy-chaoshe\dataset\kaoshi2'  # 替换为你的测试集文件夹路径
weights_base_path = "weights4/best_model_fold1.pth"  # 替换为你的基础权重文件路径

predict_image_folder(folder_path, weights_base_path)
