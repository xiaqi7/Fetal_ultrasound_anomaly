import time
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from model import resnet34 as create_model

# 定义类别名称
class_names = ['Anencephaly', 'Encephalocele_meningocele', 'Holoprosencephaly', 'normal', 'Rachischisis']

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

# 对图片进行预测
def predict_image(model, image_path):
    start_time = time.time()  # 记录开始时间

    img_tensor = load_image(image_path)

    # 获取预测结果
    model.eval()
    with torch.no_grad():
        output = model(img_tensor.unsqueeze(0))
        predicted_class = torch.argmax(output, dim=1).item()
        predicted_probabilities = torch.nn.functional.softmax(output, dim=1).squeeze().cpu().numpy()

    end_time = time.time()  # 记录结束时间

    # 打印预测结果和测试花费时间
    print(f"Predicted class: {class_names[predicted_class]}")
    print("Class probabilities:")
    for i, prob in enumerate(predicted_probabilities):
        print(f"  {class_names[i]}: {prob:.4f}")
    print(f"Test completed in {end_time - start_time:.2f} seconds.")

# 示例调用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = create_model(num_classes=5)  # 替换为你的模型定义
model.load_state_dict(torch.load("weights/best_model_fold48.pth", map_location=device))
model.to(device)
model.eval()

image_path = r'E:\qy-chaoshe\dataset\kaoshi\脊柱裂1-10\1.jpg'  # 替换为你要测试的图片路径
predict_image(model, image_path)
