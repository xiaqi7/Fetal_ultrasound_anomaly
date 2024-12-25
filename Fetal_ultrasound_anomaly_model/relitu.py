import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from model import resnet34 as create_model

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        # 使用 register_full_backward_hook 替换以避免用户警告
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_image, class_idx):
        self.model.eval()
        input_image = input_image.unsqueeze(0)

        output = self.model(input_image)
        self.model.zero_grad()
        output[0, class_idx].backward()

        gradients = self.gradients.data.cpu().numpy()[0]
        activations = self.activations.data.cpu().numpy()[0]

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.sum(weights[:, np.newaxis, np.newaxis] * activations, axis=0)

        cam = np.maximum(cam, 0)
        cam -= np.min(cam)
        cam /= np.max(cam)
        return cam


def show_cam_on_image(img, mask):
    mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))  # 调整大小为原始图像尺寸
    heatmap = cv2.applyColorMap(np.uint8(255 * mask_resized), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    overlay_image = heatmap + np.float32(img)
    overlay_image = overlay_image / np.max(overlay_image)

    return np.uint8(255 * overlay_image), np.uint8(255 * mask_resized)


# 加载和预处理输入图像
def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    img_tensor = transform(image).to(device)
    return img_tensor, np.array(image) / 255.0  # 返回原始图像用于可视化


# 生成和可视化热力图
def visualize_heatmap(model, image_path, class_idx, target_layer):
    img_tensor, img_np = load_image(image_path)
    grad_cam = GradCAM(model, target_layer)
    cam_mask = grad_cam.generate_cam(img_tensor, class_idx)

    overlay_image, heatmap_image = show_cam_on_image(img_np, cam_mask)

    # 显示叠加图像
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(overlay_image)
    plt.title('Heatmap Overlay')
    plt.axis('off')

    # 显示原始热力图
    plt.subplot(1, 2, 2)
    plt.imshow(heatmap_image, cmap='jet')
    plt.title('Original Heatmap')
    plt.axis('off')

    plt.show()


# 示例调用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = create_model(num_classes=5)  # 替换为你的模型定义
model.load_state_dict(torch.load("weights/best_model_fold1.pth", map_location=device))
model.to(device)
model.eval()

# target_layer = model.layer4[2].conv2  # 指定目标层
target_layer = model.layer4[1].conv2
image_path = r'E:\qy-chaoshe\dataset\kaoshi\reli\无脑gao2.jpg'
class_idx = 0  # 修改为你想要可视化的类索引

visualize_heatmap(model, image_path, class_idx, target_layer)

