import torch
from torchvision import transforms
from PIL import Image
import torchvision.models as models

# 设备配置
compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 图像预处理流水线
img_transforms = transforms.Compose([
    transforms.Resize((32, 32)),  # 调整图像尺寸
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
])

# 加载测试图像
test_img_path = "sample_image.jpg"  # 替换为实际图像路径
input_image = Image.open(test_img_path).convert('RGB') 

# 准备输入张量
input_batch = img_transforms(input_image).unsqueeze(0).to(compute_device)

# 加载预训练模型
trained_model = torch.load("saved_models/cnn_model_9.pth", 
                          map_location=compute_device)
trained_model.eval()  # 设置模型为评估模式

# 执行预测
with torch.no_grad():
    model_output = trained_model(input_batch)
    predicted_idx = torch.argmax(model_output, dim=1).item()

# CIFAR-10类别标签
class_labels = [
    '飞机', '汽车', '鸟类', '猫咪', '鹿',
    '狗狗', '青蛙', '马匹', '船只', '卡车'
]

# 输出预测结果
print(f"图像分类结果: {class_labels[predicted_idx]}")

# 可选: 输出所有类别的置信度分数
if False:  # 设置为True可查看详细预测分数
    softmax = torch.nn.Softmax(dim=1)
    probs = softmax(model_output)[0] * 100
    for i, (label, prob) in enumerate(zip(class_labels, probs)):
        print(f"{label}: {prob:.1f}%")