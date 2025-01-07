import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import cv2

# 1. 加载训练好的模型
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 10)  # 10 个手势类别
model.load_state_dict(torch.load('gesture_recognition_model.pth'))
model.eval()  # 设置为评估模式

# 检查 GPU 是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 2. 定义数据预处理
transform = transforms.Compose([
    transforms.ToPILImage(),  # 将 OpenCV 图像转换为 PIL 图像
    transforms.Resize((100, 100)),  # 调整图像尺寸
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize((0.6747, 0.6457, 0.6162), (0.1462, 0.1633, 0.1913)),  # 归一化
])

# 3. 打开摄像头并实时识别
cap = cv2.VideoCapture(0)  # 打开默认摄像头
class_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']  # 类别标签

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 预处理
    input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR 转 RGB
    input_image = transform(input_image).unsqueeze(0).to(device)  # 转换为模型输入格式

    # 模型推理
    with torch.no_grad():
        output = model(input_image)
        _, predicted = torch.max(output, 1)
        gesture_class = class_labels[predicted.item()]

    # 显示结果
    cv2.putText(frame, f'Gesture: {gesture_class}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Gesture Recognition', frame)

    # 按下 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()