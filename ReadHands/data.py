import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,random_split
import matplotlib.pyplot as plt

#定义数据预处理
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(), #转换为张量
    transforms.Normalize((0.6747, 0.6457, 0.6162), (0.1462, 0.1633, 0.1913)),#归一化
])

#加载数据集

dataset = datasets.ImageFolder(root='.\GestureC', transform=transform)

#划分训练集和验证集
train_size = int(0.8*len(dataset))
val_size = len(dataset)-train_size
train_dataset,val_dataset = random_split(dataset,[train_size,val_size])

#创建DataLoader
batch_size = 32
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False)

torch.save(train_dataset,'train_dataset.pt')
torch.save(val_dataset,'val_dataset.pt')

# #检查数据集
# print(f'Train dataset size:{len(train_dataset)}')
# print(f'Val dataset size:{len(val_dataset)}')
# print(f'Classes:{dataset.classes}')
# print(f'Classs to index mapping:{dataset.class_to_idx}')
#
# #检查数据样本
# def imshow(img):
#     img = img / 2 + 0.5
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
# images, labels = next(iter(train_loader))
# imshow(images[0])
# print(f'Label:{labels[0]}')