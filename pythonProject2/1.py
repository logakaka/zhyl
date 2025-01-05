import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib

matplotlib.use('TKAgg')  # 使用 'Agg' 后台用于生成图像文件，或使用 'TkAgg' 用于图形界面
import matplotlib.pyplot as plt

# 确保设备设置（GPU 或 CPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("当前使用的设备:", device)


# 定义残差块（ResidualBlock）
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# 定义ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):  # CIFAR-10 has 10 classes
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)

        # 修改这里，使用AdaptiveAvgPool2d
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # 输出为 (batch_size, 512, 1, 1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # 使用AdaptiveAvgPool2d
        x = self.avgpool(x)  # 将特征图的空间大小转为 1x1
        x = x.view(x.size(0), -1)  # 展平 (batch_size, 512)
        x = self.fc(x)
        return x


# 初始化ResNet模型
model = ResNet(ResidualBlock, [3, 4, 6, 3]).to(device)

# 打印模型结构以确认是否成功定义
print(model)

# 数据预处理与加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 使用torchvision中的CIFAR-10数据集作为示例
dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# 将数据集分割为训练集和验证集（例如，80% 训练，20% 验证）
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 设置一个更高的学习率

# 初始化损失记录列表
train_losses = []
val_losses = []
test_losses = []

# 训练模型
num_epochs = 20
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    # 训练过程
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 累加损失
        running_loss += loss.item()

    # 记录训练损失
    epoch_train_loss = running_loss / len(train_loader)
    train_losses.append(epoch_train_loss)

    # 验证过程
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(val_loader, desc="Validating", leave=False)):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()

    # 记录验证损失
    epoch_val_loss = running_val_loss / len(val_loader)
    val_losses.append(epoch_val_loss)

    print(
        f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_train_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}")

    # 测试集损失
    running_test_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(test_loader, desc="Testing", leave=False)):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_test_loss += loss.item()

    # 记录测试集损失
    epoch_test_loss = running_test_loss / len(test_loader)
    test_losses.append(epoch_test_loss)

    print(f"Epoch {epoch + 1}/{num_epochs}, Test Loss: {epoch_test_loss:.4f}")

# 绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', color='blue')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', color='red')
plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss', color='green')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training, Validation, and Test Loss')
plt.legend()
plt.grid(True)
plt.savefig('loss_curve.png')  # 保存损失图像
plt.show()  # 显示图像
