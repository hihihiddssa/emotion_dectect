import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import matplotlib.pyplot as plt
import torch.nn.functional as F

# 数据预处理和增强
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 将图像转换为单通道灰度图像
    transforms.RandomRotation(15),  # 随机旋转图像,角度范围为±15度
    transforms.RandomHorizontalFlip(),  # 随机水平翻转图像,概率为0.5
    transforms.RandomResizedCrop(48, scale=(0.8, 1.0)),  # 随机裁剪并缩放到48x48大小,裁剪比例在0.8-1.0之间
    transforms.ToTensor(),  # 将PIL图像转换为Tensor,并将像素值归一化到[0,1]
    transforms.Normalize([0.5], [0.5])  # 标准化处理,减去均值0.5并除以标准差0.5:将像素值归一化到[-1,1]
])

val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# 数据集加载
def load_datasets(data_dir='FER-2013'):
    data_dir = Path(data_dir)#将数据集路径转换为Path对象，path对象是python标准库中的一个类，用于表示文件路径
    #path使用方法：
    #1. 创建Path对象：path = Path('path/to/directory')
    #2. 访问路径的各个部分：path.parts, path.parent, path.name, path.stem, path.suffix
    #3. 路径操作：path.joinpath('subdirectory')是拼接路径, path.glob('*.txt')是查找符合条件的文件, path.rglob('*.txt')是递归查找符合条件的文件
    #4. 文件操作：path.exists()是判断路径是否存在, path.is_file()是判断路径是否是文件, path.is_dir()是判断路径是否是目录, path.mkdir(parents=True, exist_ok=True)是创建目录 
    
    train_dataset = datasets.ImageFolder(#ImageFolder是torchvision.datasets中的一个类，用于加载图像数据集，加载了data_dir路径下的所有图像，并根据文件夹名称自动分类
        root=data_dir/'train',
        transform=train_transform
    )
    
    val_dataset = datasets.ImageFolder(
        root=data_dir/'val',
        transform=val_transform
    )
    
    test_dataset = datasets.ImageFolder(
        root=data_dir/'test',
        transform=val_transform
    )
    
    return train_dataset, val_dataset, test_dataset

# 定义深度残差网络
class DeepResEmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(DeepResEmotionCNN, self).__init__()
        
        # 初始卷积层
        self.initial = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        
        # 残差块配置 [(通道数, 块数量)]
        config = [(32, 2), (64, 3), (128, 4), (256, 3)]
        
        # 构建残差层
        self.res_layers = nn.ModuleList()
        in_channels = 32
        
        for idx, (channels, num_blocks) in enumerate(config):
            blocks = []
            for i in range(num_blocks):
                stride = 2 if (i == 0 and idx > 0) else 1  # 每个阶段第一个块进行下采样
                blocks.append(ResidualBlock(in_channels, channels, stride))
                in_channels = channels
            self.res_layers.append(nn.Sequential(*blocks))
        
        # 最终分类器
        self.final = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # 参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.initial(x)
        for layer in self.res_layers:
            x = layer(x)
        x = self.final(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.dropout = nn.Dropout2d(0.2)
        
    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out, inplace=True)

# 训练函数
def train_model(train_loader, val_loader, num_epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')#使用GPU训练，如果GPU不可用，则使用CPU训练
    model = DeepResEmotionCNN().to(device)#加载训练模型
    criterion = nn.CrossEntropyLoss()#定义损失函数，交叉熵损失函数
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)#定义优化器，Adam优化器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)#定义学习率调整策略，ReduceLROnPlateau策略
    
    best_val_loss = float('inf')#定义最佳验证损失，初始化为无穷大
    patience = 15#定义早停机制，15是早停次数
    no_improve = 0#定义早停机制，0是早停次数
    
    train_losses, val_losses = [], []#定义训练损失和验证损失，初始化为空列表
    train_accs, val_accs = [], []#定义训练准确率和验证准确率，初始化为空列表
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # 记录指标
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 学习率调整
        scheduler.step()
        
        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            no_improve += 1
            
        if no_improve >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
            
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}\n')
    
    return model, train_losses, val_losses, train_accs, val_accs

# 主程序
if __name__ == "__main__":
    # 加载数据
    train_dataset, val_dataset, test_dataset = load_datasets()
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)
    
    # 开始训练
    model, train_loss, val_loss, train_acc, val_acc = train_model(train_loader, val_loader)
    
    # 保存最终模型
    torch.save(model.state_dict(), 'emotion_model.pth')
