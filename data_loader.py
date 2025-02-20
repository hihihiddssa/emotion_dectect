import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image

# FER2013Dataset类继承自torch.utils.data.Dataset
# 参数说明:
# root_dir: 数据集根目录，包含train/val/test三个子目录
# mode: 数据集模式，可选'train'/'val'/'test'
# transform: 数据预处理转换
class FER2013Dataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        self.root_dir = os.path.join(root_dir, mode)
        self.transform = transform
        self.classes = sorted(os.listdir(self.root_dir))#classes是列表，列表中每个元素是一个字符串，字符串是类别名称
        
        # 收集所有图像路径和标签
        self.image_paths = []
        self.labels = []
        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, class_name)#得到格式为：root_dir/class_name 比如：FER2013/train/0
            # 这一步之后，class_dir的格式为：[FER2013/train/0,FER2013/train/1,FER2013/train/2,FER2013/train/3,FER2013/train/4,FER2013/train/5,FER2013/train/6]
            for img_name in os.listdir(class_dir):
                self.image_paths.append(os.path.join(class_dir, img_name))#得到格式示例：[FER2013/train/0/0.jpg,FER2013/train/0/1.jpg,FER2013/train/0/2.jpg,FER2013/train/0/3.jpg,FER2013/train/0/4.jpg,FER2013/train/0/5.jpg,FER2013/train/0/6.jpg]
                self.labels.append(label)#得到格式示例：[0,0,0,0,0,0,0]

    # 返回数据集长度
    def __len__(self):
        return len(self.image_paths)#image_paths是列表，列表中每个元素是一个字符串，字符串是图像的路径

    # 获取指定索引的样本
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')  # 转为灰度图
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# 数据转换配置
data_transforms = transforms.Compose([
    transforms.Resize((48, 48)),      # FER2013原始尺寸为48x48
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 创建DataLoader示例
def get_dataloader(root_dir, batch_size=32):
    train_set = FER2013Dataset(root_dir, 'train', data_transforms)
    #传入参数：root_dir, mode='train', transform=None
    #train_set是FER2013Dataset类的一个实例，接下来可以调用FER2013Dataset类中的方法
    #train_set.len()返回训练集的长度;
    #train_set.getitem(0)返回训练集的第一个样本;train_set[0]返回训练集的第一个样本,image,label=train_set[0]
    val_set = FER2013Dataset(root_dir, 'val', data_transforms)
    
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader 