from torchvision import transforms
from torch.utils.data import DataLoader
from img_dataset import LocalDataset
# 定义数据转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # 可以添加其他的转换操作，例如标准化
])

# 创建数据集实例
data_dir = "Task2\\train"
dataset = LocalDataset(data_dir, transform=transform)

# 创建数据加载器
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
for images, labels in dataloader:
    print(f'{images[0][0]} : {labels}')
    input()