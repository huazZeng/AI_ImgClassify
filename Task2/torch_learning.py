from torchvision import transforms
from torch.utils.data import DataLoader
from img_dataset import LocalDataset

train_dataset = LocalDataset(data_dir='Task2\\train', train=True)
test_dataset = LocalDataset(data_dir='Task2\\train', train=False)

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)