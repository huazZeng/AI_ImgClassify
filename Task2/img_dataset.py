import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
class LocalDataset(Dataset):
    def __init__(self, data_dir, transform=False,train=True):
        self.data_dir = data_dir
        self.transform = transform
        self.train=train
        self.images = []
        self.labels = []
        self.transform2tensor= transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),  # 随机旋转
            transforms.ToTensor()
            ])
        self._load_data()
    
    def _load_data(self):
        # Assuming data_dir structure: data_dir/class/image.jpg
        classes = sorted(os.listdir(self.data_dir))
        for i, class_name in enumerate(classes):
            class_dir = os.path.join(self.data_dir, class_name)
            if os.path.isdir(class_dir):
                for file_name in os.listdir(class_dir):
                    if self.train and int(file_name.split('.')[0]) < 550:
                        image_path = os.path.join(class_dir, file_name)
                        self.images.append(image_path)
                        self.labels.append(i)
                    elif not self.train and int(file_name.split('.')[0]) >= 550:
                        image_path = os.path.join(class_dir, file_name)
                        self.images.append(image_path)
                        self.labels.append(i)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(image_path)
        image = self.transform2tensor(image)
        # Apply different grayscale transformations if specified
        

        return image, label