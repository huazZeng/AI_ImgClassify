import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class LocalDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []

        self._load_data()

    def _load_data(self):
        # Assuming data_dir structure: data_dir/class/image.jpg
        classes = sorted(os.listdir(self.data_dir))
        for i, class_name in enumerate(classes):
            class_dir = os.path.join(self.data_dir, class_name)
            if os.path.isdir(class_dir):
                for file_name in os.listdir(class_dir):
                    image_path = os.path.join(class_dir, file_name)
                    self.images.append(image_path)
                    self.labels.append(i)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('L')

        if self.transform:
            image = self.transform(image)

        return image, label