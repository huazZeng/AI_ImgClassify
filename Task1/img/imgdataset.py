import os
from PIL import Image
import random
import numpy as np

class imageDataset:
    def __init__(self, data_dir, transform=None, test_size=0.2, random_state=42):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.test_images= []
        self.test_labels= []
        self.train_images= []
        self.train_labels= []
        self.test_size = test_size
        self.random_state = random_state
        self._load_data()

    def _load_data(self):
        # Assuming data_dir structure: data_dir/class/image.jpg
        classes = sorted(os.listdir(self.data_dir))
        
        for i, class_name in enumerate(classes):
            class_dir = os.path.join(self.data_dir, class_name)
            if os.path.isdir(class_dir):
                for file_name in os.listdir(class_dir):
                    image_path = os.path.join(class_dir, file_name)
                    image = Image.open(image_path).convert('L')
                    image = np.array(image)
                    # Flatten to 1D vector
                    image_vector = image.flatten()
                    
                    self.images.append(image_vector)
                    onehot=np.eye(12)[i]
                    self.labels.append(onehot)
                    
        self.train_images, self.test_images, self.train_labels, self.test_labels = self.train_test_split(self.images, self.labels, self.test_size, self.random_state)
        

    def __len__(self):
        return len(self.train_images)

    

    def train_test_split(self,data, labels, test_size, random_state=None):
        if random_state:
            random.seed(random_state)
        
        # 将索引随机打乱
        indices = list(range(len(data)))
        random.shuffle(indices)
        
        # 计算测试集的大小
        num_test = int(test_size * len(data))
        
        # 分割数据和标签
        test_indices = indices[:num_test]
        train_indices = indices[num_test:]
        train_data = [data[i] for i in train_indices]
        test_data = [data[i] for i in test_indices]
        train_labels = [labels[i] for i in train_indices]
        test_labels = [labels[i] for i in test_indices]
        
        return train_data, test_data, train_labels, test_labels