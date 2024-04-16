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
                    if  int(file_name.split('.')[0]) < 550:
                        image_path = os.path.join(class_dir, file_name)
                        image_path = os.path.join(class_dir, file_name)
                        image = Image.open(image_path).convert('L')
                        image = np.array(image)
                        # Flatten to 1D vector
                        image_vector = image.flatten()
                        self.train_images.append(image_vector)
                        onehot=np.eye(12)[i]
                        self.train_labels.append(onehot)
                    elif int(file_name.split('.')[0]) >= 550:
                        image_path = os.path.join(class_dir, file_name)
                        image_path = os.path.join(class_dir, file_name)
                        image = Image.open(image_path).convert('L')
                        image = np.array(image)
                        # Flatten to 1D vector
                        image_vector = image.flatten()
                        self.test_images.append(image_vector)
                        onehot=np.eye(12)[i]
                        self.test_labels.append(onehot)


                    
      

    def __len__(self):
        return len(self.train_images)

    

    