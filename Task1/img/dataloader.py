import random
import numpy as np
class DataLoader:
    def __init__(self, dataset,labels, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.labels =labels
        self.dataset_size = len(dataset)
        self.indices = list(range(self.dataset_size))
        
    def __iter__(self):
        random.shuffle(self.indices)  # 在每次迭代开始之前打乱数据顺序
        self.start_index = 0
        return self

    def __next__(self):
        if self.start_index >= self.dataset_size:
            raise StopIteration
        batch_indices = self.indices[self.start_index:self.start_index + self.batch_size]
        datas = [self.dataset[idx] for idx in batch_indices]
        labels = [self.labels[idx] for idx in batch_indices]
        datas = [np.atleast_2d(data) for data in datas]
        labels = [np.atleast_2d(label) for label in labels]
        
        # 堆叠二维数组以形成批数据
        
        return datas,labels



