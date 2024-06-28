import torch
import torch.nn as nn
from LeCNN import LeNet
# 定义模型结构
from trainfunc import train_CNN
model=LeNet(True,True)

model.load_state_dict(torch.load('Task2\model.pth'))
train_iter,test_iter=train_CNN.loaddata('test_data\\test_data','test_data\\test_data',2880)
print(train_CNN.test(test_iter,model))
