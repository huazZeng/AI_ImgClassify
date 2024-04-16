import time
import torch
from torch import nn, optim
import sys
from trainfunc import train_CNN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5), # in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 12)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output




if __name__ == '__main__':
    net=LeNet()
    lr, num_epochs = 0.001, 100
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    train_iter,test_iter=train_CNN.loaddata('Task2\\train','Task2\\train',64)
    train_CNN.train(net, train_iter, test_iter,optimizer, device, num_epochs)
