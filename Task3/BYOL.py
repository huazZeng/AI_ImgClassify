import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.models import resnet50



class BYOL(nn.Module):
    def __init__(self, dim=256, hidden_dim=4096, output_dim=256, beta=0.99, tau_base=0.996):
        super(BYOL, self).__init__()
        self.encoder = resnet50(pretrained=False)
        #在线网络
        self.online_network = nn.Sequential(
            self.encoder.conv1,
            self.encoder.bn1,
            self.encoder.relu,
            self.encoder.maxpool,
            self.encoder.layer1,
            self.encoder.layer2,
            self.encoder.layer3,
            self.encoder.layer4,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            MLP(2048, hidden_dim, dim)
        )
        #目标网络
        self.target_network = nn.Sequential(
            self.encoder.conv1,
            self.encoder.bn1,
            self.encoder.relu,
            self.encoder.maxpool,
            self.encoder.layer1,
            self.encoder.layer2,
            self.encoder.layer3,
            self.encoder.layer4,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            MLP(2048, hidden_dim, dim)
        )
        for param in self.target_network.parameters():
            param.requires_grad = False
        #预测器
        self.predictor = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.beta = beta
        self.tau_base = tau_base
    ## 交叉计算
    def forward(self, x1, x2):
        z1_online = self.online_network(x1)
        z2_online = self.online_network(x2)
        with torch.no_grad():
            z1_target = self.target_network(x1)
            z2_target = self.target_network(x2)
        p1 = self.predictor(z1_online)
        p2 = self.predictor(z2_online)
        loss = self.byol_loss(p1, z2_target.detach()) + self.byol_loss(p2, z1_target.detach())
        return loss.mean
    #动量更新目标网络
    def update_target_network(self, current_step, max_steps):
        #动量
        tau = 1 - (1 - self.tau_base) * (torch.cos(torch.tensor([current_step / max_steps * 3.1416])) + 1) / 2
        #动量更新 target网络
        for online_params, target_params in zip(self.online_network.parameters(), self.target_network.parameters()):
            target_params.data = tau * target_params.data + (1 - tau) * online_params.data

    #论文中的表述 
    def byol_loss(online_proj, target_proj):
        
        online_proj = F.normalize(x, dim=-1, p=2)
        target_proj = F.normalize(y, dim=-1, p=2)
        return 2 - 2 * (online_proj * target_proj).sum(dim=-1)
    
    
    
#对图像进行不同的增强，增强形式可以自己添加，以下仅为示例
def byol_augment(img):
    # First augmentation
    img1 = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)(img)
    img1 = transforms.functional.to_grayscale(img1, num_output_channels=3)
    # Second augmentation
    img2 = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)(img)
    img2 = transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))(img2)
    
    return img1, img2



# Example usage
num_epochs = 1000
learning_rate = 0.2 * batch_size / 256
weight_decay = 1.5e-6
warmup_epochs = 10
batch_size = 4096
tau_base = 0.996
num_cores = 512

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
#CIFAR10数据集
train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_cores)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BYOL().to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,  weight_decay=weight_decay)

for epoch in range(num_epochs):
    for imgs, _ in train_loader:
        imgs_augmented = [byol_augment(img) for img in imgs]
        imgs1, imgs2 = zip(*imgs_augmented)
        
        # 叠加为二维向量
        imgs1 = torch.stack(imgs1).to(device)
        imgs2 = torch.stack(imgs2).to(device)
        
        loss = model.forward(img1, img2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.update_target_network(epoch,num_epochs)
