import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.models import resnet50

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class BYOL(nn.Module):
    def __init__(self, dim=256, hidden_dim=4096, output_dim=256, beta=0.99, tau_base=0.996):
        super(BYOL, self).__init__()
        self.encoder = resnet50(pretrained=False)
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
        self.predictor = MLP(dim, hidden_dim, output_dim)
        self.beta = beta
        self.tau_base = tau_base

    def forward(self, x1, x2):
        z1_online = self.online_network(x1)
        z2_online = self.online_network(x2)
        with torch.no_grad():
            z1_target = self.target_network(x1)
            z2_target = self.target_network(x2)
        p1 = self.predictor(z1_online)
        p2 = self.predictor(z2_online)
        loss = self.loss_func(p1, z2_target.detach()) + self.loss_func(p2, z1_target.detach())
        return loss

    def update_target_network(self, current_step, max_steps):
        tau = 1 - (1 - self.tau_base) * (torch.cos(torch.tensor([current_step / max_steps * 3.1416])) + 1) / 2
        for online_params, target_params in zip(self.online_network.parameters(), self.target_network.parameters()):
            target_params.data = tau * target_params.data + (1 - tau) * online_params.data

    def loss_func(self, p, z):
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        return 2 - 2 * (p * z).sum(dim=1)

# Example usage
num_epochs = 1000
learning_rate = 0.2 * batch_size / 256
momentum = 0.9
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

train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_cores)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BYOL().to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

for epoch in range(num_epochs):
    for imgs, _ in train_loader:
        img1, img2 = imgs[0].to(device), imgs[1].to(device)
        loss = model(img1, img2)
        optimizer.zero_grad()
        loss.backwar
