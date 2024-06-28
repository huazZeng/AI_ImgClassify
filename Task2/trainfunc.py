from torchvision import transforms
from torch.utils.data import DataLoader
from img_dataset import LocalDataset
import torch
from torch import nn, optim
import time

class train_CNN:

    
    def loaddata(dir1,dir2,batch_size):
        train_dataset = LocalDataset(data_dir=dir1, train=True)
        test_dataset = LocalDataset(data_dir=dir2, train=False)

        # 创建 DataLoader
        train_loader = DataLoader(train_dataset, batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
        return train_loader,test_loader
    
    
    
    def test(data_iter, net, device=None):
        if device is None and isinstance(net, torch.nn.Module):
            # 如果没指定device就使用net的device
            device = list(net.parameters())[0].device
        acc_sum, n = 0.0, 0
        loss_sum = 0.0  # 初始化损失总和
        loss_func = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            for X, y in data_iter:
                net.eval() # 评估模式, 这会关闭dropout
                X, y = X.to(device), y.to(device)
                y_hat = net(X)
                loss = loss_func(y_hat, y)
                loss_sum += loss.item() * y.shape[0]  # 损失累加
                acc_sum += (y_hat.argmax(dim=1) == y).float().sum().cpu().item()
                n += y.shape[0]  # 样本数累加
        
        return acc_sum / n, loss_sum / n  # 返回准确率和平均损失
    
    def train(net, train_iter, test_iter, optimizer,device,num_epochs):
        net = net.to(device)
        test_accs=[]
        train_accs=[]
        test_losses=[]
        train_losses=[]
        print("training on ", device)
        loss = torch.nn.CrossEntropyLoss()
        for epoch in range(num_epochs):
            net.train()
            train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
            for X, y in train_iter:
                X = X.to(device)
                y = y.to(device)
                y_hat = net(X)
                l = loss(y_hat, y)
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
                train_l_sum += l.cpu().item()
                train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
                n += y.shape[0]
                batch_count += 1
                
            test_acc,test_loss = train_CNN.test(test_iter, net)
            print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
                % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
            test_accs.append(test_acc)
            test_losses.append(test_loss)
            train_accs.append(train_acc_sum / n)
            train_losses.append(train_l_sum / batch_count)
        return test_accs,test_losses,train_accs,train_losses

    
