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
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
        return train_loader,test_loader
    
    
    
    def evaluate_accuracy(data_iter, net, device=None):
        if device is None and isinstance(net, torch.nn.Module):
            # 如果没指定device就使用net的device
            device = list(net.parameters())[0].device
        acc_sum, n = 0.0, 0
        with torch.no_grad():
            for X, y in data_iter:
                if isinstance(net, torch.nn.Module):
                    net.eval() # 评估模式, 这会关闭dropout
                    acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                    net.train() # 改回训练模式
                else: # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                    if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                        # 将is_training设置成False
                        acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item() 
                    else:
                        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item() 
                n += y.shape[0]
        return acc_sum / n
    
    def train(net, train_iter, test_iter, optimizer,device,num_epochs):
        net = net.to(device)
        print("training on ", device)
        loss = torch.nn.CrossEntropyLoss()
        for epoch in range(num_epochs):
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
            test_acc = train_CNN.evaluate_accuracy(test_iter, net)
            print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
                % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

    
