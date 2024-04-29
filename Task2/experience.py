from LeCNN import LeNet
from trainfunc import train_CNN
import torch
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def plot_train_test(data1, data2,save_path):
    epochs = len(data1)
    plt.plot(range(1, epochs + 1), data1, label='Test')
    plt.plot(range(1, epochs + 1), data2, label='Train')
    plt.xlabel('Epoch')
    plt.title('Train and Test')
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    
    
# net=LeNet(False,False)
# lr, num_epochs = 0.001, 20
# optimizer = torch.optim.Adam(net.parameters(), lr=lr)
# train_iter,test_iter=train_CNN.loaddata('Task2\\train','Task2\\train',30)
# test_accs,test_losses,train_accs,train_losses=train_CNN.train(net, train_iter, test_iter,optimizer, device, num_epochs)
# plot_train_test(test_accs,train_accs,'Task2\experience\\FF_acc.png')
# plot_train_test(test_losses,train_losses,'Task2\experience\\FF_train.png')

# net=LeNet(True,False)
# lr, num_epochs = 0.001, 20
# optimizer = torch.optim.Adam(net.parameters(), lr=lr)
# train_iter,test_iter=train_CNN.loaddata('Task2\\train','Task2\\train',30)
# test_accs,test_losses,train_accs,train_losses=train_CNN.train(net, train_iter, test_iter,optimizer, device, num_epochs)
# plot_train_test(test_accs,train_accs,'Task2\experience\\TF_acc.png')
# plot_train_test(test_losses,train_losses,'Task2\experience\\TF_train.png')

# net=LeNet(False,True)
# lr, num_epochs = 0.001, 20
# optimizer = torch.optim.Adam(net.parameters(), lr=lr)
# train_iter,test_iter=train_CNN.loaddata('Task2\\train','Task2\\train',30)
# test_accs,test_losses,train_accs,train_losses=train_CNN.train(net, train_iter, test_iter,optimizer, device, num_epochs)
# plot_train_test(test_accs,train_accs,'Task2\experience\\FT_acc.png')
# plot_train_test(test_losses,train_losses,'Task2\experience\\FT_train.png')


# net=LeNet(True,True)
# lr, num_epochs = 0.001, 20
# optimizer = torch.optim.Adam(net.parameters(), lr=lr)
# train_iter,test_iter=train_CNN.loaddata('Task2\\train','Task2\\train',30)
# test_accs,test_losses,train_accs,train_losses=train_CNN.train(net, train_iter, test_iter,optimizer, device, num_epochs)
# plot_train_test(test_accs,train_accs,'Task2\experience\\TT_acc.png')
# plot_train_test(test_losses,train_losses,'Task2\experience\\TT_train.png')


net=LeNet(False,False)
lr, num_epochs = 0.001, 20
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
train_iter,test_iter=train_CNN.loaddata('Task2\\train','Task2\\train',30)
test_accs,test_losses,train_accs,train_losses=train_CNN.train(net, train_iter, test_iter,optimizer, device, num_epochs)
plot_train_test(test_accs,train_accs,'Task2\experience\\52_acc.png')
plot_train_test(test_losses,train_losses,'Task2\experience\\52_train.png')
