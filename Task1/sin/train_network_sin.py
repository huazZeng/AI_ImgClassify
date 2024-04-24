import numpy as np
from sin_data_model import data_model
from network import network, Task_type
import pickle


import matplotlib.pyplot as plt


class train:
    def __init__(self,layer_size,last_layer_size,func,_task_type,learning_rate,lr_update,l1_lambda):
        self.learning_rate=learning_rate
        self.lr_update=lr_update
        self.layer_size=layer_size
        self.func=func
        self.data_model = data_model()
        self.neural_network = network(layer_size,last_layer_size,func,_task_type,learning_rate)
        self.loss_train=[]
        self.loss_test=[]
        self.l1_lambda=l1_lambda
        self.best_loss=1000000000
        self.wait=0
        self.patience=5
    def train(self, batch_size, num_epochs):
        self.loss_train=[]
        self.loss_test=[]
        self.data_model.get_sin_Data()
        self.data_model.divider_sin(0.8)

        for epoch in range(num_epochs):
            np.random.shuffle(self.data_model.train_data)
            batch_data = self.data_model.train_data
            trainloss=0
            for i in range(0,int(len(self.data_model.train_data)/batch_size)):
                
                inputs = batch_data[i*batch_size:i*batch_size+batch_size, 0].reshape(-1, 1)
                targets = batch_data[i*batch_size:i*batch_size+batch_size,1].reshape(-1, 1)
                # inputs = batch_data[:, 0].reshape(-1, 1)
                # targets = batch_data[:, 1].reshape(-1, 1)
                
                self.neural_network.forward(inputs)
                self.neural_network.cal_loss(targets)
                trainloss+=self.neural_network.loss_true
                self.neural_network.backward(targets,self.l1_lambda)
                self.neural_network.update(self.learning_rate)
            if epoch % 100 == 0: 
                self.loss_test.append(self.test()) 
                print(f"Epoch {epoch}, Loss: {trainloss/int(len(self.data_model.train_data)/batch_size)}")
                self.loss_train.append(trainloss/int(len(self.data_model.train_data)/batch_size))   
                 
                if self.neural_network.loss < self.best_loss:
                    self.best_loss = self.neural_network.loss
                    self.wait = 0
                else:
                    self.wait += 1
                    if self.wait >= self.patience:
                        print("Early stopping...")
                        return
            if epoch % 1000 == 0:
                if(self.lr_update):
                    self.learning_rate/=2
                
                
                
                
        
        
    def save_trainlog(self):
        with open('log.txt', 'a') as f:
            f.write(f"{self.neural_network.layer_size}")
            
    
    def epo_plot(self,path):
        # 绘制训练和测试损失曲线
        x = range(100, len(self.loss_test) * 100 + 1, 100)
        plt.plot(x, self.loss_train, label='Training Loss', color='blue')
        plt.plot(x, self.loss_test, label='Test Loss', color='red')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Test Loss')
        plt.legend()
        
        plt.savefig(path)
        plt.close()
        

    def test(self):
        inputs = self.data_model.test_data[:, 0].reshape(-1, 1)
        targets =self.data_model.test_data[:, 1].reshape(-1, 1)
        self.neural_network.forward(inputs)
        self.neural_network.cal_loss(targets)
        return self.neural_network.loss_true
        
    def save_para(self):
        data=self.neural_network.para_save()
        with open('Task1\data.pkl', 'wb') as f:
            pickle.dump(data, f)
        
# if __name__ == '__main__':
#     # trained_model=trained_model('.\Task1\data.pkl')
#     # trained_model.load_para()
#     train_model=train([1,10,100,10],1,['Relu','Relu','Relu'], Task_type.Fitting,0.001,True)
#     # train_model.neural_network=trained_model.network
#     # train_model.neural_network.learning_rate=0.008
#     train_model.train(25,2000)
#     train_model.test()
#     train_model.save_para()
#     train_model.epo_plot(1)
    
    