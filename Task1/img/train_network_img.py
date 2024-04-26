import numpy as np
from imgdataset import imageDataset
from dataloader import DataLoader
from networkimg import network, Task_type
import pickle

import matplotlib.pyplot as plt


class train:
    def __init__(self,layer_size,last_layer_size,func,learning_rate,lr_update,l1_lambda):
        self.data_model = imageDataset('Task1\\img\\train')
        self.neural_network = network(layer_size,last_layer_size,func,learning_rate,lr_update,l1_lambda)
        self.learning_rate=learning_rate
        self.data=self.neural_network.para_save()
        self.l1_lambda=l1_lambda
        self.lr_update=lr_update
        self.layer_size=layer_size
        self.func=func
        self.patience=10
        self.wait=0
        self.best_loss=1000000
        self.best_acc=0
        self.epoch_trainloss=[]
        self.epoch_testacc=[]
        self.epoch_testloss=[]
        self.epoch_trainacc=[]
        
        
        
        
        
    def train(self, batch_size, num_epochs):
        self.neural_network.batch_size=batch_size
       
        self.dataloader=DataLoader(self.data_model.train_images,self.data_model.train_labels,batch_size)
        self.testdataloader=DataLoader(self.data_model.test_images,self.data_model.test_labels,1000)
        for epoch in range(num_epochs):
            epochloss=0.0
            epochcount=0
            epochacc=0
            self.dataloader.__iter__()
            for i in range(0,int(self.data_model.__len__()/batch_size)):
                self.neural_network.loss=0
                
                inputs,targets  =  self.dataloader.__next__()
                
                epochcount+=1
                
                self.neural_network.forward(inputs)
                self.neural_network.backward(targets)
                self.neural_network.cal_loss(targets)
                self.neural_network.update()
                epochacc+=self.acc(self.neural_network.output,targets)

                epochloss+=self.neural_network.loss
            
            self.epoch_trainloss.append(epochloss/epochcount)
            self.epoch_trainacc.append(epochacc/epochcount)
            
            
            
            self.testdataloader.__iter__()
            testdata,testlabel=self.testdataloader.__next__()
            self.neural_network.forward(testdata)
            self.neural_network.cal_loss(testlabel)
            self.epoch_testloss.append(self.neural_network.loss)
            epochtestacc=self.acc(self.neural_network.output,testlabel)
            self.epoch_testacc.append(epochtestacc)
            
            
            
            print(f"Epoch {epoch}, Test acc: {epochtestacc}")
            print(f"Epoch {epoch}, Trainacc: {epochacc/epochcount}")
            print(f"Epoch {epoch}, Loss: {self.neural_network.loss}")
            
            if(epoch==30&self.lr_update):
                self.learning_rate/=10
            
            
            if self.neural_network.loss < self.best_loss or self.epoch_testacc[-1] > self.best_acc:
                self.data=self.neural_network.para_save()
        
                self.best_acc= self.epoch_testacc[-1]
                self.best_loss = self.neural_network.loss
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    print("Early stopping...")
                    return
        
    def save_trainlog(self):
        with open('log.txt', 'a') as f:
            f.write(f"{self.neural_network.layer_size}")
            
        
        
        

    def test(self):
        inputs = self.data_model.test_data[:, 0].reshape(-1, 1)
        targets =self.data_model.test_data[:, 1].reshape(-1, 1)
        self.neural_network.forward(inputs)
        self.neural_network.cal_loss(targets)
        print(f" Loss: {self.neural_network.loss_true}")
        
    def acc(self,output,targets):
        max_indices = np.argmax(output, axis=1)

        accuracy = np.mean(np.equal(max_indices, np.argmax(targets, axis=1)))
        return accuracy
            
        
    def save_para(self,path):
       
        with open(path, 'wb') as f:
            pickle.dump(self.data, f)
        
        
    def save_data(self):
        experience_data={
            'testacc':self.epoch_testacc,
            'testloss':self.epoch_testloss,
            'trainacc':self.epoch_trainacc,
            'trainloss':self.epoch_trainloss
            
        }
        with open('Task1\img\data\experience.pkl', 'wb') as f:
            pickle.dump(experience_data, f)
    def epo_plot(self,path):
        # 绘制训练和测试损失曲线
        x = range(0, len(self.epoch_trainacc), 1)
        plt.plot(x, self.epoch_trainacc, label='Training acc', color='blue')
        plt.plot(x, self.epoch_testacc, label='Test acc', color='red')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Test Loss')
        plt.legend()
        plt.savefig(path)
        plt.close()
        
    def load_para(self,path):
        with open(path, 'rb') as f:
            loaded_data = pickle.load(f)
        self.network=network(loaded_data['layer_size'],loaded_data['last_layer_size'],
                           loaded_data['func'],self.learning_rate,self.lr_update,self.l1_lambda)
        i=0
        for layer in self.network.layers:
            layer.weights=loaded_data['data'][i]
            layer.bias=loaded_data['data'][i+1]
            i+=2
        self.network.output_layer.weights=loaded_data['data'][i]
        self.network.output_layer.bias=loaded_data['data'][i+1]
        
    
        
if __name__ == '__main__':
    train_model=train([784,1176],12,['Relu'], 0.00001,True,0)
    train_model.train(30,200)
    train_model.save_para('Task1\img\data\img_model.pkl')
    train_model.epo_plot("Task1\img\experiencedata\\new.png")
    
    