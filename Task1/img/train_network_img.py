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
        self.lr_update=lr_update
        self.layer_size=layer_size
        self.func=func
        self.patience=5
        self.wait=0
        self.best_loss=1000000
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
                try:
                    inputs,targets  =  self.dataloader.__next__()
                except StopIteration:
                    break
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
            
            if self.neural_network.loss < self.best_loss:
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
            
        
    def save_para(self):
        data=self.neural_network.para_save()
        with open('Task1\img\img_model_data.pkl', 'wb') as f:
            pickle.dump(data, f)
        experience_data={
            'testacc':self.epoch_testacc,
            'testloss':self.epoch_testloss,
            'trainacc':self.epoch_trainacc,
            'trainloss':self.epoch_trainloss
            
        }
        with open('Task1\img\data\experience.pkl', 'wb') as f:
            pickle.dump(experience_data, f)
    
    def epo_plot(self,count):
        # 绘制训练和测试损失曲线
        x = range(0, len(self.epoch_trainacc), 1)
        plt.plot(x, self.epoch_trainacc, label='Training acc', color='blue')
        plt.plot(x, self.epoch_testacc, label='Test acc', color='red')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Test Loss')
        plt.legend()
        result_layersize = '_'.join(str(size) for size in self.layer_size)
        result_fuc='_'.join(self.func)
        result_lrup='_'+str(self.lr_update)
        plt.savefig('Task1\img\experiencedata\\'+result_fuc+result_layersize+result_fuc+'_'+str(count)+'.png')
        plt.close()
    
        
if __name__ == '__main__':
    train_model=train([784,1176],12,['Relu'], 0.0001,True,0)
    train_model.train(30,10)
    train_model.epo_plot(1)
    
    