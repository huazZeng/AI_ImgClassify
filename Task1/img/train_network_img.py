import numpy as np
from imgdataset import imageDataset
from dataloader import DataLoader
from networkimg import network, Task_type
import pickle

import matplotlib.pyplot as plt


class train:
    def __init__(self,layer_size,last_layer_size,func,learning_rate):
        self.data_model = imageDataset('Task1\\img\\train')
        self.neural_network = network(layer_size,last_layer_size,func,learning_rate)
    def train(self, batch_size, num_epochs):
        self.neural_network.batch_size=batch_size
        self.dataloader=DataLoader(self.data_model.train_images,self.data_model.train_labels,batch_size)
        
        for epoch in range(num_epochs):
            
            self.dataloader.__iter__()
            for i in range(0,int(self.data_model.__len__()/batch_size)):
                self.neural_network.loss=0
                try:
                    inputs,targets  =  self.dataloader.__next__()
                except StopIteration:
                    break
               
                self.neural_network.forward(inputs)
                self.neural_network.backward(targets)
                self.neural_network.cal_loss(targets)
                self.neural_network.update()
            print(f"Epoch {epoch}, Loss: {self.neural_network.loss}")
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {self.neural_network.loss}")
        
        
        
    def save_trainlog(self):
        with open('log.txt', 'a') as f:
            f.write(f"{self.neural_network.layer_size}")
            
        
        
        

    # def test(self):
        # inputs = self.data_model.test_data[:, 0].reshape(-1, 1)
        # targets =self.data_model.test_data[:, 1].reshape(-1, 1)
        # self.neural_network.forward(inputs)
        # self.neural_network.cal_loss(targets)
        # print(f" Loss: {self.neural_network.loss_true}")
        
        
    def save_para(self):
        data=self.neural_network.para_save()
        with open('Task1\img_model_data.pkl', 'wb') as f:
            pickle.dump(data, f)
        
if __name__ == '__main__':
    train_model=train([784,128,128,64],12,['Relu','Relu','Relu'], 0.001)
    train_model.train(30,1000)
    train_model.save_para()
    