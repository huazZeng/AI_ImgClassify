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
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {self.neural_network.loss}")
                
        
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
        
if __name__ == '__main__':
    train_model=train([784,1176],12,['Relu'], 0.0001)
    train_model.train(30,100)
    train_model.save_para()
    