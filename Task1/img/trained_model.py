import numpy as np
from networkimg import network
from imgdataset import imageDataset
from dataloader import DataLoader
import pickle
import matplotlib.pyplot as plt
class trained_model:
    def __init__(self,path):
        self.path=path
        self.network=None
    def load_para(self):
        with open(self.path, 'rb') as f:
            loaded_data = pickle.load(f)
        self.network=network(loaded_data['layer_size'],loaded_data['last_layer_size'],
                           loaded_data['func'],0)
        i=0
        for layer in self.network.layers:
            layer.weights=loaded_data['data'][i]
            layer.bias=loaded_data['data'][i+1]
            i+=2
        self.network.output_layer.weights=loaded_data['data'][i]
        self.network.output_layer.bias=loaded_data['data'][i+1]
        
    def test(self):
        self.data_model = imageDataset('Task1\\img\\train')
        self.testdataloader=DataLoader(self.data_model.test_images,self.data_model.test_labels,1000)
        self.testdataloader.__iter__()
        testdata,testlabel=self.testdataloader.__next__()
        self.network.forward(testdata)
        self.network.cal_loss(testlabel)
        print(self.acc(self.network.output,testlabel))
        
    def acc(self,output,targets):
        max_indices = np.argmax(output, axis=1)

        accuracy = np.mean(np.equal(max_indices, np.argmax(targets, axis=1)))
        return accuracy
    
if __name__ == '__main__':
    model=trained_model('Task1\img\img_model_data.pkl')
    model.load_para()
    model.test()