import numpy as np
from network import network
from sin_data_model import data_model
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
                           loaded_data['func'],loaded_data['_task_type'],0)
        i=0
        for layer in self.network.layers:
            layer.weights=loaded_data['data'][i]
            layer.bias=loaded_data['data'][i+1]
            i+=2
        self.network.output_layer.weights=loaded_data['data'][i]
        self.network.output_layer.bias=loaded_data['data'][i+1]
        
    def test_sin(self):
        # self.data_model=data_model()
        # self.data_model.get_sin_Data()
        # self.data_model.divider(0.99)
        
        # x_values = np.random.uniform(-np.pi, np.pi, 100)
        x_values = np.linspace(-np.pi, np.pi, 100000)
        # 生成对应的 sin(x) 值
        y_values = np.sin(x_values)
        
        inputs = x_values.reshape(-1, 1)
        targets =y_values.reshape(-1, 1)
        self.network.forward(inputs)
        self.network.cal_loss(targets)
        print(f" Loss: {self.network.loss_true}")
        plt.figure(figsize=(8, 6))
        plt.plot(inputs,targets, label='Actual', color='blue', linestyle='--')
        plt.plot(inputs,self.network.output , label='Predicted', color='red')
        plt.title('Comparison of Actual and Predicted Values')
        plt.xlabel('Input')
        plt.ylabel('Output')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        
if __name__ == '__main__':
    model=trained_model('Task1\\sin\\data.pkl')
    model.load_para()
    model.test_sin()