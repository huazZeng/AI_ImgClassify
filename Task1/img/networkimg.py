from layer import layer, func
from enum import Enum
import numpy as np

class Task_type(Enum):
    Fitting = 1
    Classification = 2

class network:
    def __init__(self, layer_sizes, last_layer_size,func,learning_rate):
        
        self.layer_size=layer_sizes
        self.last_layer_size=last_layer_size
        self.func = func
        self.output = 0 
        self.last_layer_input = 0
        
        self.layers = [layer(layer_sizes[i], layer_sizes[i+1], func[i]) for i in range(len(layer_sizes) - 1)]
        self.output_layer = layer(layer_sizes[-1], last_layer_size, "Softmax")
        self.loss = 0
        self.learning_rate = learning_rate
        self.batch_size=0
            
    def forward(self, inputs):
        output = inputs
        for layer in self.layers:
            output = layer.forward(output)
        self.last_layer_input =output
        self.output=self.output_layer.forward(output)
       
        
        return self.output
        
        
    def backward(self,target):
        gradient =  (self.output-target)/len(target)
        gradient = self.output_layer.backward(gradient)
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
            
    def cal_loss(self, targets):
        
        self.loss = -np.sum(np.sum(targets * np.log(self.output), axis=1))/len(targets)

        
       
        

        
        
    def update(self):
        for layer in self.layers:
            layer.update(self.learning_rate)
        self.output_layer.update(self.learning_rate)

    def para_save(self):
        data={
           'layer_size':self.layer_size,
           'last_layer_size' : self.last_layer_size,
           'func':self.func,
           '_task_type' :self.task_type,
           'data' : []
        }
        for layer in self.layers:
            data['data'].append(layer.weights)
            data['data'].append(layer.bias)
        data['data'].append(self.output_layer.weights)
        data['data'].append(self.output_layer.bias)
        return data
            