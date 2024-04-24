from layer import layer, func
from enum import Enum
import numpy as np

class Task_type(Enum):
    Fitting = 1
    Classification = 2

class network:
    def __init__(self, layer_sizes, last_layer_size,func, _task_type,learning_rate):
        self.task_type = _task_type
        self.layer_size=layer_sizes
        self.last_layer_size=last_layer_size
        self.func = func
        self.output = 0 
        self.last_layer_input = 0
        if _task_type == Task_type.Fitting:
            self.layers = [layer(layer_sizes[i], layer_sizes[i+1], func[i]) for i in range(len(layer_sizes) - 1)]
            self.output_layer = layer(layer_sizes[-1], last_layer_size, "No")
            self.loss = 0
            self.learning_rate = learning_rate
            
        elif _task_type == Task_type.Classification:
            self.layers = [layer(layer_sizes[i], layer_sizes[i+1], func[i]) for i in range(len(layer_sizes) - 1)]
            self.output_layer = layer(layer_sizes[-1], last_layer_size, "Softmax")
            self.loss = 0
            self.learning_rate = learning_rate
            
    def forward(self, inputs):
        output = inputs
    
        for layer in self.layers:
           
            output = layer.forward(output)
        self.last_layer_input =output
        self.output=self.output_layer.forward(output)
        
        return self.output
    
    def cal_loss(self, targets):
        if self.task_type == Task_type.Fitting :
            self.loss = np.mean((self.output - targets) ** 2) 
            self.loss_true =np.mean(np.abs(self.output - targets))
        else:
            self.loss=np.mean(np.dot(targets,-np.log(self.output).T))
            
    def backward(self, targets,l1_lambda):
        gradient = (self.output - targets) / targets.shape[0]
        gradient = self.output_layer.backward(gradient,l1_lambda)
        
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
    
    def update(self,learning_rate):
        for layer in self.layers:
            layer.update(learning_rate)
        self.output_layer.update(learning_rate)

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
            