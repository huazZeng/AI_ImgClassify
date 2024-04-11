import numpy as np
from enum import Enum
class func(Enum):
    Relu:1
    sigmoid:2
    No :3
class layer:
    def __init__(self, input_size, output_size,func):
        self.input_size = input_size
        self.output_size = output_size
        self.activations = None  # 存储激活值的变量，初始化为 None
        self.func=func
        
        # 初始化权重和偏置
        
        mean = 0  # 均值
        std = 0.01  # 标准差
        self.weights = mean + std * np.random.randn(input_size, output_size)
        self.bias = -np.random.rand(output_size)

        
    def forward(self, inputs):
        self.inputs=inputs
        # 计算加权和
        
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        # 应用激活函数
        if (self.func =="Relu"):
            self.activations = self.relu(weighted_sum)
        elif(self.func =="sigmoid"):
            self.activations = self.sigmoid(weighted_sum)
        elif(self.func =="No"):
            self.activations = weighted_sum
        elif(self.func =="Softmax"):
            self.activations =self.softmax(weighted_sum)
        return self.activations
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def softmax(self, x):
        
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def backward(self, gradient):
        if (self.func =="Relu"):
            relu_derivative =  np.where(self.activations>0, 1, 0)
            gradient *=relu_derivative
        elif(self.func =="sigmoid"):
            sigmoid_derivative = self.activations * (1 - self.activations) # sigmoid 函数的导数
            gradient *= sigmoid_derivative
            
        self.gradient_weights = np.dot(self.inputs.T, gradient)
        self.gradient_bias = np.sum(gradient, axis=0, keepdims=True)
        self.gradient_inputs = np.dot(gradient, self.weights.T)
        return self.gradient_inputs

    def update(self, learning_rate):
        self.weights -= learning_rate * self.gradient_weights
        self.bias -= learning_rate * self.gradient_bias.reshape(-1)
        
        