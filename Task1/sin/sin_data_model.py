import numpy as np
class data_model:
    def __init__(self):
        self.data=None
        self.train_data=None
        self.test_data=None
        
    def get_sin_Data(self):
        data = np.loadtxt("Task1\sin\data\sin_data_1000.txt")
        self.data = data
    
    
    def divider_sin(self,test_rate):
        np.random.shuffle(self.data)
        self.train_data = self.data[0:int(test_rate*1250),:]
        
        self.test_data = self.data[int(test_rate*1250):,:]
        
        
    # def get_char_data(self):
    
        
        