from train_network_sin import train

from network import Task_type
import numpy as np
layer_sizes=[[1,32,32],[1,10,100,10],[1,100],[1,32],[1,128]]
relu_layer_fucs=[['Relu','Relu'],['Relu','Relu','Relu'],['Relu'],['Relu'],['Relu']]
sigmoid_layer_fucs=[['sigmoid','sigmoid'],['sigmoid','sigmoid','sigmoid'],['sigmoid'],['sigmoid'],['sigmoid']]


data1=[]
data2=[]
    # train_model.neural_network=trained_model.network
    # train_model.neural_network.learning_rate=0.008
    
# result_layersize = '_'.join(str(size) for size in layer_sizes[3])
# result_fuc='_'.join(sigmoid_layer_fucs[3])
# result_lrup='_'+str(False)
# path='Task1\sin\experiencedata\\'+result_fuc+result_layersize+result_fuc+'_21'+'.png'
# train_model.epo_plot(path)


#测试网络层数对网络的影响
for i in range(0,10):
    train_model=train(layer_sizes[4],1,sigmoid_layer_fucs[4], Task_type.Fitting,0.01,False,0.0001)
    train_model.train(25,3000)
    data1.append(train_model.test())
    
for i in range(0,10):
    train_model=train(layer_sizes[3],1,sigmoid_layer_fucs[3], Task_type.Fitting,0.01,False,0.0001)
    train_model.train(25,3000)
    data2.append(train_model.test())

print(np.mean(data1))
print(np.mean(data2))
# train_model=train(layer_sizes[3],1,sigmoid_layer_fucs[3], Task_type.Fitting,0.01,False)
# train_model.train(25,3000)
# result_layersize = '_'.join(str(size) for size in layer_sizes[3])
# result_fuc='_'.join(sigmoid_layer_fucs[3])
# result_lrup='_'+str(False)
# path='Task1\sin\experiencedata\\'+result_fuc+result_layersize+result_fuc+'_lr2'+'.png'
# train_model.epo_plot(path)
    