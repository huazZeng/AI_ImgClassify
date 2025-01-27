from train_network_img import train

from networkimg import Task_type
import numpy as np
import json
layer_sizes=[[784,1024,512],[784,10,100,10],[784,100],[784,512],[784,1024]]
relu_layer_fucs=[['Relu','Relu'],['Relu','Relu','Relu'],['Relu'],['Relu'],['Relu']]
sigmoid_layer_fucs=[['sigmoid','sigmoid'],['sigmoid','sigmoid','sigmoid'],['sigmoid'],['sigmoid'],['sigmoid']]



data={}
    # train_model.neural_network=trained_model.network
    # train_model.neural_network.learning_rate=0.008
    
# result_layersize = '_'.join(str(size) for size in layer_sizes[3])
# result_fuc='_'.join(sigmoid_layer_fucs[3])
# result_lrup='_'+str(False)
# path='Task1\sin\experiencedata\\'+result_fuc+result_layersize+result_fuc+'_21'+'.png'
# train_model.epo_plot(path)


# train_model=train(layer_sizes[3],1,sigmoid_layer_fucs[3], 0.01,False,0.0001)
# train_model.save_para('Task1\img\data\ini_para.pkl')
# train_model.train(25,30)
# result_layersize = '_'.join(str(size) for size in layer_sizes[3])
# result_fuc='_'.join(sigmoid_layer_fucs[3])
# result_lrup='_'+str(False)
# path='Task1\sin\experiencedata\\'+result_fuc+result_layersize+result_fuc+'_l1ttrue'+'.png'
# train_model.epo_plot(path)


#测试网络层数对网络的影响
# train_model=train(layer_sizes[4],12,relu_layer_fucs[4], 0.0001,False,0)
# # train_model.save_para('Task1\img\data\ini_para.pkl')
# train_model.train(30,30)
# print(train_model.best_loss)

# model=train(layer_sizes[0],12,relu_layer_fucs[0],0.0001,False,0)


# model.train(30,30)
# print(train_model.best_loss)
###比较神经元数量影响
train_model=train(layer_sizes[4],12,relu_layer_fucs[4], 0.0001,False,0)
train_model.save_para('Task1\img\data\ini_para.pkl')
train_model.train(30,100)
train_model.epo_plot('Task1\img\experiencedata\单层1024.png')
data['1024']={
    'epochtestacc':train_model.epoch_testacc,
    'epochtestloss':train_model.epoch_testloss,
    'epochtrainloss':train_model.epoch_trainloss,
    'epochtrainacc':train_model.epoch_trainacc,
}



train_model=train(layer_sizes[3],12,relu_layer_fucs[3], 0.0001,False,0)
train_model.train(30,100)
train_model.epo_plot('Task1\img\experiencedata\单层512.png')
data['512']={
    'epochtestacc':train_model.epoch_testacc,
    'epochtestloss':train_model.epoch_testloss,
    'epochtrainloss':train_model.epoch_trainloss,
    'epochtrainacc':train_model.epoch_trainacc,
}


train_model=train(layer_sizes[4],12,relu_layer_fucs[4], 0.0005,False,0)
train_model.load_para('Task1\img\data\ini_para.pkl')
train_model.train(30,100)
train_model.epo_plot('Task1\img\experiencedata\单层1024-学习率.png')
data['1024-1']={
    'epochtestacc':train_model.epoch_testacc,
    'epochtestloss':train_model.epoch_testloss,
    'epochtrainloss':train_model.epoch_trainloss,
    'epochtrainacc':train_model.epoch_trainacc,
}
train_model=train(layer_sizes[4],12,relu_layer_fucs[4], 0.0001,True,0)
train_model.load_para('Task1\img\data\ini_para.pkl')
train_model.train(30,100)
train_model.epo_plot('Task1\img\experiencedata\单层1024-学习率2.png')
data['1024-2']={
    'epochtestacc':train_model.epoch_testacc,
    'epochtestloss':train_model.epoch_testloss,
    'epochtrainloss':train_model.epoch_trainloss,
    'epochtrainacc':train_model.epoch_trainacc,
}

# l1正则项
train_model=train(layer_sizes[4],12,relu_layer_fucs[4], 0.0001,True,0.000001)
train_model.load_para('Task1\img\data\ini_para.pkl')

train_model.train(30,100)
train_model.epo_plot('Task1\img\experiencedata\单层1024-正则项.png')
data['1024-3']={
    'epochtestacc':train_model.epoch_testacc,
    'epochtestloss':train_model.epoch_testloss,
    'epochtrainloss':train_model.epoch_trainloss,
    'epochtrainacc':train_model.epoch_trainacc,
}


train_model=train(layer_sizes[0],12,relu_layer_fucs[0], 0.0001,False,0)


train_model.train(30,100)
train_model.epo_plot('Task1\img\experiencedata\双层.png')
data['1024-512']={
    'epochtestacc':train_model.epoch_testacc,
    'epochtestloss':train_model.epoch_testloss,
    'epochtrainloss':train_model.epoch_trainloss,
    'epochtrainacc':train_model.epoch_trainacc,
}


with open("Task1\img\experiencedata\\data.json", "w") as json_file:
        json.dump(data, json_file, indent=4)
        
        

# train_model=train(layer_sizes[4],12,relu_layer_fucs[4], 0.0001,True,0)
# train_model.load_para('Task1\img\data\ini_para.pkl')
# train_model.train(30,30)
# print(train_model.best_loss)
# print(np.mean(data1))
# print(np.mean(data2))
# train_model=train(layer_sizes[3],1,sigmoid_layer_fucs[3], Task_type.Fitting,0.01,False)
# train_model.train(25,3000)
# result_layersize = '_'.join(str(size) for size in layer_sizes[3])
# result_fuc='_'.join(sigmoid_layer_fucs[3])
# result_lrup='_'+str(False)
# path='Task1\sin\experiencedata\\'+result_fuc+result_layersize+result_fuc+'_lr2'+'.png'
# train_model.epo_plot(path)
    

# result_layersize = '_'.join(str(size) for size in self.layer_size)
# result_fuc='_'.join(self.func)
# result_lrup='_'+str(self.lr_update)
# plt.savefig('Task1\img\experiencedata\\'+result_fuc+result_layersize+result_fuc+'_'+str(count)+'.png')