from train_network_sin import train

from network import Task_type

layer_sizes=[[1,100,10],[1,10,100,10],[1,100],[1,32],[1,128]]
relu_layer_fucs=[['Relu','Relu'],['Relu','Relu','Relu'],['Relu'],['Relu'],['Relu']]
sigmoid_layer_fucs=[['sigmoid','sigmoid'],['sigmoid','sigmoid','sigmoid'],['sigmoid'],['sigmoid'],['sigmoid']]


relu_experienceloss=[]
sigmoid_experienceloss=[]
    # train_model.neural_network=trained_model.network
    # train_model.neural_network.learning_rate=0.008
for i in range(0, len(layer_sizes)):

   
    train_model=train(layer_sizes[i],1,relu_layer_fucs[i], Task_type.Fitting,0.001,True)
    train_model.train(25,1000)
    train_model.epo_plot(1)
    relu_experienceloss.append(train_model.test())

for i in range(0, len(layer_sizes)):
    
    train_model=train(layer_sizes[i],1,sigmoid_layer_fucs[i], Task_type.Fitting,0.01,True)
    train_model.train(25,1000)
    
    train_model.epo_plot(1)
    sigmoid_experienceloss.append(train_model.test())
    