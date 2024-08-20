import random
from copy import deepcopy

import itertools

import torch.nn as nn

import torch

a=[1,2,3,4,5,6,7,8,9,10,11,12]
#b=['uno','dos','tres','cuatro','cinco','seis']

loss_fm=nn.TripletMarginLoss(margin=0.2,p=2)

# dict_ref={
#     0:[1,2,3,4,5,6,7,8,9,10],
#     1:[11,12,13,14,15,16,17,18,19,20],
#     2:[21,22,23,24,25,26,27,28,29,30]
# }

# dict_ref={
#     0:torch.arange(1,11),
#     1:torch.arange(11,21),
#     2:torch.arange(21,31),
# }

a=torch.Tensor([1,2])

dict_ref={
    0:[ torch.Tensor([1,2]) , torch.Tensor([3,4]) , torch.Tensor([5,6]) , torch.Tensor([7,8]) , torch.Tensor([9,10]) , torch.Tensor([11,12]), torch.Tensor([13,14]) , torch.Tensor([15,16]) , torch.Tensor([17,18]) , torch.Tensor([19,20]) ],
    1:[ torch.Tensor([21,22]) , torch.Tensor([23,24]) , torch.Tensor([25,26]) , torch.Tensor([27,28]) , torch.Tensor([29,30]) , torch.Tensor([31,32]), torch.Tensor([33,34]) , torch.Tensor([35,36]) , torch.Tensor([37,38]) , torch.Tensor([39,40]) ],
    2:[ torch.Tensor([41,42]) , torch.Tensor([43,44]) , torch.Tensor([45,46]) , torch.Tensor([47,48]) , torch.Tensor([49,50]) , torch.Tensor([51,52]), torch.Tensor([53,54]) , torch.Tensor([55,56]) , torch.Tensor([57,58]) , torch.Tensor([59,60]) ]
}



def grouping_list(lista,amount):
    it=iter(lista)
    return [list(group) for group in itertools.zip_longest(*[it] * amount,fillvalue=0)]

def mock_loss_fun(anchor,positive,negative):

    result=(anchor - positive) - (anchor-negative)

    return result

#numero -> embedding

for clase in dict_ref.keys():
    random.shuffle(dict_ref[clase]) #simulando el shuffle que hace el batch

    idx=int(len(dict_ref[clase])/2)

    #Los que van a ser ANCHOR y POSITIVE
    embeddings_1=dict_ref[clase][:idx]
    embeddings_2=dict_ref[clase][idx:]
    other_embeddings=[]

    new_dict=deepcopy(dict_ref)
    del new_dict[clase]

    for key,value in new_dict.items():
        other_embeddings.extend(value)
    
    random.shuffle(other_embeddings)

    groups=grouping_list(lista=other_embeddings,amount=idx)
    
    loss_history=[]

    for group in groups:
        anchors=embeddings_1
        positives=embeddings_2
        negatives=group

        for i in range(idx):
            loss=loss_fm(anchors[i],positives[i],negatives[i])
            loss_history.append(loss)
    
    for group in groups:
        positives=embeddings_1
        anchors=embeddings_2
        negatives=group
        for i in range(idx):
            loss=loss_fm(anchors[i],positives[i],negatives[i])
            loss_history.append(loss)
    
    print(loss_history)

    # print(clase)
    # print('ANCHOR',embeddings_1)
    # print('POSITIVE',embeddings_2)
    # print('NEGATIVES',other_embeddings)

    print('----------------------------------------')

    break
    

#LA LOGICA DE MI CABEZA : 10 LINEAS

#LA IMPLEMENTACION DE ESE CODIGO EN EL FRAMEWORK : 100 LINEAS



from pytorch_metric_learning import distances, losses, miners, reducers

# Define the loss function
distance = distances.CosineSimilarity()
reducer = reducers.ThresholdReducer(low=0)  # This can be customized
loss_func = losses.TripletMarginLoss(margin=0.2, distance=distance, reducer=reducer)

# Define the miner
mining_func = miners.TripletMarginMiner(
    margin=0.2, distance=distance, type_of_triplets="semihard"
)

def train(model, loss_func, mining_func, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model(data)
        
        # Mining triplets from embeddings
        indices_tuple = mining_func(embeddings, labels)
        
        # Calculate the loss
        loss = loss_func(embeddings, labels, indices_tuple)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 20 == 0:
            print(
                "Epoch {} Iteration {}: Loss = {}, Number of mined triplets = {}".format(
                    epoch, batch_idx, loss.item(), mining_func.num_triplets
                )
            )