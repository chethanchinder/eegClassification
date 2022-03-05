# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 01:55:43 2022

@author: rakshith
"""
from torch.nn.modules.activation import ReLU
from torch.nn.modules import dropout
# Start training a vanilla LSTM on this data set
#Pytorch related stuff

#Pytorch related stuff
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

#This will go inside modules.py directory 


from torch.nn.modules.activation import ReLU
from torch.nn.modules import dropout

#Writing my own dataset class that inherits pytorchs dataset class 
# https://pytorch.org/vision/stable/datasets.html
class Dataset(torch.utils.data.Dataset):
  def __init__(self,X,y):
    self.X=torch.from_numpy(X).float()
    self.y=torch.from_numpy(y).long()
  
  def __len__(self):
    return len(self.X)
  
  def __getitem__(self,index):
    return self.X[index],self.y[index]
  

def torch_data_loader(X_train,y_train,X_test,y_test, X_valid=[], y_valid=[],b_size=64):
  """
  torch_data loader that takes X_train,y_train .... as inputs remember X's here are of shape Numtrails*channels(22)*timebins(1000)
  we convert X to be shape of Numtrails*channels(22)*timebins(1000)*1(this dimension extra)

  This returns 3 data loaders of train,test and validation if valid is not None
  """
  X_train=X_train[:,:,np.newaxis,:]
  X_test=X_test[:,:,np.newaxis,:]
  if len(X_valid)==0:
    pass
  else:
    X_valid=X_valid[:,:,np.newaxis,:]

  
  # The procedure is to first call dataset and then dataloader in pytorch 
  dataset_train=Dataset(X_train,y_train)
  dataloader_train=DataLoader(dataset_train,batch_size=b_size,shuffle=True)

  dataset_test=Dataset(X_test,y_test)
  dataloader_test=DataLoader(dataset_test,batch_size=b_size,shuffle=True)

  if len(X_valid) ==0:
    dataloader_valid=[]
  else:
    dataset_valid=Dataset(X_valid,y_valid)
    dataloader_valid=DataLoader(dataset_valid,batch_size=b_size,shuffle=True)
  
  return dataloader_train,dataloader_test,dataloader_valid


#This module would be in model architectures .py

#Ideas from :https://pytorch.org/tutorials/beginner/introyt/trainingyt.html


"""
for i in range(2):

Function to train per epoch it takes model which is a nn.Module() type an optimizer which torch.optim() type
https://pytorch.org/docs/stable/optim.html for more details, Loss function is the default Loss function
Trainining data loader of torch.utils.Data loader type
"""
def train_per_epoch(model,optimizer,train_loader,loss_func=nn.CrossEntropyLoss(),printevery=10):
  
  #put model in train mode
  model.train()
  running_loss=0
  last_loss=0
  # This will go through all the batches each entry in train_loader is a batch 
  for idx,data in enumerate(train_loader):
    
    inputs,labels=data

    #zero out gradients to avoid accumulation
    optimizer.zero_grad();

    # This will call .forward method of your model
    op=model(inputs)

    loss_val=loss_func(op,labels)

    # Do  backward prop
    loss_val.backward()

    #update weights
    optimizer.step()

    # loss.item gives average loss for that batch
    running_loss+=loss_val.item()

    #last_loss is the lass fof last batch
    last_loss=loss_val.item()
    # Print some info for every 10 batches
    
    if (idx%(printevery) == 0):
      print('In training#####:batches completed={}/{}'.format(idx+1,len(train_loader)), 'The value of loss is {}'.format(running_loss/(printevery)))
      running_loss=0

    return model,last_loss



#This module would be in model architectures .py
#just putting model in test mode

def test_model(model,data_loader,loss_func=nn.CrossEntropyLoss(),print_every=10,mode="train"):
  model.eval()
  # This is for each batch
  total_correct=0
  running_loss=0
  last_loss=0
  
  for idx,data in enumerate(data_loader):
    X,y=data
    output_scores=model(X)
    y_maxs,y_pred=torch.max(output_scores.data, 1)
    # Converting everything to numpy arrays 

    y_pred_np=y_pred.cpu().detach().numpy() 
    y_actual_np=y.cpu().detach().numpy()

    total_correct=total_correct+(np.where(y_pred_np==y_actual_np)[0].shape[0])

    running_loss=running_loss+loss_func(output_scores,y).item()
    last_loss=last_loss+loss_func(output_scores,y).item()
    if idx%print_every==0:
      print('batches completed={}/{}'.format(idx+1,len(data_loader)), "The value of loss "+mode+" is {}".format(running_loss/(print_every)))
      print('batches completed={}/{}'.format(idx+1,len(data_loader)), "The value of "+mode+" accuracy is {}".format(accuracy_score(y_actual_np,y_pred_np)))
      running_loss=0
    
    #Returns lastloss and overall accuracy
  return  (last_loss/idx),(total_correct/(len(data_loader.dataset)))
  

# This part of the code trains the model for multiple epochs 
# For each epoch we calculate the testing accuracy and loss so that we dont over fit and identify the correct num of epochs

def train_multi_epochs(model,optimizer,all_data_loader,loss_func=nn.CrossEntropyLoss(),printevery=10,num_epochs=5):
  # We have all data loaders all we do now is train our model for all the epochs and for reach epoch calculate the validation 
  #accuracy/precision, training accuracy and precision

  dataloader_train,dataloader_test,dataloader_valid=all_data_loader
  eval_metrics={}
  eval_metrics['train_loss_hist']=[]
  eval_metrics['train_loss_accuracy']=[]
  eval_metrics['val_loss_hist']=[]
  eval_metrics['test_loss_hist']=[]
  eval_metrics['val_loss_accuracy']=[]
  eval_metrics['test_loss_accuracy']=[]
  max_test_accu=0
  best_model=None

  for i in range(num_epochs):
    model,train_loss=train_per_epoch(model,optimizer,dataloader_train,loss_func,printevery)
    
    train_loss,train_accu=test_model(model,dataloader_train,loss_func,printevery,mode="train")
    eval_metrics['train_loss_hist'].append(train_loss)
    eval_metrics['train_loss_accuracy'].append(train_accu)
    
    if not (len(dataloader_valid)==0):
      val_loss,val_accu=test_model(model,dataloader_valid,loss_func,printevery,mode="validation")
      eval_metrics['val_loss_hist'].append(val_loss)
      eval_metrics['val_loss_accuracy'].append(val_accu)
  
    test_loss,test_accu=test_model(model,dataloader_test,loss_func,printevery,mode="test")
    eval_metrics['test_loss_hist'].append(test_loss)
    eval_metrics['test_loss_accuracy'].append(test_accu)
    
    if test_accu > max_test_accu:
      best_model= model
      max_test_accu=test_accu
    print("Epochs done==============",i+1,"/",num_epochs)
    
  return best_model,max_test_accu,eval_metrics







