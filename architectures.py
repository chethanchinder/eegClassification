# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 12:00:34 2022

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

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

#This will go inside modules.py directory 


from torch.nn.modules.activation import ReLU
from torch.nn.modules import dropout
# Start training a vanilla LSTM on this data set


#This will go inside modules.py directory 
class LSTM(nn.Module):

  def __init__(self,input_size=22,hid_state=[64,64],num_rnns=2,dropout=0.3,num_classes=4) -> None:
      super(LSTM,self).__init__()
      # nn.LSTM is defined by pytorch already 
      # Please google for definition
      self.lstm_module=nn.LSTM(input_size=input_size,hidden_size=hid_state[0],num_layers=num_rnns,
                               batch_first=True,dropout=0)
      
      #Fully connected net
      self.fc_module=nn.Sequential(
         nn.Linear(hid_state[0],hid_state[0]),
         nn.ReLU(inplace=True) ,
         nn.BatchNorm1d(num_features=hid_state[0]),
         nn.Dropout(p=dropout),
         nn.Linear(hid_state[0],hid_state[1]),
         nn.ReLU(inplace=True) ,
         nn.BatchNorm1d(num_features=hid_state[1]),
         nn.Dropout(p=dropout),
         nn.Linear(hid_state[1],num_classes)

      )

  def forward(self,X):
    #X inputs are of form X= num_trails*num_channels*1*time_bins( as this format compatible for CNN)
    N,C,H,W=X.size()
    X=X.view(N,C,W)

    # This will give a tensor of shape num_trails*time_bins*input_size
    X=X.permute(0,2,1)
    lstm_out,_=self.lstm_module(X)

    # LSTM output of size
    #Num trials,L(time bins),H(hidden state size)
    fc_out=self.fc_module(lstm_out[:,-1,:])
    return fc_out

