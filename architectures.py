# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 12:00:34 2022

@author: rakshith
"""

from torch.nn.modules.activation import ReLU
from torch.nn.modules import dropout
# Start training a vanilla LSTM on this data set
#Pytorch related stuff
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision


#This will go inside modules.py directory 


class LSTM(nn.Module):

  def __init__(self,input_size=22,hid_state=32,num_rnns=2,dropout=0) -> None:
      super(LSTM,self).__init__()
      self.lstm_module=nn.LSTM(input_size=input_size,hidden_size=hid_state,num_layers=num_rnns,
                               batch_first=True,dropout=0 )
      
      #Fully
      self.fc_module=nn.Sequential(
         nn.Linear(hid_state,hid_state)
         nn.ReLU() 

      )





