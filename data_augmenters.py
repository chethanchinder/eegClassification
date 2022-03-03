# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 23:31:57 2022

@author: rakshith
"""

import numpy as np

class data_augmenter(object):
    """ 
    Class for data augmentation most of the methods here take in an Input of shape 
    N(no of samples)x 22(num of channels) x 1000( time bins)
    Perform some sort of data augmentation and return an output of Kx22x1000
    """
    def __init__(self):
        pass
    
    def flip_across_time(self,X,y,time_axis=2):
        """ Takes 2 numpy arrays X, y flips X across the time_axis defined default is 2
        returns back flipped X and y , here y labels dont change this effectively doubles your data set
        """
        X_flip_time=np.flip(X,axis=2)
        #print(X[0,0,:])
        #print(X_flip_time[0,0,:])
        return X_flip_time,y