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

    # Data augmentation example which is subsample and add noise

    def sub_sample_maxpool(self,X,y,sub_sample,average,noise):
        
        total_X = None
        total_y = None
        
        # Trimming the data (sample,22,1000) -> (sample,22,500)
        X = X[:,:,0:500]
        print('Shape of X after trimming:',X.shape)
        
        # Maxpooling the data (sample,22,1000) -> (sample,22,500/sub_sample)
        X_max = np.max(X.reshape(X.shape[0], X.shape[1], -1, sub_sample), axis=3)
        
        
        total_X = X_max
        total_y = y
        print('Shape of X after maxpooling:',total_X.shape)
        
        # Averaging + noise 
        X_average = np.mean(X.reshape(X.shape[0], X.shape[1], -1, average),axis=3)
        X_average = X_average + np.random.normal(0.0, 0.5, X_average.shape)
        
        total_X = np.vstack((total_X, X_average))
        total_y = np.hstack((total_y, y))
        print('Shape of X after averaging+noise and concatenating:',total_X.shape)
        
        # Subsampling
        
        for i in range(sub_sample):
            
            X_subsample = X[:, :, i::sub_sample] + \
                                (np.random.normal(0.0, 0.5, X[:, :,i::sub_sample].shape) if noise else 0.0)
                
            total_X = np.vstack((total_X, X_subsample))
            total_y = np.hstack((total_y, y))
            
        
        print('Shape of X after subsampling and concatenating:',total_X.shape)
        return total_X,total_y
    # overall pipeline for  Frequency domain trimming and subsampling 
# The input is of format Num_trials* channels * time bins

    def sub_sample_maxpool_freq(self,X,y,sub_sample=2,average=2,noise=2):
        X_fft_time=np.fft.fft(X,axis=2)

        N=X_fft_time.shape[2]
        time_bins=np.arange(X_fft_time.shape[2])
        print(X_fft_time.shape)

        #Now we need to take the second half of the signal and place it in the beginning
        X_fft_freq=np.concatenate((X_fft_time[:,:,(N//2):N],X_fft_time[:,:,0:(N//2)]),axis=2)
        print(X_fft_freq.shape)

        # Once this is done I will trim based on the graphs obtained to num_trails*22*(index 250:750)
        # Trimming the data (sample,22,1000) -> (sample,22,500)
        X= X_fft_freq[:,:,250:750]
        print('Shape of X after trimming:',X.shape)
            
        # Maxpooling the data (sample,22,1000) -> (sample,22,500/sub_sample)
        X_max = np.max(X.reshape(X.shape[0], X.shape[1], -1, sub_sample), axis=3)
            
            
        total_X = X_max
        total_y = y
        print('Shape of X after maxpooling:',total_X.shape)
            
        # Averaging + noise 
        X_average = np.mean(X.reshape(X.shape[0], X.shape[1], -1, average),axis=3)
        X_average = X_average + np.random.normal(0.0, 0.5, X_average.shape)
            
        total_X = np.vstack((total_X, X_average))
        total_y = np.hstack((total_y, y))
        print('Shape of X after averaging+noise and concatenating:',total_X.shape)
            
        # # Subsampling
            
        for i in range(sub_sample):
          X_subsample = X[:, :, i::sub_sample] + \
                                      (np.random.normal(0.0, 0.5, X[:, :,i::sub_sample].shape) if noise else 0.0)
          total_X = np.vstack((total_X, X_subsample))
          total_y = np.hstack((total_y, y))
                
            
        print('Shape of X after subsampling and concatenating:',total_X.shape)
        return np.absolute(total_X),np.absolute(total_y)



      