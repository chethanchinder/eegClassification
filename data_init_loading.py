# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 10:31:54 2022

@author: rakshith
"""

import numpy as np
#import torch
import seaborn as sns
import matplotlib.pyplot as plt


class data_init_loader(object):
    """
    Class for initial part of data loading , like converting labels in range 0-3
    Also has utility functions to return X,y examples to wrt to a specific person
    Splitting train data into train and validation
    """
    def __init__(self):
        pass
    
    def make_labels(self,y_train_valid,y_test=[]):
        """
        Takes in ylabels in the range in 769 returns ylabels in range 0-3
        """
        y_train_valid=y_train_valid-769
        if y_test.shape[0]:
            y_test=y_test-769

        return y_train_valid,y_test
    
    def train_valid_split(self,X_train_valid,y_train_valid,person_train_valid=np.asarray([]),valsplit=0.2):
        """
        Function to split the given train_valid combination into training and validation in a 20% 80% split
        Input X_train_valid, y_train_valid, person_train_valid
        Output X_train, X_valid, y_train, y_valid, person_train, person_valid
        """
        X_train_valid=np.asarray(X_train_valid)
        y_train_valid=np.asarray(y_train_valid)
        full_size=X_train_valid.shape[0]
        val_idxs=np.random.choice(int(full_size), int(valsplit*full_size),replace=False)
        tr_idxs = np.array(list(set(range(full_size)).difference(set(val_idxs))))
        X_train=X_train_valid[tr_idxs]
        y_train=y_train_valid[tr_idxs]
        X_valid=X_train_valid[val_idxs]
        y_valid=y_train_valid[val_idxs]
        if not(person_train_valid.shape[0]==0):
          person_train=person_train_valid[tr_idxs]
          person_valid=person_train_valid[val_idxs]
        else:
          person_train=np.asarray([])
          person_valid=np.asarray([])
        print(X_train.shape,y_train.shape,X_valid.shape,y_valid.shape,person_train.shape)
        return X_train,X_valid,y_train,y_valid,person_train,person_valid
    

    def make_subject_arr(self,X,y,person_array,person=0):
        """
        Given a train and label arrays just return train and label arrays based on the person
        Input X,y, person_array,person
        Output X,y corresponding to that particular person
        """
        pers_idx=np.argwhere(person_array.flatten()==person).flatten()
        #print("Person Index shape : ",pers_idx.shape)
        X_person=X[pers_idx,:,:]
        y_person=y[pers_idx]
        return X_person,y_person   
    
    def make_task_arr(self,X,y,task=0):
        """
        Given a train and label arrays just return train and label arrays based on the one of the 4 tasks
        Input X,y, task [ 0-3]
        Output X,y corresponding to that particular task
        """
        task_idx=np.argwhere(y.flatten()==task)
        X_task=X[task_idx]
        y_task=y[task_idx]
        return X_task,y_task
    
    def visualize_heatmap(self,X,y,per_arr, task=None):
        """
        
        Parameters
        ----------
        X_arr : An array of shape N, 22, 1000 (22 EEG Chhanels , 1000 time bins)
            DESCRIPTION.
        y_arr : An array of shape N,
            DESCRIPTION
        per_arr : An array os shape N,1
            DESCRIPTION.
        task : task number we want to visualize
        person : Subject number we want to visualize
            DESCRIPTION. The default is None.

        Returns
        -------
        Just visualizes the above as a heatmap
        None.
        """
        fig,ax=plt.subplots(figsize=(12,7))
     
        if not (task==None):
            task_idxs=np.argwhere(y==task).flatten()
            print(task_idxs.shape)
            idx_to_plot=np.random.choice(task_idxs)
            X_to_plt=X[idx_to_plot,:,:]
            title=" Heat map for a specific task number "+str(task)+" Sub no "+str(per_arr[idx_to_plot,0])
        else:
            idx_to_plot=np.random.choice(X.shape[0]).flatten()
            X_to_plt=X[idx_to_plot,:,:]
            title=" Heat map for a general task number  "+str(y[idx_to_plot])+ " Sub no " +str(per_arr[idx_to_plot,0])
        plt.title(title)
        #print("*"*20)
        print(X_to_plt.shape)    
        sns.heatmap(X_to_plt, cmap="rocket")
        plt.xlabel("Time bins")
        plt.ylabel("EEG channels")
        plt.show()
        
