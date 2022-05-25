#!/usr/env/bin python3
import torch
from torch.utils.data import TensorDataset, DataLoader
import h5py
import numpy as np
import sys

torch.manual_seed(777)
kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available else {}

class FeatureDataset:
    #def __init__(self, file_name):
    def __init__(self, file_list):
        self.feature = []
        self.label = []        
        for file_name in file_list:
            self.feature_h5 = h5py.File(file_name, 'r', libver='latest', swmr=True)
            fileFeature = []
            fileLabels = []
            for _,group in self.feature_h5.items():
                for dName,data in group.items():
                    if dName == 'images':
                        fileFeature.append(data)
                    elif dName == 'labels':
                        fileLabels.append(data)
            fileFeature = np.concatenate(fileFeature)
            fileLabels = np.concatenate(fileLabels)
            #print(fileFeature.shape)
            #print(fileLabels.shape)
            self.feature.append(fileFeature)
            self.label.append(fileLabels)
        self.feature = np.concatenate(self.feature)
        self.label = np.concatenate(self.label)
        #print(self.feature.shape)
        #print(self.label.shape)
    
    def get_dataset(self, split_type=None):
        ds = TensorDataset(torch.tensor(self.feature),torch.tensor(self.label))
        length = [int(len(ds)*0.7),int(len(ds)*0.2)]
        length.append((len(ds)-sum(length)))
        trnSet,valSet,tstSet = torch.utils.data.random_split(ds,length)
        #split_type = 'training'
        if split_type == 'training':
            return trnSet
        elif split_type == 'validation':
            return valSet
        else:
            return tstSet
        

    def get_dataloader(self, split_type=None):
        #split_type = 'validation'
        if split_type == 'training':
            shuffle = True
        elif split_type == 'validation':
            shuffle = False
        elif split_type == 'test':
            shuffle = False
        dataset = self.get_dataset(split_type = split_type)
        return DataLoader(dataset, batch_size = 128, shuffle = shuffle, **kwargs)
