import numpy as np
import pandas as pd
import torch.utils.data

from sklearn.preprocessing import StandardScaler

import os

class Series_dataset(torch.utils.data.Dataset):
    
    def __init__(self, dataset_name):
        
        np_dict = np.load(dataset_name,allow_pickle=True).item()
        #self.items = []
        #for item,length in zip(np_dict["series"],np_dict["length"]):
        #    self.items.append(item[:length].astype(np.float32))
        self.items = np_dict["series"].astype(np.float32)
        self.targets = np_dict["churn"].astype(np.float32)#[inds]
        self.length = np_dict["length"]#[inds]
        self.input_size = np_dict["series"][0].shape[1]
        
    def __len__(self):        
        return self.targets.shape[0]
    
    def __getitem__(self,index):
        
        return self.items[index],self.targets[index],self.length[index]
