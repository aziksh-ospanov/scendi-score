# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 14:28:54 2024

@author: admin
"""
import glob
import re
from datasets import load_dataset

class TextFilesDataset():
    
    def __init__(self, pth):
        self.pth = pth
        self.dataset = load_dataset(self.pth)
        
        self.dataset = self.dataset.filter(lambda example: re.sub('[^0-9a-zA-Z]+', '', example['text']) != '')
        self.dataset = self.dataset['train']
        print(type(self.dataset))
        
        # s = re.sub('[^0-9a-zA-Z]+', '*', s)
        
    def load_dataset(self, pth):
        dataset = load_dataset("text", data_dir=pth, sample_by="paragraph")
        
        return dataset
#%%
