# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 18:37:24 2024

@author: admin
"""

import torch
from scendi.schur_features.TextFeatureExtractor import TextFeatureExtractor
import clip




class TextCLIPFeatureExtractor(TextFeatureExtractor):
    def __init__(self, save_path=None, logger=None, API_KEY = 'your_api_key'):
        self.name = "clip"

        super().__init__(save_path, logger)

        self.features_size = 512
        
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        self.model, _ = clip.load("ViT-B/32", device="cuda")
        self.model.eval()
    
    
    
    def get_feature_batch(self, text_batch):
        inputs = clip.tokenize(text_batch).to(self.device)
        features = self.model.encode_text(inputs)
        
        # print(f'FEATURE SHAPE: {features.shape}')
        
        return features