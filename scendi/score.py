import torch
import torchvision.datasets as dset
import pandas as pd
from argparse import ArgumentParser, Namespace
from .metric.algorithm_utils import *
from os.path import join

from scendi.schur_features.ImageCLIPFeatureExtractor import ImageCLIPFeatureExtractor
from scendi.schur_features.TextCLIPFeatureExtractor import TextCLIPFeatureExtractor

import time
import logging
import sys

def get_logger(filepath='./logs/novelty.log'):
    '''
        Information Module:
            Save the program execution information to a log file and output to the terminal at the same time
    '''

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)

    parent_dir = os.path.dirname(filepath)
    if parent_dir and not os.path.exists(parent_dir):
        # recursive mkdir: creates all missing intermediate directories
        os.makedirs(parent_dir, exist_ok=True)

    file_handler = logging.FileHandler(filepath)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    return logger

class ScendiEvaluator():
    def __init__(self, logger_path : str, sigma : float, result_name: str, num_samples: int = 5000, batchsize: int = 128, rff_dim: int = 0, save_visuals_path: str = 'visuals'):
        self.logger_path = logger_path
        self.sigma = sigma
        self.num_samples = num_samples
        self.batchsize = batchsize
        self.rff_dim = rff_dim
        self.save_visuals_path = save_visuals_path

        self.current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        self.result_name = '{}_num_{}_sigma_{}'.format(result_name, num_samples, sigma)
        self.save_feats_name = '{}_num_{}'.format(result_name, num_samples)


        self.feature_extractor = None
        self.schur_image_feature_extractor = None
        self.schur_text_feature_extractor = None
        self.name_feature_extractor = None
        self.running_logger = None

        self.init_running_logger()
        self.running_logger.info("Scendi Evaluator Initialized.")
        
    
    def init_running_logger(self):
        self.running_logger = get_logger(join(self.logger_path, 'run_{}_{}.log'.format(self.result_name, self.current_time)))
    
    def set_schur_feature_extractor(self, name: str, save_path=None):
        self.save_path = save_path
        if name.lower() == 'clip':
            self.schur_image_feature_extractor = ImageCLIPFeatureExtractor(save_path, logger=self.running_logger)
            self.schur_text_feature_extractor = TextCLIPFeatureExtractor(save_path, logger=self.running_logger)
        else:
            raise NotImplementedError(
                f"Cannot get feature extractor '{name}'. Expected one of ['clip']"
            )
        self.name_feature_extractor = name.lower()
        self.running_logger.info("Initialized feature-extractor network: {}".format(self.name_feature_extractor))
        
        try:
            self.A_star = torch.load(os.path.join(self.save_path, self.name_feature_extractor, f'a_star/{self.result_name}.pt'))
        except:
            self.A_star = None
        
    def scendi_clustering_of_dataset(self,
                                          prompts,
                                          image_dataset: torch.utils.data.Dataset,
                                          recompute=False,
                                          paired_test_feats = None):
        
        args = Namespace(num_samples=self.num_samples, 
                         batchsize=self.batchsize, 
                         sigma=self.sigma, 
                         rff_dim=self.rff_dim,
                         logger=self.running_logger,
                         backbone=self.name_feature_extractor,
                         visual_name=self.result_name,
                         current_time=self.current_time,
                         path_save_visual=f'./{self.save_visuals_path}/modes_schur_rff',
                         feat_save_path = self.save_path,
                         num_visual_mode=5,
                         num_img_per_mode=50,
                         resize_img_to=224,
                         kernel='gaussian',
                         device = 'cuda:0'
        )
        
        self.running_logger.info("Running RFF approximation with dim: {}x2".format(args.rff_dim))
        self.running_logger.info("Num_samples_per_distribution: {}, Sigma: {}".format(args.num_samples, args.sigma))
        self.running_logger.info('test dataset length: {}'.format(len(image_dataset)))
        
        if self.schur_text_feature_extractor is None or self.schur_image_feature_extractor is None:
            self.running_logger.info("Feature extractor is not specified, use default CLIP.")
            self.set_schur_feature_extractor(name='clip', logger=self.running_logger)
        
        if paired_test_feats is None:
            with torch.no_grad():
                self.running_logger.info("Calculating image test feats:")
                image_test_feats, image_test_idxs = self.schur_image_feature_extractor.get_features_and_idxes(image_dataset, 
                                                                        name = 'test_' + self.save_feats_name, 
                                                                        recompute=recompute, 
                                                                        num_samples=args.num_samples, 
                                                                        batchsize=args.batchsize)
                self.running_logger.info("Calculating text test feats:")
                text_test_feats, text_test_idxs = self.schur_text_feature_extractor.get_features_and_idxes(prompts, 
                                                                        name = 'test_' + self.save_feats_name, 
                                                                        recompute=recompute, 
                                                                        num_samples=args.num_samples, 
                                                                        batchsize=args.batchsize)
        visualise_schur_image_modes_rff(image_test_feats, image_dataset, image_test_idxs, text_test_feats, prompts, args)
        

    def scendi_score(self,
                        prompts,
                        test_dataset: torch.utils.data.Dataset,
                        paired_test_feats = None,
                        recompute=False,
                        kernel='gaussian'):
        
        args = Namespace(num_samples=self.num_samples, 
                         batchsize=self.batchsize, 
                         sigma=self.sigma, 
                         rff_dim=self.rff_dim,
                         logger=self.running_logger,
                         backbone=self.name_feature_extractor,
                         visual_name=self.result_name,
                         current_time=self.current_time,
                         path_save_visual=f'./{self.save_visuals_path}/modes_schur',
                         feat_save_path = self.save_path,
                         num_visual_mode=5,
                         num_img_per_mode=50,
                         resize_img_to=224,
                         kernel = kernel,
                         device = 'cuda:0'
        )
        
        self.running_logger.info("Running RFF approximation with dim: {}x2".format(args.rff_dim))
        self.running_logger.info("Num_samples_per_distribution: {}, Sigma: {}".format(args.num_samples, args.sigma))
        self.running_logger.info('test dataset length: {}'.format(len(test_dataset)))
        
        if self.schur_text_feature_extractor is None or self.schur_image_feature_extractor is None:
            self.running_logger.info("Feature extractor is not specified, use default CLIP.")
            self.set_schur_feature_extractor(name='clip', logger=self.running_logger)
        
        if paired_test_feats is None:
            with torch.no_grad():
                self.running_logger.info("Calculating image test feats:")
                image_test_feats, image_test_idxs = self.schur_image_feature_extractor.get_features_and_idxes(test_dataset, 
                                                                        name = 'test_' + self.save_feats_name, 
                                                                        recompute=recompute, 
                                                                        num_samples=args.num_samples, 
                                                                        batchsize=args.batchsize)
                self.running_logger.info("Calculating text test feats:")
                text_test_feats, text_test_idxs = self.schur_text_feature_extractor.get_features_and_idxes(prompts, 
                                                                        name = 'test_' + self.save_feats_name, 
                                                                        recompute=recompute, 
                                                                        num_samples=args.num_samples, 
                                                                        batchsize=args.batchsize)
        score, _ = rff_sce_from_feats(text_test_feats, image_test_feats, args, K=None)
        
        return score