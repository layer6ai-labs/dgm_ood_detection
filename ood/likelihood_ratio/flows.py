"""
This is the baseline obtained from "Input complexity and out-of-distribution detection with 
likelihood-based generative models" by Serra et al. https://arxiv.org/abs/1909.11480

They perform a straightforward method for OOD detection where they consider the compression
size needed as a proxy for the negative entropy term to correct the likelihood.

They name this value L and the score "S" is then equal to "- likelihood - S".

"""


from ood.base import OODBaseMethod
import torch
import typing as th
import numpy as np
from ood.wandb_visualization import visualize_scatterplots
from tqdm import tqdm
import math
from math import inf
from model_zoo.utils import load_model_with_checkpoints
from lid.utils import get_device_from_loader

import os
from dotenv import load_dotenv

class LikelihoodRatio(OODBaseMethod):
    """
    This OOD detection method visualizes trends of the latent statistics that are being calculated in the ood.methods.linear_approximations.latent_statistics.
    
    You specify a latent_statistics_calculator_class and a latetn_statistics_calculator_args and it automatically instantiates a latent statistics calculator.
    
    """
    def __init__(
        self,
        likelihood_model: torch.nn.Module,
        
        x_loader: th.Optional[torch.utils.data.DataLoader] = None,
        in_distr_loader: th.Optional[torch.utils.data.DataLoader] = None,
        
        # for logging args
        verbose: int = 0,
        
        # 
        reference_model_config: th.Optional[th.Dict] = None,
    ):
        super().__init__(
            likelihood_model = likelihood_model,
            x_loader=x_loader,
            in_distr_loader=in_distr_loader,
        )
        self.verbose = verbose

        # disable all the parameters in the model
        self.likelihood_model.eval()
  
        load_dotenv(override=True)
        
        if 'MODEL_DIR' in os.environ:
            model_root = os.environ['MODEL_DIR']
        else:
            model_root = './runs'
        
        self.reference_model = load_model_with_checkpoints(
            config=reference_model_config, 
            root=model_root, 
            device=get_device_from_loader(self.x_loader),
        )
        
        self.reference_model.eval()
        self.reference_model.denoising_sigma = False
        self.reference_model.dequantize = False
        self.reference_model.background_augmentation = None
        
        # get rid of all the randomness in the log_prob
        self.likelihood_model.denoising_sigma = False
        self.likelihood_model.dequantize = False
        self.likelihood_model.background_augmentation = None
        
        # iterate over all the parameters of likelihood_model and turn off their gradients
        # for faster performance
        for param in self.likelihood_model.parameters():
            param.requires_grad = False
        for param in self.reference_model.parameters():
            param.requires_grad = False
        
    
    def run(self):
            
        
        if self.verbose > 0:
            loader_decorated = tqdm(self.x_loader, desc="computing likelihoods for model and reference", total=len(self.x_loader))
        else:
            loader_decorated = self.x_loader
            
        log_likelihoods = None
        ref_log_likelihoods = None
        
        for x_batch in loader_decorated:
            
            with torch.no_grad():
                log_likelihoods_batch = self.likelihood_model.log_prob(x_batch).cpu().numpy().flatten()
                ref_log_likelihoods_batch = self.reference_model.log_prob(x_batch).cpu().numpy().flatten()
            
            log_likelihoods = log_likelihoods_batch if log_likelihoods is None else np.concatenate([log_likelihoods, log_likelihoods_batch])
            ref_log_likelihoods = ref_log_likelihoods_batch if ref_log_likelihoods is None else np.concatenate([ref_log_likelihoods, ref_log_likelihoods_batch])
        
        visualize_scatterplots(
            scores = np.stack([log_likelihoods, ref_log_likelihoods]).T,
            column_names = ['log-likelihood', 'ref-log-likelihood'],
        )
        
        