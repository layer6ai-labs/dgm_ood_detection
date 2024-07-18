from model_zoo.density_estimator.diffusions import ScoreBasedDiffusion
from ood.base import OODBaseMethod
import typing as th
import torch
from ood.methods.utils import buffer_loader
import numpy as np
import os
import json
from ood.wandb_visualization import visualize_scatterplots
from tqdm import tqdm
from notebooks.roc_analysis import get_auc, get_roc_graph, get_convex_hull
import time

class DiffusionLikelihoodHPTuning(OODBaseMethod):
    """
    score-based ood detection method for diffusion model
    """
    def __init__(
        self,
        
        # The basic parameters passed to any OODBaseMethod
        likelihood_model: ScoreBasedDiffusion,    
        x_loader: th.Optional[torch.utils.data.DataLoader] = None,
        in_distr_loader: th.Optional[torch.utils.data.DataLoader] = None,
        checkpoint_dir: th.Optional[str] = None,
        
        # The log prob calculation args
        log_prob_kwargs: th.Optional[th.Dict[str, th.Any]] = None,
        # for logging args
        verbose: int = 0,
    ):
        self.likelihood_model = likelihood_model
        super().__init__(
            x_loader=x_loader, 
            likelihood_model=likelihood_model, 
            in_distr_loader=in_distr_loader, 
            checkpoint_dir=checkpoint_dir,
        )
        self.verbose = verbose
        
        self.log_prob_kwargs = log_prob_kwargs or {}

    def run(self):
        
        in_data_batch = next(iter(self.in_distr_loader))
        out_data_batch = next(iter(self.x_loader))
        
        ts = time.time()
        all_data = torch.cat([in_data_batch, out_data_batch], dim=0)
        # join and make it parallel
        all_log_probs = self.likelihood_model.log_prob(all_data, **self.log_prob_kwargs).cpu().numpy().flatten()
        in_log_probs = all_log_probs[:len(in_data_batch)]
        out_log_probs = all_log_probs[len(in_data_batch):]
        
        time_spent = (time.time() - ts)
        x, y = get_roc_graph(
            pos_x=in_log_probs, 
            neg_x=out_log_probs, 
            verbose=0,
        )
        x, y = get_convex_hull(x, y)
        auc = get_auc(x, y)

        # AUC analysis
        num_steps = self.log_prob_kwargs.get("steps", 1000)
        num_samples = self.log_prob_kwargs.get("trace_calculation_kwargs", {}).get("sample_count", 100)
        
        visualize_scatterplots(
            scores = np.stack([[num_steps], [auc]]).T,
            column_names=["num_steps", "auc"],
        )
        visualize_scatterplots(
            scores = np.stack([[num_samples], [auc]]).T,
            column_names=["num_samples", "auc"],
        )
        
        
        # time analysis
        visualize_scatterplots(
            scores = np.stack([[num_samples], [time_spent]]).T,
            column_names=["num_samples", "time"],
        ) 
        visualize_scatterplots(
            scores = np.stack([[num_steps], [time_spent]]).T,
            column_names=["num_steps", "time"],
        )
        
        # average log prob analysis
        mean_in_log_probs = np.mean(in_log_probs)
        mean_out_log_probs = np.mean(out_log_probs)
        visualize_scatterplots(
            scores = np.stack([[num_samples], [mean_in_log_probs]]).T,
            column_names=["num_samples", "mean_in_log_prob"],
        ) 
        visualize_scatterplots(
            scores = np.stack([[num_samples], [mean_out_log_probs]]).T,
            column_names=["num_samples", "mean_out_log_prob"],
        )
        visualize_scatterplots(
            scores = np.stack([[num_steps], [mean_in_log_probs]]).T,
            column_names=["num_steps", "mean_in_log_prob"],
        ) 
        visualize_scatterplots(
            scores = np.stack([[num_steps], [mean_out_log_probs]]).T,
            column_names=["num_steps", "mean_out_log_prob"],
        )
        
        # STD log prob analysis
        std_in_log_probs = np.std(in_log_probs)
        std_out_log_probs = np.std(out_log_probs)
        visualize_scatterplots(
            scores = np.stack([[num_samples], [std_in_log_probs]]).T,
            column_names=["num_samples", "std_in_log_probs"],
        ) 
        visualize_scatterplots(
            scores = np.stack([[num_samples], [std_out_log_probs]]).T,
            column_names=["num_samples", "std_out_log_probs"],
        )
        visualize_scatterplots(
            scores = np.stack([[num_steps], [std_in_log_probs]]).T,
            column_names=["num_steps", "std_in_log_probs"],
        ) 
        visualize_scatterplots(
            scores = np.stack([[num_steps], [std_out_log_probs]]).T,
            column_names=["num_steps", "std_out_log_probs"],
        )
        