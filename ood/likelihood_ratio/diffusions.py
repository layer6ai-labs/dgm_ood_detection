"""
The method proposed by Goodier et al. (https://arxiv.org/pdf/2310.17432.pdf)
with some modifications to fit the score-based diffusion framework.
"""

from ..base import OODBaseMethod
from model_zoo.density_estimator.diffusions import ScoreBasedDiffusion
import torch
from ood.wandb_visualization import visualize_histogram, visualize_trends

from typing import List, Optional, Literal
import wandb
from tqdm import tqdm
import numpy as np

def _internal_L_t_frac(
    likelihood_model: ScoreBasedDiffusion,
    x: torch.Tensor,
    t_frac: float,
    device: torch.device,
    verbose: int = 0,
    num_samples: int = 100,
    chunk_size: int = 128,
):
    """
    Compute the proxy $L_\\theta^{t_frac}$ that was presented in the paper by Goodier et al.
    for score-based diffusion models.
    
    We sample num_sample noised out versions of the input 'x'. The noising out is done through
    the Gaussian kernel from time 0 to time t_frac * T uniformly. 
    We then use the L1-version of the score-matching objective and take the weighted average
    of these loss values with a linear importance weigthing mechanism.
    
    Args:
        likelihood_model: the score-based diffusion model
        x: the input tensor [batch_size, ...]
        t_frac: the fraction of the total time to consider
        device: the device to use
        verbose: the verbosity level
        num_samples: the number of samples to take for computing the weighted average loss
        chunk_size: the size of the chunks to use for parallel computation
    Returns:
        the weighted average loss (batch_size,)
    """
    x_repeated = x.cpu().repeat_interleave(num_samples, dim=0)
    if verbose > 0:
        x_repeated_wrapped = tqdm(torch.split(x_repeated, chunk_size), desc="Computing losses", total=(len(x_repeated) + chunk_size - 1) // chunk_size)
    else:
        x_repeated_wrapped = torch.split(x_repeated, chunk_size)
    all_losses_frac = []
    for x_splitted in x_repeated_wrapped:
        x_splitted = x_splitted.to(device)
        losses = likelihood_model.loss(x_splitted, t_low=0.0, t_high=t_frac, weighting_scheme='custom', weighting_fn=lambda t: t + 0.5, return_aggregated=False, distance_type='l1')
        all_losses_frac.append(losses.cpu())

    all_losses_frac = torch.cat(all_losses_frac, dim=0)

    # reshape losses
    return torch.mean(all_losses_frac.reshape(-1, num_samples), dim=-1)

class DiffusionCCLR(OODBaseMethod):
    """
    The method proposed by Goodier et al. (https://arxiv.org/pdf/2310.17432.pdf)
    with some modifications to fit the score-based diffusion framework.
    
    This point takes in a time fraction `t_frac` and then computes the weighted 
    average loss of timesteps smaller than T * t_frac. Does the same for the total
    loss and then computes the ratio of the two as a score for OOD detection.
    """
    
    def __init__(
        self,
        *args,
        t_frac: float = 0.5,
        verbose: int = 0,
        num_samples: int = 100,
        chunk_size: int = 128,
        **kwargs,
    ):
        """
        Args:
            num_samples: 
                The number of samples to take for computing the weighted average loss for
                both the total and the partial loss.
                
            t_frac:
                The fraction of the total time to consider for the partial loss.
                
            chunk_size:
                The size of the chunks to use for computing the loss in parallel.
                
            verbose: the verbosity level
        """
        super().__init__(*args, **kwargs)
        self.likelihood_model: ScoreBasedDiffusion
        if not isinstance(self.likelihood_model, ScoreBasedDiffusion):
            raise ValueError("The likelihood model should be a ScoreBasedDiffusion model")
        self.verbose = verbose
        
        self.num_samples = num_samples
        self.t_frac = t_frac
        
        self.chunk_size = chunk_size
        
    @torch.no_grad()
    def run(self):
        """
        The main method to run for OOD detection which might log important information on W&B.
        
        The way this function works is that it first subsamples a small amount of the training data
        for scale selection. When that is done, it uses the same scale to log all the LID estimates 
        and likelihood values per datapoint in self.x_loader.
        """

        if self.verbose == 1:
            x_loader_wrapper = tqdm(self.x_loader, total=len(self.x_loader), desc="Computing ratios for OOD datapoints")
        else:
            x_loader_wrapper = self.x_loader

        all_scores = []
        idx = 0
        for x in x_loader_wrapper:
            idx += 1
            if self.verbose > 1:
                print(f"Computing scores for batch [{idx}/{len(self.x_loader)}]")
                
            device = x.device
            
            
            if self.verbose > 1:
                print("Computing average fraction loss")
            frac_avg_loss = _internal_L_t_frac(
                likelihood_model=self.likelihood_model,
                x=x,
                t_frac=self.t_frac,
                device=device,
                verbose=self.verbose - 1,
                num_samples=self.num_samples,
                chunk_size=self.chunk_size,
            )
            
            if self.verbose > 1:
                print("Computing average total loss")
            
            total_avg_loss = _internal_L_t_frac(
                likelihood_model=self.likelihood_model,
                x=x,
                t_frac=1.0,
                device=device,
                verbose=self.verbose - 1,
                num_samples=self.num_samples,
                chunk_size=self.chunk_size,
            )
            
            all_scores.append((frac_avg_loss - total_avg_loss).cpu())
            
        all_scores = torch.cat(all_scores, dim=0).detach().cpu().numpy()
        
        
        # compute the pdf of the z-scores
        visualize_histogram(
            scores=all_scores,
            x_label="Cclr with frac = {}".format(self.t_frac),
        )
    

class AvgLossTrend(OODBaseMethod):
    """
    The method proposed by Goodier et al. (https://arxiv.org/pdf/2310.17432.pdf)
    This code sweeps over t_frac and shows the average loss for each t_frac.
    """
    
    def __init__(
        self,
        *args,
        verbose: int = 0,
        num_samples: int = 100,
        chunk_size: int = 128,
        t_frac_count: int = 100,
        t_frac_gamma: float = 1.0,
        **kwargs,
    ):
        """
        Args:
            num_samples: 
                The number of samples to take for computing the weighted average loss for
                both the total and the partial loss.
                
            chunk_size:
                The size of the chunks to use for computing the loss in parallel.
            
            t_frac_count:
                The number of t_frac values to sweep over.
                
            verbose: the verbosity level
        """
        super().__init__(*args, **kwargs)
        self.likelihood_model: ScoreBasedDiffusion
        if not isinstance(self.likelihood_model, ScoreBasedDiffusion):
            raise ValueError("The likelihood model should be a ScoreBasedDiffusion model")
        self.verbose = verbose
        
        self.num_samples = num_samples
        self.chunk_size = chunk_size
        
        self.all_t_frac = np.linspace(0, 1 - 1e-6, t_frac_count) ** t_frac_gamma + 1e-6
        
    @torch.no_grad()
    def run(self):
        """
        The main method to run for OOD detection which might log important information on W&B.
        
        The way this function works is that it first subsamples a small amount of the training data
        for scale selection. When that is done, it uses the same scale to log all the LID estimates 
        and likelihood values per datapoint in self.x_loader.
        """
        
        x_batch = next(iter(self.x_loader))
        
        all_trends = []
        if self.verbose > 0:
            all_t_frac_wrapped = tqdm(self.all_t_frac, desc="Computing trends")
        else:
            all_t_frac_wrapped = self.all_t_frac
            
        for t_frac in all_t_frac_wrapped:
            
            avg_loss = _internal_L_t_frac(
                likelihood_model=self.likelihood_model,
                x=x_batch,
                t_frac=t_frac,
                device=x_batch.device,
                verbose=self.verbose - 1,
                num_samples=self.num_samples,
                chunk_size=self.chunk_size,
            )
            all_trends.append(avg_loss.cpu().numpy())
        
        all_trends = np.stack(all_trends, axis=0)
        
        visualize_trends(
            scores=all_trends.T,
            t_values=self.all_t_frac,
            title="CCLR trend",
            x_label="t_frac",
            y_label="avg_loss",
            with_std=True,
        )