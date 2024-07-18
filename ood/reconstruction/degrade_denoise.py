
from ..base import OODBaseMethod
from ..utils import buffer_loader
from model_zoo.density_estimator.diffusions import ScoreBasedDiffusion
import torch
from ood.wandb_visualization import visualize_histogram
from typing import List, Optional, Literal
import torchvision
import wandb
from tqdm import tqdm
from scipy.stats import norm
from lpips import LPIPS
import numpy as np

class DegradeDenoise(OODBaseMethod):
    """
    Method drawn from Graham et al.
    https://arxiv.org/pdf/2211.07740.pdf
    
    In a nutshell, this method noises out a datapoint using the Gaussian kernel of a diffusion
    and then denoises it using the same diffusion. The L2 norm of the difference between the
    resulting denoised point and the original point is used for OOD detection. In addition,
    they also consider the LPIPS distance between the original and denoised points.
    
    For more detail, they employ a validation set and compute the mean distance metric and (L2 or LPIPS).
    Then, for every datapoint, they compute the z-score of that distance metric as a score for OOD detection.
    """
    
    def __init__(
        self,
        *args,
        num_time_steps: int = 100,
        steps: int = 100,
        validation_size: int = 16,
        methods_to_include: Optional[List[str]] = None,
        T: float = 1.0,
        gamma: float = 1.0,
        verbose: int = 0,
        **kwargs,
    ):
        """
        Args:
            num_time_steps: 
                the number of time steps to use for degrade and denoising process
                if for example 10 is picked, then this means that there are 10 points of time
                between 0 and T that we will use to degrade and denoise the data.
            steps:
                The number of steps that the diffusion takes between the fixed timesteps.
                For example, if `num_time_steps` is 10 and `steps` is 100, then the diffusion
                takes 1000 steps between 0 and T.
                
            validation_size: 
                the number of samples to use for validation and coming up with the mean and
                stadard deviation of the reconstruction scores.
                
            methods_to_include: the methods to include for reconstruction scores. 
                Currently, only 'l2' and 'lpips' are supported.
                
                - 'l2': the L2 norm of the difference between the original and denoised point
                - 'lpips': the LPIPS distance between the original and denoised point

            T: the final time step for the diffusion process
            
            gamma: The larger the gamma, the scheduling will become more and more exponential-like
                
            verbose: the verbosity level
        """
        super().__init__(*args, **kwargs)
        self.validation_size = validation_size
        self.all_t = T * torch.linspace(0, 1, num_time_steps + 1)[1:] ** gamma
        self.likelihood_model: ScoreBasedDiffusion
        if not isinstance(self.likelihood_model, ScoreBasedDiffusion):
            raise ValueError("The likelihood model should be a ScoreBasedDiffusion model")
        self.steps_per_time_step = steps
        if methods_to_include is None:
            methods_to_include = ['l2', 'lpips']
        self.methods_to_include = methods_to_include
        self.verbose = verbose
        
        self.perceptual_function = None
        
        if 'lpips' in self.methods_to_include:
            self.perceptual_function = LPIPS(
                pretrained=True,
                net='alex',
                version="0.1",
                lpips=True,
                spatial=False,
                pnet_rand=False,
                pnet_tune=False,
                use_dropout=True,
                model_path=None,
                eval_mode=True,
                verbose=False,
            )
        
    def _compute_reconstruction_scores(self, x, x_reconstructed): 
        # returns a K x batch_size tensor for K methods
        ret = []
        if 'l2' in self.methods_to_include:
            diffs = ((x - x_reconstructed)**2).reshape(x.shape[0], -1)  
            l2 = torch.mean(diffs, dim=1)
            ret.append(l2)
        if 'lpips' in self.methods_to_include:
            self.perceptual_function = self.perceptual_function.to(x.device)
            scores = self.perceptual_function(x, x_reconstructed).flatten()
            ret.append(scores)
        return torch.stack(ret)
    
    def _inner_visualize(self, x, all_x_reconstructed, all_x_degraded, lbl):
         
        # fit all the images in x in a 3 by 3 grid using torchvision
        x_grid = torchvision.utils.make_grid(x[:min(len(x), 9)], nrow=3)
        wandb.log({
            f"{lbl}_images/origin": wandb.Image(x_grid),
        })
        for i, x_reconstructed in enumerate(all_x_reconstructed):
            x_reconstructed_grid = torchvision.utils.make_grid(x_reconstructed[:min(len(x), 9)], nrow=3)
            x_degraded_grid = torchvision.utils.make_grid(all_x_degraded[i][:min(len(x), 9)], nrow=3)
            wandb.log({
                f"{lbl}_reconstructed_images/{i+1}": wandb.Image(x_reconstructed_grid)
            })
            wandb.log({
                f"{lbl}_degraded/{i+1}": wandb.Image(x_degraded_grid)
            })      
            
    
    def run(self):
        """
        The main method to run for OOD detection which might log important information on W&B.
        
        The way this function works is that it first subsamples a small amount of the training data
        for scale selection. When that is done, it uses the same scale to log all the LID estimates 
        and likelihood values per datapoint in self.x_loader.
        """
        with torch.no_grad():
            buffer = buffer_loader(self.in_distr_loader, self.validation_size, limit=1)
            for _ in buffer:
                validation_loader = _
                break
            
            # compute the reconstruction scores for the validation set        
            all_reconstruction_scores = []
            if self.verbose > 0:
                validation_loader_wrapper = tqdm(validation_loader, total=self.validation_size, desc="Setting the mean and std of reconstruction scores using validation set")
            else:
                validation_loader_wrapper = validation_loader
                
            buffer_idx = 0
            for x in validation_loader_wrapper:
                buffer_idx += 1
                all_x_reconstructed, all_x_degraded = self.likelihood_model.degrade_denoise(
                    x, 
                    all_t=self.all_t,
                    all_steps=self.steps_per_time_step,
                    return_degraded=True,
                    verbose=self.verbose - 1,
                )    
                if self.verbose > 2 and buffer_idx == 1:
                    self._inner_visualize(x, all_x_reconstructed, all_x_degraded, f'validation_{buffer_idx}')
                
                all_reconstruction_scores_along_timesteps = []
                for x_reconstructed in all_x_reconstructed:
                    x_reconstructed = x_reconstructed.to(x.device)
                    reconstruction_scores = self._compute_reconstruction_scores(x, x_reconstructed) # returns a [K x batch_size] tensor
                    all_reconstruction_scores_along_timesteps.append(reconstruction_scores.detach().cpu())
                
                # list of batch_size x K tensors
                all_reconstruction_scores_along_timesteps = torch.stack(all_reconstruction_scores_along_timesteps) # N x K x batch_size
                all_reconstruction_scores_along_timesteps = all_reconstruction_scores_along_timesteps.transpose(0, 2) # batch_size x K x N
                all_reconstruction_scores.append(all_reconstruction_scores_along_timesteps)
            all_reconstruction_scores = torch.cat(all_reconstruction_scores, dim=0)
        
            mean_reconstruction_scores = all_reconstruction_scores.mean(dim=0).transpose(0, 1) # N x k
            std_reconstruction_scores = all_reconstruction_scores.std(dim=0).transpose(0, 1) + 1e-6 # N x k
            
            all_z_scores = []
            if self.verbose > 0:
                x_loader_wrapper = tqdm(self.x_loader, total=len(self.x_loader), desc="Computing z-scores for OOD samples")
            else:
                x_loader_wrapper = self.x_loader
            idx = 0
            for x in x_loader_wrapper:
                idx += 1
                all_x_reconstructed, all_x_degraded = self.likelihood_model.degrade_denoise(
                    x, 
                    all_t=self.all_t,
                    all_steps=self.steps_per_time_step,
                    return_degraded=True,
                    verbose=self.verbose - 1,
                )   
                if self.verbose > 2 and idx == 1:
                    self._inner_visualize(x, all_x_reconstructed, all_x_degraded, 'ood')
                all_z_scores_along_timesteps = [] 
                for idx, x_reconstructed in enumerate(all_x_reconstructed):
                    x_reconstructed = x_reconstructed.to(x.device)
                    reconstruction_scores = self._compute_reconstruction_scores(x, x_reconstructed) # returns a [K x batch_size] tensor
                    mean_reconstruction_scores_along_timesteps = mean_reconstruction_scores[idx] # K
                    std_reconstruction_scores_along_timesteps = std_reconstruction_scores[idx] # K
                    z_score = (reconstruction_scores - mean_reconstruction_scores_along_timesteps[:, None].to(x.device)) / std_reconstruction_scores_along_timesteps[:, None].to(x.device) # K x batch_size
                    all_z_scores_along_timesteps.append(z_score.cpu())
                all_z_scores_along_timesteps = torch.stack(all_z_scores_along_timesteps) # N x K x batch_size
                # take the average z_score across the first and second dimension
                avg_z_scores = all_z_scores_along_timesteps.mean(dim=0).mean(dim=0) # batch_size
                all_z_scores.append(avg_z_scores)
            all_z_scores = torch.cat(all_z_scores, dim=0).detach().cpu().numpy()
            
            # NOTE: z_score is a strange terminology used by the paper!
            # z_score in fact is a form of standardized dissimilarity measure between the reconstructions of OOD data and the data itself.
            visualize_histogram(
                scores=-all_z_scores,
                x_label='negative z_scores',
            )