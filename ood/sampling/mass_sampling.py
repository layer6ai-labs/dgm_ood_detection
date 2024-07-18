

import torch
from ood.base_method import OODBaseMethod
import typing as th
from ood.latent.utils import buffer_loader
from tqdm import tqdm
from ood.base_method import OODBaseMethod
import typing as th
from ood.visualization import visualize_histogram, visualize_trends, visualize_scatterplots
import numpy as np
from tqdm import tqdm
import wandb
import dypy as dy
import torchvision
from scipy.stats import gennorm, norm, laplace, uniform
from scipy.special import gammaln

def compute_log_volume(d, r, p):
    """Compute the log volume of an Lp ball."""
    log_volume = d * np.log(r) + d * (np.log(2) + gammaln(1 + 1/p)) - gammaln(1 + d/p)
    return log_volume

def get_lp_uniform(n, d, radius_l, radius_r, p):
    # generate n uniform random variables
    # from an lp-ball with between distances of radius_l and radius_r
    # in a d-dimensional space
    
    u = np.random.uniform(size=n)
    radii_samples = np.power(u, 1.0 / d) * (radius_r - radius_l) + radius_l
    
    if p == 'inf':
        x_s = uniform.rvs(loc=-1, scale=2, size=(n, d))
        x_s = x_s / np.linalg.norm(x_s, ord=np.inf, axis=1)[:, None]
    elif abs(p - 2) < 1e-6:
        x_s = norm.rvs(size=(n, d))
        x_s = x_s / np.linalg.norm(x_s, ord=2, axis=1)[:, None]
    elif abs(p - 1) < 1e-6:
        x_s = laplace.rvs(size=(n, d))
        x_s = x_s / np.linalg.norm(x_s, ord=1, axis=1)[:, None]
    else:
        x_s = gennorm.rvs(beta=p, size=(n, d))
        x_s = x_s / np.linalg.norm(x_s, ord=p, axis=1)[:, None]
    
    x_s = x_s * radii_samples[:, None]
    
    return x_s

class ProbabilityMassSampling(OODBaseMethod):
    def __init__(
        self,
        likelihood_model: torch.nn.Module,
        
        x_loader: th.Optional[torch.utils.data.DataLoader] = None,
        in_distr_loader: th.Optional[torch.utils.data.DataLoader] = None,
        
        
        # for logging args
        verbose: int = 0,
        
        # The range of the radii to show in the trend
        radii_range: th.Optional[th.Tuple[float, float]] = None,
        radii_n: int = 100,
        
        # visualize the trend with or without the standard deviations
        with_std: bool = False,
        sample_count: int = 100,
        p: float = 2.0,
    ):
        """
        Inherits from EllipsoidBaseMethod and adds the visualization of the trend.
        
        Args:
            radii_range (th.Optional[th.Tuple[float, float]], optional): The smallest and largest distance to consider.
            radii_n (int, optional): The number of Radii scales in the range to consider. Defaults to 100.
            verbose: Different levels of logging, higher numbers means more logging
            
        """
        super().__init__(
            likelihood_model = likelihood_model,
            x_loader=x_loader,
            in_distr_loader=in_distr_loader,
        )
        self.likelihood_model.eval()
        self.likelihood_model.dequantize = False
        self.likelihood_model.denoising_sigma = False
        
        self.verbose = verbose
        self.radii = np.linspace(*radii_range, radii_n)
        self.with_std = with_std
        self.sample_count = sample_count
        self.p = p
        
    def run(self):
        """
        This function runs the calculate statistics function over all the different
        r values on the entire ood_loader that is taken into consideration.
        """
        
        all_log_probs = None
        
        # split the loader and calculate the trend
        trends = [] # This is the final trend
        volume_trend = []
        trend_without_volume = []
        
        if self.verbose > 0:
            rng = tqdm(self.radii, desc="running radii")
        else:
            rng = self.radii
        
        idx = 0
        for r in rng:
            idx += 1
            
            L = 0
            R = 0
            for x_batch in self.x_loader:
                R += x_batch.shape[0]
                new_log_probs = None
                d = x_batch.numel() // x_batch.shape[0]
                for i in range(self.sample_count):
                    if self.verbose > 0:
                        rng.set_description(f"sampling [{i+1}/{self.sample_count}]")
                    u = get_lp_uniform(
                        n = x_batch.shape[0],
                        d = d,
                        radius_l = 0.0,
                        radius_r = r,
                        p = self.p,
                    )
                    u = torch.from_numpy(u).reshape(x_batch.shape).to(x_batch.device).float()
                    perturbed_batch = self.likelihood_model._data_transform(x_batch) + u
                    
                    with torch.no_grad():
                        log_probs_perturbed = self.likelihood_model._nflow.log_prob(perturbed_batch).cpu().numpy().flatten()
                        
                        log_probs_perturbed = np.where(
                            np.isnan(log_probs_perturbed), 
                            np.log(1e-10 * np.ones_like(log_probs_perturbed)), 
                            log_probs_perturbed
                        )
                        new_log_probs = log_probs_perturbed if new_log_probs is None else np.logaddexp(
                            new_log_probs + np.log(i),
                            log_probs_perturbed ,
                        ) - np.log(i + 1)
                
                new_log_probs += compute_log_volume(d=d, r=r, p=self.p)
                
                if idx == 1:
                    all_log_probs = new_log_probs if all_log_probs is None else np.concatenate([all_log_probs, new_log_probs])
                else:
                    all_log_probs[L:R] = new_log_probs
                L += x_batch.shape[0] 
            
            trends.append(np.copy(all_log_probs))
            vol = compute_log_volume(d=d, r=r, p=self.p)
            volume_trend.append([vol])
            trend_without_volume.append(np.copy(all_log_probs) - vol)
            
        visualize_trends(
            scores=np.stack(trends).T,
            t_values=self.radii,
            x_label="r",
            y_label="log_estimated_prob",
            title=f"Trend of the average estimated probability",
            with_std=self.with_std,
        )

        visualize_trends(
            scores=np.stack(trend_without_volume).T,
            t_values=self.radii,
            x_label="r",
            y_label="log_estimated_prob",
            title=f"Trend of the average estimated probability without volume",
            with_std=self.with_std,
        )
        
        visualize_trends(
            scores=np.stack(volume_trend).T,
            t_values=self.radii,
            x_label="r",
            y_label="volume",
            title=f"Trend of the volume",
            with_std=self.with_std,
        )