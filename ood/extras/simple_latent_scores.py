"""
These methods only work on flow models.
They get the latent representation of the data and then compute a score based on that.

Some of the methods just consider the Lp norm of the latent representation of the data.
Some other consider the probability in an Lp ball of radius r around the latent representation of the data.
Others might incorporate semantic information in the local latent space; for example, they
might consider an ellipsoid around the latent representation of the data and then calculate
the probability measure of the ellipsoid. This ellipsoid might be semantically aware with
more variations on the dimensions that are more important for the data.
"""
import typing as th
import torch
from .base_method import OODBaseMethod
import dypy as dy 
import numpy as np
import wandb
from tqdm import tqdm
from scipy.stats import norm, ncx2
from chi2comb import chi2comb_cdf, ChiSquared
import time
import math
import dypy as dy
from ood.wandb_visualization import visualize_histogram
from nflows import transforms, distributions, flows, utils


class LatentScore(OODBaseMethod):
    """
    Calculates a score only based on the latent representation of the data.
    
    For example, if we are considering an Lp norm based method and p is given and the
    score is the norm, then it just calculates the norm of the latent representation 
    of the data. According to Nalisnick et al. (2019), the norm for p=2 of the latent 
    representation should fix the pathologies 
    
    TODO: for some reason we can't reproduce for CIFAR10-vs-SVHN
    """
    def __init__(
        self,
        likelihood_model: torch.nn.Module,
        
        
        
        x: th.Optional[torch.Tensor] = None,
        x_batch: th.Optional[torch.Tensor] = None,
        x_loader: th.Optional[torch.utils.data.DataLoader] = None,
        logger: th.Optional[th.Any] = None,
        #
        score_type: th.Literal['norm', 'prob'] = 'norm',
        score_args: th.Optional[th.Dict[str, th.Any]] = None,
        # 
        progress_bar: bool = True,
        bincount: int = 5,
        
        **kwargs,
    ) -> None:
        super().__init__(x_loader=x_loader, x=x, x_batch=x_batch, likelihood_model=likelihood_model, logger=logger, **kwargs)
        
        if x is not None:
            self.x_batch = x.unsqueeze(0)
        
        self.progress_bar = progress_bar
        
        self.bincount = bincount
        
        self.score_type = score_type
        self.score_args = score_args if score_args is not None else {}
    
    def calc_score(self, z):
        if self.score_type == 'norm':
            p = self.score_args['p']
            if p == 'inf':
                return np.max(np.abs(z), axis=-1)
            else:
                return np.linalg.norm(z, ord=p, axis=-1)
        elif self.score_type == 'prob':
            radius = self.score_args['radius']
            p = self.score_args['p']
            eps_correction = self.score_args['eps_correction'] if 'eps_correction' in self.score_args else 1e-6
            if p == 'inf':
                log_score = 0
                for z_i in z:
                    r = z_i + radius
                    l = z_i - radius
                    # calculate the standard gaussian CDF of l and r
                    cdf_l = norm.cdf(l)
                    cdf_r = norm.cdf(r)
                    p_i = cdf_r - cdf_l
                    p_i = np.clip(p_i, a_min=eps_correction, a_max=1.0)
                    log_p_i = np.log(p_i)
                    log_score += log_p_i
                return log_score
            else:
                raise NotImplementedError('p != inf not implemented yet!')
        else:
            raise ValueError(f'Unknown score type {self.score_type}')
            
    def run(self):
        """
        Creates a histogram of scores, with the scores being the lp-norm of the latent representation of the data.
        """
        if not hasattr(self.likelihood_model, '_nflow'):
            raise ValueError('The likelihood model must have a _nflow attribute that returns the number of flows.')
        
            
        with torch.no_grad():
            all_scores = None
            if self.x_loader is not None:
                if self.progress_bar:
                    iterable = tqdm(self.x_loader)
                else:
                    iterable = self.x_loader
                for x_batch, _ in iterable:
                    z = self.likelihood_model._nflow.transform_to_noise(x_batch).cpu().detach().numpy()   
                    new_scores = self.calc_scores(z) 
                    all_scores = np.concatenate([all_scores, new_scores]) if all_scores is not None else new_scores
            else:
                z = self.likelihood_model._nflow.transform_to_noise(self.x_batch).cpu().detach().numpy()   
                all_scores = self.calc_score(z)

        visualize_histogram(
            all_scores,
            bincount=self.bincount,
            x_label=f'Score {self.score_type}',
            title=f'Score {self.score_type} histogram',
            y_label='Frequency',
        )
        # # create a density histogram out of all_scores
        # # and store it as a line plot in (x_axis, density)
        # hist, bin_edges = np.histogram(all_scores, bins=self.bincount, density=True)
        # density = hist / np.sum(hist)
        # centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        # # get the average distance between two consecutive centers
        # avg_dist = np.mean(np.diff(centers))
        # # add two points to the left and right of the histogram
        # # to make sure that the plot is not cut off
        # centers = np.concatenate([[centers[0] - avg_dist], centers, [centers[-1] + avg_dist]])
        # density = np.concatenate([[0], density, [0]])
        
        # data = [[x, y] for x, y in zip(centers, density)]
        # table = wandb.Table(data=data, columns = ['score', 'density'])
        # wandb.log({'score_density': wandb.plot.line(table, 'score', 'density', title='Score density')})
    
    