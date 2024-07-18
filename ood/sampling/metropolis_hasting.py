"""
This piece of code tries to estimate for a given point x', 
the probability of sampling something in radius range 'r' of x'.

To do that, we have a sequence r1 < r2 < ... < rk

and for each pair r = ri < ri+1 = R, we compute the conditional
probabilty of sampling a point in the 'r' ball around x' given samples generated from the
R ball.
"""

import torch
import typing as th
import numpy as np
from tqdm import tqdm
from ood.base_method import OODBaseMethod
from ood.visualization import visualize_trends

class MetropolisHastingEstimator(OODBaseMethod):
    def __init__(
        self,
        likelihood_model: torch.nn.Module,
        
        x: th.Optional[torch.Tensor] = None,
        x_batch: th.Optional[torch.Tensor] = None,
        x_loader: th.Optional[torch.utils.data.DataLoader] = None,
        logger: th.Optional[th.Any] = None,
        in_distr_loader: th.Optional[torch.utils.data.DataLoader] = None,
        
        sample_count: int = 10, 
        proposal_std: float = 1.0,
        use_adaptive_proposal: bool = False,
        
        # for logging args
        verbose: int = 0,
        
        # The range of the radii to show in the trend
        radii_range: th.Optional[th.Tuple[float, float]] = None,
        radii_n: int = 100,
        
        # visualization arguments
        visualization_args: th.Optional[th.Dict[str, th.Any]] = None,
        
        # include reference or not
        include_reference: bool = False,
    ):
        super().__init__(
            x_loader=x_loader, 
            x=x, 
            x_batch=x_batch, 
            likelihood_model=likelihood_model, 
            logger=logger, 
            in_distr_loader=in_distr_loader, 
        )
        
        self.radii = np.linspace(*radii_range, radii_n)
        
        self.sample_count = sample_count
        self.proposal_std = proposal_std
        self.use_adaptive_proposal = use_adaptive_proposal
        
        self.visualization_args = visualization_args or {}
        
        self.include_reference = include_reference
        
        if self.x is not None:    
            self.x_batch = self.x.unsqueeze(0)
        
        if self.x_batch is not None:
            # create a loader with that single batch
            self.x_loader = [(self.x_batch, None, None)]
        
        self.verbose = verbose
        
    
    def run(self):
        # This function first calculates the ellipsoids for each datapoint in the loader self.x_loader
        # This calculation is carried out with the ellipsoid calculator that is passed in the constructor.
        # Then, it increases the radius "r" to mintor how the CDF changes per datapoint
        # for visualization of multiple datapoints, the visualize_trend function is used.
        
        
        def get_trend(loader):
            
               
            # split the loader and calculate the trend
            
            
            r = self.radii[0]
            if self.verbose > 1:
                radii_range = tqdm(self.radii[1:])    
            else:
                radii_range = self.radii[1:]
            
            trend = []
            # create a -inf vector of size 10 in numpy
            with torch.no_grad():
                cumul_log_cdfs = [np.full((x.shape[0],), -float('inf')) for x in loader]
                cumul_samples_seen = [np.zeros(x.shape[0]) for x in loader]
            
            for R in radii_range:
                
                # p(B_r(x) | B_R(x))
                
                inner_trend = []
                for x, cumul_log_cdf, cumul_sample_seen in zip(loader, cumul_log_cdfs, cumul_samples_seen):
                    
                    proposal_std = self.proposal_std * R
                    if self.use_adaptive_proposal:
                        
                        for _ in range(100):
                            new_batch = x + torch.randn_like(x) * proposal_std
                            # for each single point, accept only if it fits in a ball of radius R 
                            # from the original point
                            
                            in_range = torch.norm((new_batch - x).reshape(x.shape[0], -1), p=2, dim=-1) < r + 1e-6
                            
                            # count the number of points in range
                            if in_range.sum() < 0.5:
                                proposal_std *= 0.9
                            else:
                                break
                    
                    
                    prev_samples = x
                    with torch.no_grad():
                        prev_log_probs = self.likelihood_model.log_prob(x).cpu().numpy().flatten()
                    
                    for _ in range(self.sample_count):
                        new_samples = prev_samples + torch.randn_like(x) * proposal_std
                        in_R_range = (torch.norm((new_samples - x).reshape(x.shape[0], -1), p=2, dim=-1) < R).cpu().numpy()
                            
                        with torch.no_grad():
                            new_log_probs = self.likelihood_model.log_prob(new_samples).cpu().numpy().flatten()
                            
                        log_alpha = new_log_probs - prev_log_probs
                        u = np.log(np.random.uniform(size=x.shape[0]))
                        accept = in_R_range & (log_alpha > u)
                        
                        # check whether the new samples are inside the smaller range as well or not!
                        in_r_range = (torch.norm((new_samples - x).reshape(x.shape[0], -1), p=2, dim=-1) < r).cpu().numpy()
                        
                        radii_range.set_description("Acceptance rate: %.2f [%d/%d]" % (accept.sum() / x.shape[0], _, self.sample_count))
                        
                        
                        new_vals = np.where(in_r_range, new_log_probs, np.full((x.shape[0],), -float('inf')))
                        cumul_log_cdf = np.where(
                            accept, 
                            np.logaddexp(cumul_log_cdf + np.log(cumul_sample_seen), new_vals) - np.log(cumul_sample_seen + 1), 
                            cumul_log_cdf
                        )
                        
                        cumul_sample_seen += accept
                    
                    inner_trend.append(cumul_log_cdf)
                inner_trend = np.concatenate(inner_trend, axis=0)
                trend.append(inner_trend)
                
                r = R

            return np.stack(trend, axis=1)
        
        if self.verbose > 0:
            print("Calculating trend ...")
            
        trend = get_trend(self.x_loader)
        
        # add reference if the option is set to True
        reference_trend = None
        if self.include_reference:
            
            if self.verbose > 0:
                print("Calculating reference trend ...")
            
            reference_trend = get_trend(self.in_distr_loader)
                  
        visualize_trends(
            scores=trend,
            t_values=self.radii[1:],
            reference_scores=reference_trend,
            x_label="r",
            y_label="logP(B_r(x)_st_B_R(x))",
            title=f"Trend of the average log P(B_r(x) | B_R(x))",
            **self.visualization_args,
        )