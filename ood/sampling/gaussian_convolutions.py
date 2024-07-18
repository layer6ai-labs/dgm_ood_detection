# A hybrid sampling technique to compute the convulution 
# of your p_theta with N(0, rI)

# 1. Sampling:
#    - Sample from N(0, rI) and compute p_theta(x - z) and average out
# 2. Analytical approach (only applies to two step models):
#   - Compute p_theta(x) * N(0, rI) using the formula for the convolution of two gaussians
#   by linearly approximating the log of the density

import typing as th
import torch
import numpy as np
from ood.base_method import OODBaseMethod
from ood.methods.linear_approximations import GaussianConvolutionStatsCalculator
from tqdm import tqdm
from ood.visualization import visualize_trends

class GaussianConvTrend(OODBaseMethod):
    def __init__(
        self,
        likelihood_model: torch.nn.Module,
        
        x: th.Optional[torch.Tensor] = None,
        x_batch: th.Optional[torch.Tensor] = None,
        x_loader: th.Optional[torch.utils.data.DataLoader] = None,
        logger: th.Optional[th.Any] = None,
        in_distr_loader: th.Optional[torch.utils.data.DataLoader] = None,
        
        sample_count: int = 10,
        
        # Latent statistics calculator
        analytical_boosting: bool = False,
        boosting_ratio: float = -1,
        boosting_radius_limit: float = -1, 
        
        # Encoding and decoding model in case boosting happens
        encoding_model_class: th.Optional[str] = None, 
        encoding_model_args: th.Optional[th.Dict[str, th.Any]] = None,
        
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
        """
        Inherits from EllipsoidBaseMethod and adds the visualization of the trend.
        
        Args:
            radii_range (th.Optional[th.Tuple[float, float]], optional): The smallest and largest distance to consider.
            radii_n (int, optional): The number of Radii scales in the range to consider. Defaults to 100.
            visualization_args (th.Optional[th.Dict[str, th.Any]], optional): Additional arguments that will get passed on
                visualize_trend function. Defaults to None which is an empty dict.
            include_reference (bool, optional): If set to True, then the training data trend will also be visualized.
        """
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
        
        self.visualization_args = visualization_args or {}
        
        self.include_reference = include_reference
        
        if self.x is not None:    
            self.x_batch = self.x.unsqueeze(0)
        
        if self.x_batch is not None:
            # create a loader with that single batch
            self.x_loader = [(self.x_batch, None, None)]
        
        self.verbose = verbose
        
        self.analytical_boosting = analytical_boosting
        if self.analytical_boosting:
            self.gaussian_statistics_calculator = GaussianConvolutionStatsCalculator(
                likelihood_model=self.likelihood_model,
                encoding_model_class=encoding_model_class,
                encoding_model_args=encoding_model_args or {},
                verbose=verbose,
                acceleration=3,
            )
            
    def run(self):
        # This function first calculates the ellipsoids for each datapoint in the loader self.x_loader
        # This calculation is carried out with the ellipsoid calculator that is passed in the constructor.
        # Then, it increases the radius "r" to mintor how the CDF changes per datapoint
        # for visualization of multiple datapoints, the visualize_trend function is used.
        
        
        def get_trend(loader):
            
               
            # split the loader and calculate the trend
            
            
            if self.verbose > 1:
                radii_range = tqdm(self.radii)    
            else:
                radii_range = self.radii
            
            trend = [] 
            for r in radii_range:
                inner_trend = []
                for x in loader:
                    x = self.likelihood_model._data_transform(x)
                    
                    log_prob_history = None
                    for _ in range(self.sample_count):
                        # calculate the sampling radius
                        sampling_r = r
                        if self.analytical_boosting:
                            if self.boosting_ratio > 0:
                                sampling_r = sampling_r * self.boosting_ratio
                            if self.boosting_radius_limit > 0:
                                sampling_r = min(sampling_r, self.boosting_radius_limit)
                                
                        z = torch.randn_like(x) * sampling_r
                        
                        perturbed_x = x + z
                        
                        if not self.analytical_boosting:
                            with torch.no_grad():
                                log_probs = self.likelihood_model.log_prob(self.likelihood_model._inverse_data_transform(perturbed_x)).cpu()
                            log_prob_history = log_probs.unsqueeze(0) if log_prob_history is None else torch.cat([log_prob_history, log_probs.unsqueeze(0)], dim=0)
                        else:
                            raise NotImplementedError("analytical boosting is not implemented yet!")
                    log_prob_perturbed = torch.logsumexp(log_prob_history, dim=0) - np.log(self.sample_count)
                    inner_trend.append(log_prob_perturbed)
                trend.append(torch.cat(inner_trend, dim=0).numpy())

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
            t_values=self.radii,
            reference_scores=reference_trend,
            x_label="r",
            y_label="logP_r(x)",
            title=f"Trend of the average log p_r(x)",
            **self.visualization_args,
        )