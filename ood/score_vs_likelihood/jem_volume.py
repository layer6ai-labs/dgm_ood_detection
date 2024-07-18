"""
A simple implementation of the baseline presented by:
https://openreview.net/pdf?id=Hkxzx0NtDB
"YOUR CLASSIFIER IS SECRETLY AN ENERGY BASED
MODEL AND YOU SHOULD TREAT IT LIKE ONE" published at ICLR 2020.

This implementation claims that the norm of the derivative of the likelihood\
itself w.r.t. the input datapoint acts as a proxy for the probability mass.

We question this assumption by doing the same thing on our flow-based models.
We indicate that while this metric works for the pathological cases, it fails in
the non-pathological direction.
"""
from ood.base_method import OODBaseMethod
import torch
import typing as th
import numpy as np
from ood.visualization import visualize_histogram, visualize_scatterplots
from tqdm import tqdm

class JEMVol(OODBaseMethod):
    """
    This OOD detection method visualizes trends of the latent statistics that are being calculated in the ood.methods.linear_approximations.latent_statistics.
    
    You specify a latent_statistics_calculator_class and a latetn_statistics_calculator_args and it automatically instantiates a latent statistics calculator.
    
    """
    def __init__(
        self,
        likelihood_model: torch.nn.Module,
        
        x_loader: th.Optional[torch.utils.data.DataLoader] = None,
        in_distr_loader: th.Optional[torch.utils.data.DataLoader] = None,
        
        # the hyperparameters related to the autodiff method used for calculating the
        # derivative itself
        use_functorch: bool = True,
        use_forward_mode: bool = True,
        use_vmap: bool = True,
        chunk_size: int = 1,
        
        # The norm that is used for the derivative
        lp_norm_order: int = 2,
        
        # for logging args
        verbose: int = 0,
        
    ):
        super().__init__(
            likelihood_model = likelihood_model,
            x_loader=x_loader,
            in_distr_loader=in_distr_loader,
        )
        self.verbose = verbose

        # disable all the parameters in the model
        self.likelihood_model.eval()
        
        # get rid of all the randomness in the log_prob
        self.likelihood_model.denoising_sigma = False
        self.likelihood_model.dequantize = False
        
        # iterate over all the parameters of likelihood_model and turn off their gradients
        # for faster performance
        for param in self.likelihood_model.parameters():
            param.requires_grad = False
            
        self.use_functorch = use_functorch
        self.use_forward_mode = use_forward_mode
        self.use_vmap = use_vmap
        self.chunk_size = chunk_size
        self.verbose = verbose
        
        self.lp_norm_order = lp_norm_order
        
    def run(self):
        
        def calc_norm(x):
            return torch.norm(x, p = self.lp_norm_order).cpu().numpy()
        
        def _log_prob_func(x):
            with torch.no_grad():
                return self.likelihood_model.log_prob(x)
            
        
        scores = []
        half_len = None
        
        if self.verbose > 0:
            loader_decorated = tqdm(self.x_loader, desc="computing derivetives for batch", total=len(self.x_loader))
        else:
            loader_decorated = self.x_loader
        
        log_likelihoods = None  
        for x_batch in loader_decorated:
            
            with torch.no_grad():
                log_likelihoods_batch = self.likelihood_model.log_prob(x_batch).cpu().numpy().flatten()
            log_likelihoods = log_likelihoods_batch if log_likelihoods is None else np.concatenate([log_likelihoods, log_likelihoods_batch])
            
            L = 0
            chunk_idx = 0
            while L < len(x_batch):
                if self.verbose > 0:
                    loader_decorated.set_description(f"Running on chunk [{chunk_idx + 1}/{(len(x_batch) + self.chunk_size - 1) // (self.chunk_size)}]")
                    
                R = min(L + self.chunk_size, len(x_batch))
                x_chunk = x_batch[L:R]
                L += self.chunk_size
                chunk_idx += 1
                # Calculate the jacobian of the decode function
                if self.use_functorch:
                    jac_fn = torch.func.jacfwd if self.use_forward_mode else torch.func.jacrev
                    if self.use_vmap:
                        # optimized implementation with vmap, however, it does not work as of yet
                        jac = torch.func.vmap(jac_fn(_log_prob_func))(x_chunk)
                    else:
                        jac = jac_fn(_log_prob_func)(x_chunk)
                else:
                    jac = torch.autograd.functional.jacobian(_log_prob_func, x_chunk)

                if not self.use_vmap or not self.use_functorch:
                    half_len = len(x_chunk.shape)
                    for j in range(jac.shape[0]):
                        slice_indices = [j] + [slice(None)] * (half_len - 1) + [j] + [slice(None)]
                        scores.append(calc_norm(jac[tuple(slice_indices)].flatten()))
                else:
                    for j in range(jac.shape[0]):
                        scores.append(calc_norm(jac[j].flatten()))
                    
        scores = np.array(scores)
        
        visualize_scatterplots(
            scores = np.stack([log_likelihoods, scores]).T,
            column_names = ['log-likelihood', 'derivative'],
        )
        
        