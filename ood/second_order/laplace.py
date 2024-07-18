from .base_method import OODBaseMethod
import torch
import typing as th
import dypy as dy
import wandb
from tqdm import tqdm
from math import inf
import numpy as np
from ..local_optimization.local_optimization import OODLocalOptimization
import matplotlib.pyplot as plt

class Laplace(OODBaseMethod):
    """
    Calculates the laplace score.
    """
    def __init__(
        self,
        likelihood_model: torch.nn.Module,
        representation_rank: th.Optional[int] = None,
        x_loader: th.Optional[torch.utils.data.DataLoader] = None,
        logger: th.Optional[th.Any] = None,
        data_limit: th.Optional[int] = None,
        use_functorch: bool = False,
        use_local_optimization: bool = False,
        local_optimization_args: th.Optional[th.Dict[str, th.Any]] = None,
        score_type: th.Literal['ignore_negative', 'ignore_tails'] = 'ignore_negative',
        score_args: th.Optional[th.Dict[str, th.Any]] = None,
        bincount: int = 100,
        correction_coefficient: float = 1.0,
        visualization_range: th.Optional[th.Tuple[float, float]] = None,
        **kwargs,
    ) -> None:
        super().__init__(x_loader=x_loader, likelihood_model=likelihood_model, logger=logger, **kwargs)
        self.likelihood_model.denoising_sigma = 0
        self.likelihood_model.dequantize = False
        
        self.representation_rank = representation_rank
        self.data_limit = inf if data_limit is None else data_limit
        
        self.use_functorch = use_functorch
        self.local_optimization_args = local_optimization_args
        self.use_local_optimization = use_local_optimization
        
        self.score_type = score_type
        self.score_args = score_args or {}
        
        self.bincount = bincount
        self.correction_coefficient = correction_coefficient
        
    def run(self):
            
        def nll_func(repr_point):
            if self.representation_rank is None:
                return -self.likelihood_model.log_prob(repr_point.unsqueeze(0)).squeeze(0)
            else:
            
                def hook_fn(module, args, output):
                    # return a tuple where the first element is self.rep_point
                    # and the rest of the elements are output[1:] that are detached
                    # from the graph
                    return (repr_point.unsqueeze(0),) + tuple([x.clone().detach() for x in output[1:]])
                
                repr_module = self.likelihood_model.get_representation_module(self.representation_rank)
                handle = repr_module.register_forward_hook(hook_fn)
                dummy = torch.zeros(self.input_shape).to(device).unsqueeze(0)
                ret = -self.likelihood_model.log_prob(dummy).squeeze(0)
                handle.remove()
                return ret
             
        device = self.likelihood_model.device
        
        if self.progress_bar:
            iterable = tqdm(range(self.data_limit))
        else: 
            iterable = range(self.data_limit)
        
        all_scores = []
            
        # for any index i find the corresponding element in the corresponding batch
        lm = 0
        real_ind = 0
        tt = iter(self.x_loader)
        
        for ind in iterable:
            if ind < lm:
                x = current_batch[real_ind]
            else:
                current_batch = next(tt)
                real_ind = ind - lm
                lm += len(current_batch)
                x = current_batch[real_ind]
            real_ind += 1
            
            x = x.to(device)
            base_ll = self.likelihood_model.log_prob(x.unsqueeze(0)).squeeze(0)
            
            self.input_shape = x.shape
            if self.use_local_optimization:
                # instantiate a local optimization module
                local_optimization_module = OODLocalOptimization(likelihood_model=self.likelihood_model, x=x, representation_rank=self.representation_rank, logger=self.logger, **self.local_optimization_args)
                # run the local optimization module
                
                repr_point = local_optimization_module.run().squeeze(0).to(device)
            else:
                if self.representation_rank is None:
                    repr_point = x.clone()
                else:
                    repr_module = self.likelihood_model.get_representation_module(self.representation_rank)
                
                    def hook_fn1(module, args, output):
                        self.repr_point = output[0].clone().detach()
                    handle = repr_module.register_forward_hook(hook_fn1)
                    # call the log_prob function just to store the representation in self.repr_point
                    self.likelihood_model.log_prob(x.unsqueeze(0))
                    # remove the hook
                    handle.remove()
                    repr_point = self.repr_point.squeeze(0)
                
            # calculate the Hessian w.r.t x0
            if self.correction_coefficient > 1e-6:
                # TODO: make this compatible with functorch
                if self.use_functorch:
                    hess = torch.func.hessian(nll_func)(repr_point)
                else:
                    hess = torch.autograd.functional.hessian(nll_func, repr_point)

                # Hess has weired dimensions now, we turn it into a 2D matrix
                # by reshaping only the first half.
                # For example, if the shape of hess is (2, 2, 2, 2), then we reshape it to (4, 4)
                # For images of size (1, 28, 28), the shape of hess is (1, 28, 28, 1, 28, 28)
                # and we reshape it to (784, 784)
                hess = hess.squeeze()
                first_half = 1
                for i, dim in enumerate(hess.shape):
                    if i < len(hess.shape) // 2:
                        first_half *= dim
                hess = hess.reshape(first_half, -1).detach()
                
                # get all the eigenvalues of the hessian
                eigvals, eigvectors = torch.linalg.eigh(hess)

                # sort all the eigenvalues
                eigvals = eigvals.sort(descending=False).values
                
                if self.score_type == 'ignore_negative':
                    correction = torch.sum(torch.log(eigvals[eigvals > 0]))
                elif self.score_type == 'ignore_tails':
                    t_l = 0
                    if 'factor_left' in self.score_args:
                        t_l = int(self.score_args['factor_left'] * len(eigvals))
                    t_r = 1
                    if 'factor_right' in self.score_args:
                        t_r = max(1, int(self.score_args['factor_right'] * len(eigvals)))
                    
                    new_eigvals = eigvals[t_l:-t_r]
                    
                    correction = torch.sum(torch.log(new_eigvals[new_eigvals > 0]))
                else:
                    raise ValueError(f'Unknown score type: {self.score_type}')

                score = base_ll - self.correction_coefficient * correction
            else:
                score = base_ll
            all_scores.append(score.detach().cpu().item())
        
        # create a density histogram out of all_scores
        # and store it as a line plot in (x_axis, density)
        all_scores = np.array(all_scores)
        hist, bin_edges = np.histogram(all_scores, bins=self.bincount, density=True)
        density = hist / np.sum(hist)
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        # get the average distance between two consecutive centers
        avg_dist = np.mean(np.diff(centers))
        # add two points to the left and right of the histogram
        # to make sure that the plot is not cut off
        centers = np.concatenate([[centers[0] - avg_dist], centers, [centers[-1] + avg_dist]])
        density = np.concatenate([[0], density, [0]])
        
        data = [[x, y] for x, y in zip(centers, density)]
        table = wandb.Table(data=data, columns = ['score', 'density'])
        wandb.log({'score_density': wandb.plot.line(table, 'score', 'density', title='Score density')})