
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

class HessianSpectrumMonitor(OODBaseMethod):
    """
    This class of methods calculates the derivative of the likelihood model
    log_prob w.r.t the representation, and then calculates the eigenvalues of
    the hessian of the log_prob w.r.t the representation.
    
    After calculating the eigenvalues for each datapoint, it then sorts all the
    eigenvalues and logs a histogram of the eigenvalues.
    
    For multiple datapoints, it calculates eig_mean[k] which is the average k'th largest
    eigenvalue for all the datapoints. It also calculates eig_std[k] which is the standard
    deviation of the k'th largest eigenvalue for all the datapoints.
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
        
        # plotting the second order and first_order characteristics
        plot_std: bool = False,
        second_order_score_type: th.Literal['ignore_negative', 'ignore_tails', 'absolute_value', 'force_negative'] = 'ignore_negative',
        second_order_score_args: th.Optional[th.Dict[str, th.Any]] = None,
        
        cancel_background: bool = False,
        
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
        self.plot_std = plot_std
        
        self.second_order_score_type = second_order_score_type
        self.second_order_score_args = second_order_score_args or {}
        
        self.cancel_background = cancel_background
        
    def run(self):
        zero_order_characteristics = []
        first_order_characteristics = []
        second_order_characteristics = []
        
        def nll_func(repr_point, dummy=None):
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
                dummy = (torch.zeros(self.input_shape) if dummy is None else dummy).to(device).unsqueeze(0)
                ret = -self.likelihood_model.log_prob(dummy).squeeze(0)
                handle.remove()
                return ret
        
        if self.use_functorch:
            self.hessian_function = torch.func.hessian(nll_func)
            
        eigval_history = None
        device = self.likelihood_model.device
        
        if self.progress_bar:
            iterable = tqdm(range(self.data_limit))
        else: 
            iterable = range(self.data_limit)
            
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
                    
            
            # Second order characteristics
            # activate requires_grad for repr_point
            repr_point.requires_grad = True
            
            if self.cancel_background:
                dummy = torch.zeros_like(x).detach()
            else:
                dummy = x.clone().detach()
                
            loss = nll_func(repr_point, dummy=dummy)
            # First order characteristics
            zero_order_characteristics.append(loss.item())
            
            loss.backward()
            first_order_score = torch.norm(repr_point.grad, p=1.0) / repr_point.grad.numel()
            first_order_characteristics.append(first_order_score.detach().cpu().item())
            # zero out the gradient
            repr_point.grad.zero_()
            # deactivate requires_grad for repr_point
            repr_point.requires_grad = False
            
                
            # calculate the Hessian w.r.t x0
            
            # TODO: make this compatible with functorch
            if self.use_functorch:
                hess = self.hessian_function(repr_point)
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
            
            if self.second_order_score_type == 'ignore_negative':
                second_order_score = torch.sum(torch.log(eigvals[eigvals > 1e-6]))
            elif self.second_order_score_type == 'absolute_value':
                second_order_score = torch.sum(torch.log(torch.abs(eigvals)))
            elif self.second_order_score_type == 'force_negative':
                repl = torch.where(eigvals > 1e-6, eigvals, torch.ones_like(eigvals) * 1e-6)
                second_order_score = torch.sum(torch.log(repl))
            elif self.second_order_score_type == 'ignore_tails':
                t_l = 0
                if 'factor_left' in self.second_order_score_args:
                    t_l = int(self.second_order_score_args['factor_left'] * len(eigvals))
                t_r = 1
                if 'factor_right' in self.second_order_score_args:
                    t_r = max(1, int(self.second_order_score_args['factor_right'] * len(eigvals)))
                
                new_eigvals = eigvals[t_l:-t_r]
                
                second_order_score = torch.sum(torch.log(new_eigvals[new_eigvals > 1e-6]))
            else:
                raise ValueError(f'Unknown score type: {self.second_order_score_type}')
            
            second_order_characteristics.append(second_order_score.detach().cpu().item())
            # detach and put it on cpu and add it to the list
            if eigval_history is None:
                eigval_history = eigvals.detach().cpu().unsqueeze(0)
            else:
                eigval_history = torch.cat([eigval_history, eigvals.detach().cpu().unsqueeze(0)], dim=0)
        
        data = [[x, y, z] for x, y, z in zip(zero_order_characteristics, first_order_characteristics, second_order_characteristics)]
        table = wandb.Table(data=data, columns=["nll", "||grad nll||/dim", "log det Hessian nll"])
        
        wandb.log({
            'characteristics (0-vs-1)': wandb.plot.scatter(table, "nll", "||grad nll||/dim", title="0-vs-1")
        })
        wandb.log({
            'characteristics (0-vs-2)': wandb.plot.scatter(table, "nll", "log det Hessian nll", title="0-vs-2")
        })
        wandb.log({
            'characteristics (1-vs-2)': wandb.plot.scatter(table, "||grad nll||/dim", "log det Hessian nll", title="1-vs-2")
        })
        
        # calculate the mean and std of the k'th largest eigenvalue
        # for all the datapoints using the eigval_history
        eig_mean = torch.mean(eigval_history, dim=0)
        if self.plot_std:
            eig_std = torch.std(eigval_history, dim=0)
        x_axis = torch.arange(len(eig_mean))
        
        # create three lists, one with eig_mean, one with eig_mean + eig_std and one with eig_mean - eig_std
        # and log them to wandb
        if self.plot_std:
            eig_mean_list = eig_mean.tolist()
            eig_plus_std_list = (eig_mean + eig_std).tolist()
            eig_minus_std_list = (eig_mean - eig_std).tolist()
            x_axis_list = x_axis.tolist()
            
            wandb.log({
                'hessian_eigen_spectrum': wandb.plot.line_series(
                    xs = [x_axis_list, x_axis_list, x_axis_list],
                    ys = [eig_mean_list, eig_plus_std_list, eig_minus_std_list],
                    keys=['eig_mean', 'eig_mean + eig_std', 'eig_mean - eig_std'],
                    title="Hessian Eigen Spectrum",
                )
            })
        else:
            data = [[x, y] for x, y in zip(x_axis.tolist(), eig_mean.tolist())]
            table = wandb.Table(data=data, columns = ["i", "eig_mean"])
            wandb.log(
                {"hessian_eigen_spectrum" : wandb.plot.line(table, "i", "eig_mean",
                    title="Hessian Eigen Spectrum")})