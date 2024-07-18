

import torch
import torch.nn as nn
import typing as th
import numpy as np
import functools
from tqdm import tqdm
import math
import warnings
from . import DensityEstimator
from ..utils import batch_or_dataloader
import dypy as dy
from diffusers import UNet2DModel
from einops import repeat
from typing import Union, List

class ScoreBasedDiffusion(DensityEstimator):
    """
    A simple implementation of SDE-driven diffusion (Song et al. 2020, https://arxiv.org/abs/2011.13456)

    This code adds functionalities to diffusion models to 
        (1) compute log_probabilities.
        (2) compute log of the probabilities convolved with an arbitrary variance Gaussian.
        (3) fast LID estimates directly with the continuous flow formulation of diffusions.

    The code assumes a variance preserving stochastic differential equations (VP-SDE):
    d x(t) = sqrt(\\beta(t)) d W(t),

    The corresponding probability flow ODE would be as follows:
    d/dt x(t) =  1/2 . \\beta(t) . \\nabla_{x(t)}  \\log p_t(x(t))

    Some notations used in the code:

    B(t) := \\int_0^t \\beta(s) ds (from 0 to \\infty)
    \\sigma^2(t) := (1 - e^{-B(t)}) (from 0 to 1.0)
    r(t) := \\log (e^{B(t)} - 1). \\log (from -\\infty to +\\infty)
    """
    def __init__(
        self,
        data_shape: th.Union[th.Tuple, th.List[int]],
        score_network: th.Union[torch.nn.Module, str],
        *args,
        score_network_kwargs: th.Optional[th.Dict] = None,
        T: float = 1.,
        beta_min: float = 0.1,
        beta_max: float = 20,
        **kwargs,
    ):
        """
        **Important**
            The score network outputs \\sigma(t) * true_score. 
            This parameterization helps with numerical stability as typically when the model is on a lower-dimensional manifold,
            the score explodes when it arrives at small values of t (t -> 0^+). The \\sigma(t) term cancels the explosion out.
            
        Args:
            score_network (torch.nn.module ): A model that takes in data 'x' and outputs the score with the same dimension as 'x'
            data_shape: The shape of the data in the diffusion model
            T: The final timestep
            beta_min: The scheduler for beta is assumed to start from beta_min and linearly increase to beta_max
            beta_max: The scheduler for beta is assumed to start from beta_min and linearly increase to beta_max
        """
        
        super().__init__(*args, data_shape=data_shape, **kwargs)
        
            
        if isinstance(score_network, str):
            score_network = dy.eval(score_network)
            self.score_network = score_network(**score_network_kwargs)
        else:
            self.score_network = score_network
        
            
        self.x_shape = data_shape
        self.T = T
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.beta_diff = (self.beta_max - self.beta_min) / self.T # Store this value

    def _get_drift(self, x, t):
        raise NotImplementedError("This function should be implemented in the child class")
    
    def _get_center(self, x, t):
        raise NotImplementedError("This function should be implemented in the child class")
    
    def _get_initial_log_prob(self, x):
        raise NotImplementedError("This function should be implemented in the child class")
    
    def _get_beta(self, t):
        """
        The value of \\beta(t) in the SDE.
        Here, we assume that \\beta(t) linearly increases from `beta_min` to `beta_max`.
        """
        return self.beta_min + self.beta_diff * t
    
    def _get_B(
        self,
        t,
    ):
        """
        The integral of \\beta(t) from time 0 to t.
        """
        return self.beta_min * t + self.beta_diff * t * t / 2.
    

    
    def degrade_denoise(
        self,
        x: torch.Tensor,
        all_t: Union[float, torch.Tensor],
        all_steps: Union[int, List[int], torch.IntTensor] = 10,
        return_degraded: bool = False,
        verbose: int = 0,
    ):
        """
        Degrade the input 'x' at all the timesteps in 'all_t' and then denoise it using the reverse mapping.
        
        Then return the denoised points.
        
        Args:
            x: The input data
            all_t: The timesteps at which to degrade the data
            all_steps: The number of steps to take for the reverse mapping
        Returns:
            A list of tensors, each of which is the denoised version of the input at the corresponding timestep.
        """
        x = self._data_transform(x)
        if not isinstance(all_t, torch.Tensor):
            all_t = torch.tensor(all_t).float()
        all_t = all_t.to(x.device)
        
        sigma_lst = 0
        if isinstance(all_steps, int):
            all_steps = [all_steps for _ in range(len(all_t))]
        
        denoised = []
        degraded = []
        
        sigma2_lst = 0.
        noise_lst = 0.
        steps_accum = 0
        if verbose > 0:
            iterator = tqdm(zip(all_steps, all_t), total=len(all_steps), desc="Degrading and denoising")
        else:
            iterator = zip(all_steps, all_t)
            
        for steps, t in iterator:
            eps = torch.randn_like(x) # Take random noise of the same shape as the input
            sigma2_t, _ = self._get_sigma(t)
            sigma2_t = sigma2_t.reshape(-1, *[1 for _ in range(x.dim()-1)])
            noise_new = torch.sqrt((sigma2_t - sigma2_lst)) * eps
            noise = noise_new + noise_lst
            x_degraded = self._get_center(x, t) + noise
            sigma2_lst = sigma2_t
            noise_lst = noise
            steps_accum += steps
            degraded.append(x_degraded.detach().cpu().clone())
            x_reconstructed = self.reverse_mapping(x_degraded, t_end=t, steps=steps_accum)
            denoised.append(self._inverse_data_transform(x_reconstructed).detach().cpu().clone())
            
        return denoised, degraded if return_degraded else denoised
    
    def _get_sigma(self, t):
        """ Return both \\sigma^2(t) and \\sigma(t) """
        sigma2_t = 1.0 - torch.exp(- self._get_B(t))
        sigma2_t = sigma2_t[..., None]
        return sigma2_t, torch.sqrt(sigma2_t)
    
    def _get_unnormalized_score(self, x, t):
        """ Returns the output of the network """
        return self.score_network(x, t.repeat(x.shape[0]))
    
    def get_true_score(self, x, t):
        """
        Returns the true score by dividing it again with \\sigma(t) to cancel out the effect of the reparametrization.
        """
        # print the dtype of t and x
        t = t.to(x.device).float()
        _, sigma_t = self._get_sigma(t)
        sigma_t = sigma_t.to(x.device).float()
        return self.score_network(x, t.repeat(x.shape[0])) / sigma_t
    

    def reverse_mapping(
        self,
        z: torch.Tensor,
        eps: float = 1e-5,
        steps: int = 1000,
        t_end: th.Optional[th.Union[float, torch.Tensor]] = None,
    ):
        """
        This function starts off with a noise z and performs either the reverse probability flow ODE
        or the reverse diffusion SDE to obtain a set of samples. This involves running iterations
        on an Euler-based SDE or ODE solver.
        
        This function is a proxy for flow matching when use_probability_flow is set to True 
        
        Args:
            z (Tensor of shape [batch_size, x_shape]): The starting noise
            eps: The starting timestep (this is for numerical stability)
            steps: The number of steps that the Euler solver takes
            use_probability_flow: Toggle stochasticity
        Returns:
            x: (Tensor of shape [batch_size, x_shape])
        """
        if t_end is None:
            t_end = self.T
        device = z.device
        with torch.no_grad():
            ts = torch.linspace(t_end, eps, steps=steps).to(device)
            delta_t = (t_end - eps) / (steps - 1)
            for t in ts:
                score = self.get_true_score(z, t)
                beta = self._get_beta(t)
                # Use Euler Murayama SDE solver:
                z_interim = torch.randn(z.shape).to(device)
                z += - delta_t * self._get_drift(z, t) + delta_t * beta * score + torch.sqrt(beta * delta_t) * z_interim
        return z
    
    
    def _score_jacobian_trace(
        self,
        x: torch.Tensor,
        t: th.Union[float, torch.Tensor],
        # Set by default to the less efficient but accurate method:
        method: th.Literal['hutchinson_gaussian', 'hutchinson_rademacher', 'deterministic'] = 'deterministic', 
        # The number of samples if one opts for estimation methods to save time:
        sample_count: th.Optional[int] = 100,
        custom_score: th.Optional[th.Callable] = None, 
        true_score: bool = True,
        parallel_batch_size: int = 128,
        verbose: int = 0,
    ):
        """
        This function computes tr(\\nabla_x s_{\\theta}(x, t)) using Jacobian vector products.
        
        If the size of the input matrix is small, one can opt for the 'determinstic' approach
        whereas when it is large, one might go for the hutchinson trace estimators.
        
        The implementation of all of these is as follows:
            A set of vectors of the same dimension as data are sampled and the value [v^T \\nabla_x v^T score(x, t)] is
            computed using jvp. Finally, all of these values are averaged.
            
        Args:
            x: a batch of inputs [batch_size, input_dim]
            t: time in which the score is being evaluated
            
            method: chooses between the types of methods to evaluate trace
                `hutchinson_gaussian`: Gets samples v from an isotropic Gaussian (NOTE: This estimator is not very good!)
                `hutchinson_rademacher`: Gets samples where each entry is 1 or -1 with probability 0.5. (NOTE: This is the true Hutchinson estimator)
                `deterministic`: This is not an estimator and v_i = sqrt(d) * e_i. One can show that this would produce the exact value of the tract.
            
            sample_count (Optiobal[int]): The number of samples for the stochastic methods.
            
            true_score (bool): 
                By default, this value uses the true score, but internally we sometimes need to set this 
                off for more stable approximations (check LID estimators)
        Returns:
            traces (torch.Tensor): A tensor of size [batch_size,] where traces[i] is the trace computed for 'i'
        """
        if custom_score:
            score_fn = functools.partial(custom_score, t=t)
        elif true_score:
            score_fn = functools.partial(self.get_true_score, t=t)
        else:
            score_fn = functools.partial(self._get_unnormalized_score, t=t)
            
        torch.manual_seed(100)
    
        batch_size = x.shape[0]
        data_shape = x.shape[1:]
        
        all_quadretic = []
        if method == 'deterministic': 
            d = x.numel() // x.shape[0]
            sample_count = d
            
        if method == 'hutchinson_gaussian':
            warnings.warn("The Gaussian-based hutchinson estimator is not the best! Try 'hutchinson_rademacher' instead.")
            all_v = torch.randn(size=(batch_size * sample_count, *data_shape)).cpu().float()
        elif method == 'hutchinson_rademacher':
            all_v = torch.randint(size=(batch_size * sample_count, *data_shape), low=0, high=2).cpu().float() * 2 - 1.
        elif method == 'deterministic':
            all_v = torch.eye(d).cpu().float() * math.sqrt(d)
            all_v = all_v.repeat_interleave(batch_size, dim=0).reshape((batch_size * sample_count, *data_shape))
        else:
            raise ValueError(f"Method {method} for trace computation not defined!")
        
        all_x = x.cpu().unsqueeze(0).repeat(sample_count, *[1 for _ in range(x.dim())]).reshape(batch_size * sample_count, *data_shape)
        
        all_quadretic = []
        rng = list(zip(all_v.split(parallel_batch_size), all_x.split(parallel_batch_size)))
        if verbose > 0:
            rng = tqdm(rng, desc="Computing traces")
        for vv in rng:
            v_batch, x_batch = vv
            v_batch = v_batch.to(x.device)
            x_batch = x_batch.to(x.device)
            all_quadretic.append(
                torch.sum(
                    v_batch * torch.func.jvp(score_fn, (x_batch, ), tangents=(v_batch, ))[1], 
                    dim=tuple(range(1, x.dim()))
                )
            )
        all_quadretic = torch.cat(all_quadretic)
        all_quadretic = all_quadretic.reshape((sample_count, x.shape[0]))
        return all_quadretic.mean(dim=0)

        
    
    def rho_t(
        self,
        x: torch.Tensor,
        t: float,
        adjust_center: bool = True,
        steps: int = 1000,
        device: th.Optional[torch.device] = None,
        trace_calculation_kwargs: th.Optional[th.Dict] = None,
        verbose: int = 1,
    ):
        """
        Compute the density of convolving points in 'x' with the appropriate Gaussian at time 't'; this
        Gaussian has a variance of "(e^B(t) - 1) . I"
        
        This is done through the instantaneous change-of-variables formulaton of the log_probabilities.
        For more context on the implementation, check out our document.
        
        Args:
            x: a batch of input tensors
            t: the timestep of the convolution.
            steps: The number of steps for the ODE solver
            trace_calculation_kwargs: These are arguments related to the hutchinson trace estimation which is needed to solve the ODE
            verbose: When above 0, shows a progress-bar of the ODE as it is being solved
        Returns:
            A tensor of size (batch_size, ) with the i'th element being the corresponding Gaussian convolution.
        """
        x = x.clone().detach()
        x = self._data_transform(x)
        
        if device is None:
            device = x.device
            
        trace_calculation_kwargs = trace_calculation_kwargs or {}
        trace_calculation_kwargs['verbose'] = trace_calculation_kwargs.get('verbose', verbose-1)
        
        with torch.no_grad():
            
            x = x.to(device)
            
            batch_size = x.shape[0]
            d = x.numel() // batch_size
            device = x.device
            
            # The timestep T probability is N(0, sqrt(1 - sigma_t) X0, sigma_t^2 I) where sigma_t -> 1
            log_p = torch.sum(torch.zeros_like(x), dim=tuple(range(1, x.dim())))
            
            # NOTE: Change made
            ts = torch.linspace(t, self.T, steps=steps).to(device)
            delta_s = (self.T - t) / (steps - 1)
            
            
            if adjust_center:
                x = self._get_center(x, t)
            
            rng = tqdm(ts, desc="Iterating the SDE") if verbose > 0 else ts
            
            for s in rng:
                beta_s = self._get_beta(s)
                score = self.get_true_score(x, s)                
                log_p -= delta_s * 0.5 * beta_s * self._score_jacobian_trace(x, s, **trace_calculation_kwargs)
                # NOTE: changes made here
                x -= delta_s * beta_s * 0.5 * score - delta_s * self._get_drift(x, s)
            
            # TODO: add the drift related coefficient
            
            log_p += self._get_initial_log_prob(x, t)
        
        return log_p
    
    @batch_or_dataloader()
    def log_prob(
        self,
        x: torch.Tensor,
        eps: float = 1e-2,
        steps: int = 1000,
        device: th.Optional[torch.device] = None,
        trace_calculation_kwargs: th.Optional[th.Dict] = None,
        verbose: int = 1,
    ):
        """
        The log probability of x estimated using an ODESolver.
        We use the fact that convolving with the Gaussian at a very small timestep
        will not change the original density as much. This is due to the fact that
        for small timesteps, the Gaussian that is being convolved with is essentially the degenerate 
        Dirac delta.
        """
        # clone the tensor to avoid in-place operations
        x = x.clone().detach()
        x = x.to(device)
        
        return self.rho_t(
            x=x,#self._get_center(x, eps),
            adjust_center=False,
            t=eps,
            steps=steps,
            device=device,
            trace_calculation_kwargs=trace_calculation_kwargs,
            verbose=verbose,
        )
    
     
    
    def sample(
        self, 
        n_samples, 
        eps=1e-4, 
        steps=1000, 
        device: th.Optional[torch.device] = None,
    ):
        """
        Producing samples using the reverse mapping function by first sampling from the latent space and them
        passing them through an ODE or SDE solver.
        
        Returns:
            A tensor of shape (n_samples, x_shape)
        """
        if device is None:
            device = getattr(self, 'device', torch.device('cpu'))
        
        _, sigma_T = self._get_sigma(torch.tensor(self.T).float().to(device))
        z = torch.randn((n_samples,) + tuple(self.x_shape)).to(device) * sigma_T
        
        ret = self.reverse_mapping(
            z,
            eps=eps,
            steps=steps,
        )
        return self._inverse_data_transform(ret)
    
    
    
    @batch_or_dataloader(agg_func=lambda x: torch.mean(torch.Tensor(x)))
    def loss(
        self, 
        x, 
        t_low: float = 0.0,
        t_high: float = 1.0,
        weighting_scheme: th.Literal['likelihood', 'custom'] = 'likelihood',
        weighting_fn: th.Optional[th.Callable[[torch.Tensor], torch.Tensor]] = None,
        return_aggregated: bool = True,
        distance_type: th.Literal['l1', 'l2'] = 'l2',
    ):
        """
        The score matching loss used for training the diffusion model.
        
        For a given x, a random time is sampled and then the weight for that time is picked as \\sigma^(t) 
        (The likelihood weighting). Finally, the difference between the score and the noise value times this weight is minimized 
        which corresponds to the denoising score matching loss.
        """
        x = self._data_transform(x)
        t = t_low + (t_high - t_low) * self.T * torch.rand(x.shape[0], device=x.device) # Take a random timestep between [0, T]
        
        eps = torch.randn_like(x) # Take random noise of the same shape as the input
        sigma2_t, sigma_t = self._get_sigma(t)
        sigma2_t = sigma2_t.reshape(-1, *[1 for _ in range(x.dim()-1)])
        sigma_t = sigma_t.reshape(-1, *[1 for _ in range(x.dim()-1)])
        x_input = self._get_center(x, t) + sigma_t * eps
        
        unnormalized_score = self.score_network(x_input, t)
        
        if distance_type == 'l2':
            unnormalized_error = torch.square(unnormalized_score + eps)
        elif distance_type == 'l1':
            unnormalized_error = torch.abs(unnormalized_score + eps)
        else:
            raise ValueError(f"Distance type {distance_type} not defined!")
        
        if weighting_scheme == 'likelihood':
            error = unnormalized_error
        elif weighting_scheme == 'custom':
            if weighting_fn is None:
                raise ValueError("For custom weighting, a `weighting_fn` must be provided!")
            weights = weighting_fn(t).reshape(-1, *[1 for _ in range(x.dim()-1)])
            normalization_factor = sigma2_t if distance_type == 'l2' else sigma_t
            error = unnormalized_error / normalization_factor * weights
        else:
            raise ValueError(f"Weighting scheme {weighting_scheme} not defined!")
        loss = torch.sum(error.flatten(start_dim=1), dim=1)
        if return_aggregated: # TODO: is this needed?
            return loss.mean()
        else:
            return loss
        
class VESDE_Diffusion(ScoreBasedDiffusion):
    
    model_type = "VESDE_Diffusion"
    
    def __init__(
        self,
        *args,
        T: float = 1.,
        g_min: float = -5,
        g_max: float = 5,
        **kwargs,
    ):
        self.T = T
        self.g_min = g_min
        self.g_max = g_max
        self.g_diff = (self.g_max - self.g_min) / self.T # Store this value
        return super().__init__(
            *args,
            T=T,
            beta_min=math.exp(g_min)/self.g_diff,
            beta_max=math.exp(g_max)/self.g_diff,
            **kwargs,
        )
            
    def _get_beta(self, t):
        """
        The value of \\beta(t) in the SDE.
        Here, we assume that \\beta(t) linearly increases from `beta_min` to `beta_max`.
        """
        if isinstance(t, torch.Tensor):
            return torch.exp(self.g_min + self.g_diff * t) * self.g_diff
        else:
            return math.exp(self.g_min + self.g_diff * t) * self.g_diff
    
    def _get_B(
        self,
        t,
    ):
        """
        The integral of \\beta(t) from time 0 to t.
        """
        if isinstance(t, torch.Tensor):
            return (torch.exp(self.g_min + self.g_diff * t) - math.exp(self.g_min))
        
        return (math.exp(self.g_min + self.g_diff * t) - math.exp(self.g_min))
    
    
    def _get_sigma(self, t):
        """ Return both \\sigma^2(t) and \\sigma(t) """
        sigma2_t = self._get_B(t)
        sigma2_t = sigma2_t[..., None]
        return sigma2_t, torch.sqrt(sigma2_t)
    
    def _get_drift(self, x, t):
        return torch.zeros_like(x)
    
    def _get_center(self, x, t):
        return x
    
    
    def _get_initial_log_prob(self, x, t):
        d = x.numel() // x.shape[0]
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t).float().to(x.device)
        sigma2_t, _ = self._get_sigma(torch.tensor(self.T).float().to(x.device))
        sigma2_t = sigma2_t.squeeze().item()
        ret = 0.5 * d * np.log(2 * np.pi)
        ret = -0.5 * torch.sum(x * x, dim=tuple(range(1, x.dim()))) / sigma2_t + ret.item()
        return ret


class VPSDE_Diffusion(ScoreBasedDiffusion):
    
    model_type = "VPSDE_Diffusion"

    def _get_drift(self, x, t):
        return - 0.5 * self._get_beta(t) * x
    
    def _get_center(self, x, t: th.Union[float, torch.Tensor]):
        B_t = self._get_B(t)
        if isinstance(B_t, float):
            exp_B_t = math.exp(-B_t/2)
            return x * exp_B_t
        exp_B_t = torch.exp(-B_t/2).to(x.device)
        exp_B_t = exp_B_t.reshape(-1, *[1 for _ in range(x.dim() - 1)])
        return exp_B_t * x
    
    def _get_initial_log_prob(self, x, t):
        d = x.numel() // x.shape[0]
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t).float().to(x.device)
        ret = 0.5 * (self._get_B(self.T) - self._get_B(t)) * d
        sigma2_t, _ = self._get_sigma(torch.tensor(self.T).float().to(x.device))
        sigma2_t = sigma2_t.squeeze().item()
        ret += 0.5 * d * np.log(2 * np.pi)
        ret = -0.5 * torch.sum(x * x, dim=tuple(range(1, x.dim()))) / sigma2_t + ret.item()
        return ret