from ..base import ModelBasedLID
from .jacobian_calculator import FlowJacobianCalculator
from typing import Callable, Iterable
import torch
from ..base import LIDInputType, ScaleType
from tqdm import tqdm
import math
import functools
import dypy as dy
from abc import ABC, abstractmethod


class FastFlowLIDEstimator(ModelBasedLID, ABC):
    """
    This class contains a set of LID estimators that use the Jacobian of the flow mapping function
    to estimate LID.
    
    Some examples include:
    1) Looking at the SVD decomposition of the Jacobian and setting a threshold for the singular values
    2) Linearly approximating the Gaussian convolutions and using a LIDL-type estimator for the 
        derivative of this Jacobian w.r.t. the log standard deviation of the Gaussian.
        
    In all cases, the class needs a flow_jacobian_calculator that can be used to calculate the Jacobian
    of the flow mapping function. The jacobian calculator in turn contains the setting for the functional
    pytorch methods that are used to calculate the Jacobian. For more information, check the `jacobian_calculator.py`
    file.
    """
    def __init__(
        self,
        *args,
        flow_jacobian_calculator_class: str,
        flow_jacobian_calculator_kwargs: dict,
        verbose: int = 0, # verbosity
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.flow_jacobian_calculator_instantiate = functools.partial(dy.eval(flow_jacobian_calculator_class), **flow_jacobian_calculator_kwargs)
        self.verbose = verbose
        
    def fit(self):
        '''After training the model (if applicable), the jacobian calculator is instantiated.'''
        super().fit()
        self.jacobian_calculator = self.flow_jacobian_calculator_instantiate(self.model)
    
    def buffer_data(self, x: LIDInputType):
        '''This function is incomplete but performs checks that appear in all the child classes'''
        
        # check if x is an iterable or not,
        # if not, make it iterable and store this info to unwind later
        self.data_is_iterable = True
        if not isinstance(x, Iterable):
            self.buffered_x = [x]
            self.data_is_iterable = False
        else:
            self.buffered_x = x
            
        # Compute the Jacobian and latent values
        # self.buffered_jax would be an iterable of batches with size [batch_size, ambient_dim, latent_dim]
        # self.bufefred_z would be an iterable of batches with size [batch_size, latent_dim]
        self.buffered_jax, self.buffered_z = self.jacobian_calculator.calculate_jacobian(self.buffered_x, flatten=True)

    
    def compute_lid_buffer(self, scale: ScaleType = None):
        '''This function is incomplete but performs checks that appear in all the child classes'''
        if scale is None:
            raise ValueError("Scale cannot be None for FastFlowLIDEstimator")
        
        if not hasattr(self, 'buffered_jax') or not hasattr(self, 'buffered_z'):
            raise ValueError("Compute LID called before buffering or caching the data!")

class ThresholdSVDFlowLIDEstimator(FastFlowLIDEstimator):
    """
    To perform LID computation, this method computes the Jacobian of the flow mapping function
    at every point that is given, then a simple threshold derived from the `scale` parameter
    is used to threshold on the singular values of the Jacobian. The number of singular values
    that are above that threshold are then used to estimate the LID.
    
    NOTE: the actual threshold is obtained by exponentiating the scale, i.e., threshold = exp(2 * scale)
    
    This is based on Horvat et al. (2021) and is a simple and fast method for LID estimation.
    (https://proceedings.neurips.cc/paper_files/paper/2022/file/4f918fa3a7c38b2d9b8b484bcc433334-Paper-Conference.pdf)
    """
    def buffer_data(self, x: LIDInputType):
        
        # call the parent to buffer the jacobians and the latents
        super().buffer_data(x)
            
        # visualize progressbar if verbose > 0
        if self.verbose > 0:
            jax_wrapped = tqdm(self.buffered_jax, desc="calculating eigendecomposition of jacobians")
        else:
            jax_wrapped = self.buffered_jax
        
        self.jacobian_singular_vals = []
        self.jacobian_latent_rotation = []
        
        for j in jax_wrapped:
            j = j.to(self.device)
            
            # take care of extremes and corner cases
            # This should take place to avoid any CUDA errors while dealing with
            # large datasets 
            jtj = torch.matmul(j.transpose(1, 2), j)
            jtj = 0.5 * (jtj.transpose(1, 2) + jtj) # forcably symmetrize
            jtj = torch.clamp(jtj, min=-10**4.5, max=10**4.5) # clamp to get rid of ilarge values
            jtj = torch.where(jtj.isnan(), torch.zeros_like(jtj), jtj) # get rid of NaNs
            
            # perform eigendecomposition
            L, Q = torch.linalg.eigh(jtj)
            L = torch.where(L > 1e-20, L, 1e-20 * torch.ones_like(L))
            
            # move to CPU memory to circumvent overloading the GPU memory
            self.jacobian_singular_vals.append(L.cpu())
            self.jacobian_latent_rotation.append(Q.cpu())
    
    def compute_lid_buffer(self, scale: ScaleType = None):
        super().compute_lid_buffer(scale)
        
        lid_values = []
        # the actual threshold is obtained by exponentiating the scale
        # (this is for consistency reasons)
        var = math.exp(2 * scale)
        for batch_singular_values in self.jacobian_singular_vals:
            batch_singular_values = batch_singular_values.to(self.device)
            batch_lid = torch.sum((batch_singular_values > var).int(), dim=1)
            lid_values.append(batch_lid)
        
        return lid_values if self.data_is_iterable else lid_values[0]
        
    

class _SpectralFastFlowLID(ThresholdSVDFlowLIDEstimator, ABC):
    """
    These sets of classes 
    """
    
    @abstractmethod
    def _calc(self, z_0, jtj_eigvals, var, jtj_rot):
        """
        Given a single jacobian alongside the eigehvalues of J^TJ and the rotations,
        computes the latent statistical value itself. This function is internally used in calculate_statistics,
        but isolates the actual methametical computations per-datapoint.
        """
        pass
    
    def compute_lid_buffer(self, scale: ScaleType = None):
        super().compute_lid_buffer(scale)
        
        lid_values = []
        
        var = math.exp(2 * scale)
        for batch_jtj_eigvals, batch_jtj_rot, batch_z in zip(self.jacobian_singular_vals, self.jacobian_latent_rotation, self.buffered_z):
            
            inner_values = []
            for jtj_eigvals, jtj_rot, z_0 in zip(batch_jtj_eigvals, batch_jtj_rot, batch_z):
                jtj_eigvals = jtj_eigvals.to(self.device)
                jtj_rot = jtj_rot.to(self.device)
                z_0 = z_0.to(self.device)
                inner_values.append(self._calc(jtj_eigvals, var, jtj_rot, z_0))
            lid_values.append(torch.stack(inner_values))
        
        if self.data_is_iterable:
            return lid_values
        return lid_values[0]


class SpectralLogGaussianConvolutionFlowEstimator(_SpectralFastFlowLID):
    """
    NOTE: 
        This isn't an LID estimator! But the implementation is so similar to the LID estimators
        that it makes sense to put it here, and instead provide an alternate API for the user to use.
        
    Even though this class inherits from the LID estimator, it is not an LID estimator. It is estimating
    a value that also has a scale parameter, but it is not the LID.
    
    The main methods are:
    
    1) buffer_data: Given a set of data points, it buffers the Jacobians and the latent values
    2) compute_rho_buffered: Given a scale, it computes the latent statistical values for the buffered data
    3) compute_rho: Given a set of data points and a scale, it computes the latent statistical values for the data points
    4) fit: This function performs any necessary fitting for the model.
    """
    
    def _calc(self, z_0, jtj_eigvals, var, jtj_rot):
        """
        Given a single jacobian alongside the eigehvalues of J^TJ and the rotations,
        computes the latent statistical value itself. This function is internally used in calculate_statistics,
        but isolates the actual methametical computations per-datapoint.
        """
        
        d = len(z_0)
        log_pdf = -0.5 * d * math.log(2 * math.pi)
        log_pdf = log_pdf - 0.5 * torch.sum(torch.log(jtj_eigvals + var))
        z_ = (jtj_rot.T @ z_0.reshape(-1, 1)).reshape(-1)
        log_pdf = log_pdf - torch.sum(jtj_eigvals * z_ * z_ / (jtj_eigvals + var)) / 2
           
        return log_pdf

    def compute_rho_buffered(
        self,
        scale: ScaleType,
    ):
        return self.compute_lid_buffer(scale)

    def compute_rho(
        self,
        x: LIDInputType,
        scale: ScaleType,
        use_cache: bool = True,
    ):
        return self.estimate_lid(x, scale, use_cache)
    
class SpectralLinearizationFlowLIDEstimator(_SpectralFastFlowLID):
    """
    This one uses the implementation of the Gaussian convolution to estimate the LID
    by just taking the derivative of the log likelihood w.r.t. the log standard deviation.
    """ 
    def _calc(self, z_0, jtj_eigvals, var, jtj_rot):
        """
        Given a single jacobian alongside the eigehvalues of J^TJ and the rotations,
        computes the latent statistical value itself. This function is internally used in calculate_statistics,
        but isolates the actual methametical computations per-datapoint.
        """
                    
        z_ = (jtj_rot.T @ z_0.reshape(-1, 1)).reshape(-1)
        ret = - torch.sum(1 / (jtj_eigvals + var))
        ret = ret + torch.sum(jtj_eigvals * (z_ / (jtj_eigvals + var)) ** 2)
        
        return ret * var
    