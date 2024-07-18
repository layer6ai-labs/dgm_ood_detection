from abc import ABC, abstractmethod

import torch
import numpy as np
from typing import Union, Optional, Iterable, Callable
from .utils import get_device_from_loader
import warnings

LIDInputType = Union[torch.Tensor, np.ndarray]
ScaleType = Union[float, int]

class LocalIntrinsicDimensionEstimator(ABC):
    
    def __init__(
        self,
        data,
        ambient_dim: int,
        ground_truth_lid = None,
    ):
        """
        Initialize the estimator with the data that we are planning to use for LID estimation 
        and also take in an optional ground truth LID if it is available
        """
        self.data = data
        self.ground_truth_lid = ground_truth_lid
        self.ambient_dim = ambient_dim
        
    @abstractmethod
    def fit(
        self,
    ):
        """
        Fit the estimator to the data and do one-time processing on the model if necessary 
        before using the lid estimation methods.
        """
        pass
    

    @abstractmethod
    def buffer_data(
        self,
        x: Union[LIDInputType, Iterable[LIDInputType]],
    ):
        """
        Store data and perform any preprocessing necessary for LID estimation on that 
        particular set of data. This is useful for caching and speedup purposes.
        
        Args:
            x: A batch [batch_size, data_dim] or an iterable over the batches of data points at which to estimate the LID.
        """
        pass
    
    @abstractmethod
    def compute_lid_buffer(
        self,
        scale: Optional[ScaleType] = None,
    ):
        """
        Compute the LID for the buffered data, but with a different scale.
        This is useful for caching and speedup purposes, because many times
        we keep the data the same but change the scale.
        
        For more information on the scale, see the `estimate_lid` function.
        
        Args:
            scale (Optional[Union[float, int]]):
                The scale at which to estimate the LID. If None, the scale will be estimated from the data
                when set to None, the scale will be set automatically.
        Returns:
            lid: A batch [batch_size, data_dim] or an iterable over the batches of LID estimates, depending on the buffer type.
        """
        pass
    
    def estimate_lid(
        self,
        x: Union[LIDInputType, Iterable[LIDInputType]],
        scale: Optional[ScaleType] = None,
        use_cache: bool = False,
    ):
        """
        Estimate the local intrinsic dimension of the data at given points. 
        The input is batched, so the output should be batched as well.
        there is also a scale parameter that can be used to estimate the LID at different scales
        by scale we mean what level of data perturbation do we ignore. 
        As an extreme example, for an excessively large scale, the LID will be the same for all points and equal to 0.
        On the other hand, for an excessively small scale, the LID will be the same for all points and equal to the ambient dimension
        
        Args:
            x:
                A batch [batch_size, data_dim] or an iterable over the batches of data points at which to estimate the LID.
            scale (Optional[Union[float, int]]):
                The scale at which to estimate the LID. If None, the scale will be estimated from the data
            use_cache: bool
                Whether to use the cache for the LID estimation. Cache is used for speedup, and it should only be used 
                when the 'x' input is "identical" to the input 'x' that was given the last time the function was called.
                *Setting this to True is not advised anymore, and the `buffer_data` and `compute_lid_buffer` functions should be used instead.*
        Returns:
            lid:
                Returns a batch (batch_size, ) or iterable of LID values for the input data, depending on the input type.
        """
        if not use_cache:
            self.buffer_data(x)
        else:
            warnings.warn("Instead of using `estimate_lid` with `use_cache`, use `buffer_data` and `compute_lid_buffer` separately!")
        return self.compute_lid_buffer(scale)
    

class ModelBasedLID(LocalIntrinsicDimensionEstimator):
    """
    An abstract class for estimators that use a generative model that matches the distribution
    to estimate LID. An example of such a method is LIDL (https://arxiv.org/abs/2206.14882).
    """
    def __init__(
        self,
        ambient_dim: int,
        model: torch.nn.Module,
        train_fn: Callable[[torch.nn.Module, torch.utils.data.Dataset], torch.nn.Module] = None,
        device: Optional[torch.device] = None,
        ground_truth_lid = None,
        data: Optional[torch.utils.data.Dataset] = None,
    ):
        """

        Args:
            model (torch.nn.Module):
                The likelihood model that the LID estimator will use for estimation
            train_fn (Callable[[torch.nn.Module, torch.utils.data.Dataset], torch.nn.Module]): 
                This is a function that takes in a torch module and a torch dataset and trains the module on the dataset.
            device (Optional[torch.device]):
                The device used for computation. If None, the device will be inferred from the data.
        """
        super().__init__(
            data=data,
            ambient_dim=ambient_dim,
            ground_truth_lid=ground_truth_lid,
        )
        # check if self.data is a torch Dataset or not
        if self.data is not None and not isinstance(self.data, torch.utils.data.Dataset):
            raise ValueError("The data input to the constructor should be a torch Dataset")
        
        self.model = model
        self.train_fn = train_fn if train_fn is not None else (lambda model, data: model)
        model_device = next(model.parameters()).device
        self.device = device if device is not None else model_device
        self.model = self.model.to(self.device)
    
    def fit(self):
        '''Fit the estimator to the data'''
        self.model = self.train_fn(self.model, self.data)
        
        