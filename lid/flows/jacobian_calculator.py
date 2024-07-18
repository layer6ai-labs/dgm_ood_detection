
import torch
from tqdm import tqdm
from ..utils import stack_back_iterables
import numpy as np
import functools
from abc import ABC, abstractmethod
from typing import Iterable, Union, Optional
import warnings

class FlowJacobianCalculator:
    """
    This is a class that performs Jacobian computation using functional pytorch. We assume that the `nflow_wrapper`
    which is passed to the constructor is a wrapper around the flow models in the nflows library 
    (https://github.com/bayesiains/nflows) and that we can access the value by calling `nflow_wrapper._nflow`
    
    The main method here is the `calculate_jacobian` method that takes in a torch loader of different datapoints
    and returns the Jacobians of the decode function at those points.
    
    In addition, the nflows wrapper is assumed to have two functions _data_transform and _inverse_data_transform
    which is applied to the data before being passed to the nflows transforms. If these functions are not present,
    it is replaced by an identity function.
    
    
    For optimizing the computation there are a lot of handy parameters that are specified
    in the constructor of the class.
    
    1.  With `use_functorch` set to False, it would use the original autograd functionality
        of pytorch to come up with the jacobian which uses the backward-backward trick. 
        The computation is extremely slow, but always reliable!
    2.  When `use_functorch` is set to True in the encoding model, then it either performs
        backward mode or forward mode that can be set for calculating the Jacobians. There are
        also knobs to use `vmap` or not. If `vmap` is set to False, then cross batch gradient computations
        will also happen. Also, the parameter chunk_size allows parallel computing of the Jacobians themselves.
    
    **In practice, using the vmap and functorch with the largest permittable chunksize would be the fastest
    configuration.** 
    """
    def __init__(
        self,
        nflow_wrapper: torch.nn.Module,
        use_vmap: bool = False,
        use_functorch: bool = True,
        use_forward_mode: bool = True,
        chunk_size: Optional[int] = None,
        # verbosity
        verbose: int = 0,
        # neuanced parameter for data preprocessing and transformation
        diff_transform: bool = False,
    ) -> None:  
        """
        nflow_wrapper:      The flow likelihood model that is used for the encoding and decoding, it is of type
                            torch.nn.Module
        use_vmap:           Whether to use the vmap function of functorch or not.
        use_functorch:      Whether to use functorch or not. If set to False, then the torch.autograd.functional
        use_forward_mode:   Whether to use the forward mode or the reverse mode of functorch.
        chunk_size:         The size of the chunks to be used for the computation of the Jacobian. If set to None,
                            then the whole batch is used for the computation of the Jacobian.
        verbose:            The verbosity level of the computation. If set to 0, then no progress bar is shown.  
                            if verbose > 0, then a progress bar is shown for the computation of the Jacobian. 
        diff_transform:     The nflow_wrapper has a form of preprocessing on the data before passing to the actual
                            flow transform. When `diff_transform=True`, then transformation is considered as part of the
                            encoding and decoding process. That said, `diff_transoform=True` is only valid when these transformations are
                            actually differentiable, otherwise, Jacobian cannot be computed.
        """
        
        if not hasattr(nflow_wrapper, "_nflow"):
            raise ValueError("The nflow_wrapper does not have the attribute _nflow. Please make sure that the nflow_wrapper is a valid wrapper for an nflow flow model.")
        
        if not hasattr(nflow_wrapper, "_data_transform"):
            # add an identity method to the nflow_wrapper
            nflow_wrapper._data_transform = lambda x: x
        
        if not hasattr(nflow_wrapper, "_inverse_data_transform"):
            # add an identity method to the nflow_wrapper
            nflow_wrapper._inverse_data_transform = lambda x: x
            
        self.nflow_wrapper = nflow_wrapper
        # Set the model to evaluation mode if its not already!
        # This is because we need the model to act in a deterministic way
        # this gets rid of all the dropout randomness or the buffers used for
        # potential batch normalization
        
        self.nflow_wrapper.eval()
        
        # iterate over all the parameters of nflow_wrapper and turn off their gradients
        # for faster performance
        for param in self.nflow_wrapper.parameters():
            param.requires_grad = False
            
        self.use_functorch = use_functorch
        self.use_forward_mode = use_forward_mode
        self.use_vmap = use_vmap
        self.chunk_size = chunk_size
        self.verbose = verbose
        self.diff_transform = diff_transform
    
    def calculate_jacobian(
        self, 
        loader: Iterable[torch.Tensor], 
        flatten: bool = True
    ):
        """
        This function takes in an iterable with data of the form
        [
            (x_batch_1),
            (x_batch_2),
            ...,
            (x_batch_n)
        ]
        where each x_batch_i is a batch of data points.
        
        This function calculates the encoding of each of the datapoints,
        then using the latent representation, calculates the Jacobian of
        the function that maps the latent encoding to the final datapoint itsel
        implemented by the decode function.
        
        
        Args:
            loader:     The loader of the data points.
            flatten:    When set to True the datapoints are flattened and the returned list will contain
                        jacobians of size [batch_size x numel_ambient x numel_latent]
                        Otherwise, the shapes are preserved and we will have jacobians of size [batch_size x shape_ambient x shape_latent]
        Returns:
            The function returns a list in the same format as the input that is acted upon as a loader, each element of the loader
            will contain a batch of Jacobians that have a specific shape specified by the 'flatten' argument.
        """
    
        # set a progress bar if verbose > 0
        if self.verbose > 0:
            loader_decorated = tqdm(loader, desc="Computing Jacobians of loader")
        else:
            loader_decorated = loader
        
        jax = []
        z_values = []
           
        for x_batch in loader_decorated:
            
            if not self.diff_transform:
                x_batch = self.nflow_wrapper._data_transform(x_batch)
                
            # encode to obtain the latent representation in the Gaussian space
            
            z = self.encode(x_batch)
            # count the number of NaNs in z
                
            # Since Jacobian computation is heavier than the normal batch_wise
            # computations, we have another level of batching here
            step = self.chunk_size
            if self.chunk_size is None:
                step = z.shape[0]
            
            progress = 0
            half_len = None
            
            for l in range(0, z.shape[0], step):
                progress += 1
                if self.verbose > 1:
                    loader_decorated.set_description(f"Computing jacobian chunk [{progress}/{z.shape[0] // step}]")
        
                r = min(l + step, z.shape[0])
                
                # Get a batch of latent representations
                z_s = z[l:r]
                
                # check if z_s contains any NaN values
                if torch.isnan(z_s).any():
                    warnings.warn(f">> NaN values detected in the latent representation. Skipping batch {l}:{r} of the data.")
                    # replace the nans with 0.0
                    z_s = torch.where(torch.isnan(z_s), torch.zeros_like(z_s), z_s)
                
                # Calculate the jacobian of the decode function
                if self.use_functorch:
                    jac_fn = torch.func.jacfwd if self.use_forward_mode else torch.func.jacrev
                    if self.use_vmap:
                        # optimized implementation with vmap, however, it does not work as of yet
                        jac_until_now = torch.func.vmap(jac_fn(functools.partial(self.decode, batchwise=False)))(z_s)
                    else:
                        jac_until_now = jac_fn(self.decode)(z_s)
                else:
                    jac_until_now = torch.autograd.functional.jacobian(self.decode, z_s)

                
                # Reshaping the jacobian to be of the shape (batch_size, latent_dim, latent_dim)
                if self.use_vmap and self.use_functorch:
                    if flatten:
                        jac = jac_until_now.reshape(z_s.shape[0], -1, z_s.numel() // z_s.shape[0])
                    else:
                        jac = jac_until_now
                else:
                    if flatten:
                        jac_until_now = jac_until_now.reshape(z_s.shape[0], -1, z_s.shape[0], z_s.numel() // z_s.shape[0])
                    jac = []
                    
                    if half_len is None:
                        half_len = 0
                        while np.prod(jac_until_now.shape[:half_len]) != np.prod(jac_until_now.shape[half_len:]):
                            half_len += 1
                    
                    for j in range(jac_until_now.shape[0]):
                        slice_indices = [j] + [slice(None)] * (half_len - 1) + [j] + [slice(None)] * (len(jac_until_now.shape) - half_len - 1)
                        jac.append(jac_until_now[tuple(slice_indices)])
                    jac = torch.stack(jac)
                if flatten:
                    z_s = z_s.reshape(z_s.shape[0], -1)
                z_values.append(z_s.cpu().detach())
                jax.append(jac.cpu().detach())

        
        # return jax, z_values 
        return stack_back_iterables(loader, jax, z_values)
        
    def encode(self, x, batchwise: bool = True):
        # turn off the gradient for faster computation
        # because we don't need to change the model parameters
        # self.nflow_wrapper.eval()
        # with torch.no_grad():
        if self.diff_transform:
            x = self.nflow_wrapper._data_transform(x)
        if batchwise:
            z = self.nflow_wrapper._nflow.transform_to_noise(x)
        else:
            z = self.nflow_wrapper._nflow.transform_to_noise(x.unsqueeze(0)).squeeze(0)
        return z

    def decode(self, z, batchwise: bool = True):
        # turn off the gradient for faster computation
        # because we don't need to change the model parameters
        # self.nflow_wrapper.eval()
        # with torch.no_grad():
        if batchwise:
            x, logdets = self.nflow_wrapper._nflow._transform.inverse(z)
        else:
            x, logdets = self.nflow_wrapper._nflow._transform.inverse(z.unsqueeze(0))
            x = x.squeeze(0)
        
        if self.diff_transform:
            x = self.nflow_wrapper._inverse_data_transform(x)

        return x
