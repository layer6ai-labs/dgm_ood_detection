from nflows.distributions import Distribution, StandardNormal
from nflows.flows.base import Flow
import typing as th
from . import DensityEstimator
from ..utils import batch_or_dataloader
from nflows.transforms import Transform, CompositeTransform,  MultiscaleCompositeTransform, PiecewiseRationalQuadraticCouplingTransform, ActNorm
from nflows.transforms.coupling import CouplingTransform
from nflows.transforms import AffineCouplingTransform
import dypy as dy
import torch
from pprint import pprint
from nflows.nn import nets as nets
import copy
import math
import wandb 

 
class ConfigurableCouplingTransform(Transform):
    """
    A wrapper for the nflow coupling transforms that
    can be fully configured using a set of primitives in
    a configuration file.
    """
    def __init__(
        self,
        coupling_transform_cls: th.Union[str, th.Type[CouplingTransform]],
        net: th.Dict[str, th.Any],
        mask: th.Dict[str, th.Any],
        **kwargs,
    ):
        """
        Here we define the network that the coupling transform uses
        alongside a set of parameters defining the mask.
        
        Moreover, here we define the coupling transform class that is being used as well.
        This can work with any coupling transform class that is defined in nflows.

        Args:
            net:
                class_path: The class of network which is being used
                            Note that the class constructor must have the first
                            two arguments as in_features and out_features
                init_args: The arguments to be passed to the class constructor besides
                            in_features and out_features
            mask:
                dim: The dimension of the mask
                mask_function: The function that is used for determining the mask
                                it follows the standard dypy FunctionDescriptor format.
                flip: Whether to flip the mask or not
        """
        
        # call the super init method for nn.Module
        super().__init__()
        
        # make sure that the arguments pertaining to the net creation are okay
        if 'class_path' not in net:
            raise ValueError("net must have a class_path")
        if 'init_args' not in net:
            raise ValueError("net must have init_args")
        if 'activation' in net['init_args'] and isinstance(net['init_args']['activation'], str):
            net['init_args']['activation'] = dy.eval(net['init_args']['activation'])
                
        # define a transform_net_create_fn
        def create_net(in_features, out_features):
            return dy.eval(net['class_path'])(in_features, out_features, **net['init_args'])
                
        real_mask = torch.ones(mask['dim'])
        if 'mask_function' in mask:
            if isinstance(mask['mask_function'], str):
                mask_function = dy.eval(mask['mask_function'])
            elif isinstance(mask['mask_function'], dict):
                mask_function = dy.eval_function(**mask['mask_function'])
        else:
            mask_function = lambda ind: ind % 2 == 0
        
        for i in range(mask['dim']):
            real_mask[i] = real_mask[i] * -1 if mask_function(i) else real_mask[i]
        
        if 'flip' in mask and mask['flip']:
            real_mask = real_mask * -1
        
        # set the wrapping value
        self._value = dy.eval(coupling_transform_cls)(
            mask=real_mask,
            transform_net_create_fn=create_net,
            **kwargs,
        )
        
        
    
    def forward(self, inputs, context=None):
        """Replace the wrapper forward method with the actual forward method of the coupling transform"""
        return self._value.forward(inputs, context=context)

    def inverse(self, inputs, context=None):
        """Replace the wrapper inverse method with the actual inverse method of the coupling transform"""
        return self._value.inverse(inputs, context=context)
        

class LogitTransform(Transform):
    """Base class for all transform objects."""
    def __init__(self, alpha=0.05, eps=1e-6):
        super().__init__()
        
        if alpha < eps:
            raise ValueError("alpha must be greater than eps")
        
        self.alpha = alpha
        self.eps = eps
        
    def log(self, x: torch.Tensor):
        """Log function with a small offset for numerical stability."""
        return torch.log(x + self.eps)
    
    def logit(self, x: torch.Tensor):
        """Logit function with a small offset for numerical stability."""
        return torch.log(x / (1 - x + self.eps))
    
    def forward(self, inputs, context=None):
        # get all the non-batch dimensions
        dims = list(range(1, len(inputs.shape)))
        
        pre_logit = self.alpha + (1 - self.alpha) * inputs
        
        logdets = torch.sum(math.log(1 - self.alpha) - self.log(1 - pre_logit) - self.log(pre_logit), dim=dims)
        
        
        return self.logit(pre_logit), logdets
        
    def inverse(self, inputs, context=None):
        # get all the non-batch dimensions
        dims = list(range(1, len(inputs.shape)))
        
        sigm = torch.sigmoid(inputs)
        
        return sigm, torch.sum(self.log(1 - sigm) + self.log(sigm) - math.log(1 - self.alpha), dim=dims)
    
class NormalizingFlow(DensityEstimator):
    """
    A wrapper class for nflows Normalizing Flows.
    
    It is compeletely generic in the sense that you can define any type of flow
    using a primitive configurations file.
    """
    
    model_type = "nf"

    def __init__(self, 
                 dim: th.Optional[int] = None, 
                 transform : th.Union[
                     Transform, 
                     th.List[Transform], 
                     th.Dict[str, th.Any],
                     th.List[th.Dict[str, th.Any]]
                 ] = None, 
                 base_distribution: th.Optional[th.Union[Distribution, th.Dict[str, th.Any]]]=None, 
                 **kwargs):
        super().__init__(**kwargs)
        
        
        
        if isinstance(transform, list):
            all_transforms = [x for x in transform if x is not None]
        else:
            all_transforms = [transform]
        
        is_multiscale = []
        multiscale_args = []
        any_multiscale = False
        
        actual_all_transforms = []
        for i in range(len(all_transforms)):
            # check if all_transforms[i] is a repeat block or not
            if 'repeat_n' in all_transforms[i] and 'content' in all_transforms[i]:
                for _ in range(all_transforms[i]['repeat_n']):
                    # copy everything in all_transforms[i]['content']
                    if not isinstance(all_transforms[i]['content'], list):
                        all_transforms[i]['content'] = [all_transforms[i]['content']]
                    for content in all_transforms[i]['content']:
                        all_transforms_i_copy = copy.deepcopy(content)
                        actual_all_transforms.append(all_transforms_i_copy)
            else:
                actual_all_transforms.append(all_transforms[i])
                
        all_transforms = actual_all_transforms
        
        for i in range(len(all_transforms)):
            if isinstance(all_transforms[i], dict):
               
                transform_cls = dy.eval(all_transforms[i]['class_path'])
                transform_args = all_transforms[i]['init_args']
                
                if 'multiscale' in all_transforms[i]:
                    is_multiscale.append(True)
                    multiscale_args.append(all_transforms[i]['multiscale'])
                    any_multiscale = True
                else:
                    is_multiscale.append(False)
                    multiscale_args.append({})
                    
                all_transforms[i] = transform_cls(**transform_args)
                
                self.add_representation_module(all_transforms[i], rank=i)
        
        if isinstance(transform, list):
            if any_multiscale:
                # count the number of True in is_multiscale
                num_multiscale = sum(is_multiscale)
                self.transform = MultiscaleCompositeTransform(num_transforms=num_multiscale + (not is_multiscale[-1]))
                current_transform = []
                last_shape = None
                for i in range(len(all_transforms)):    
                    current_transform.append(all_transforms[i])
                    if i == len(all_transforms) - 1 or is_multiscale[i]:
                        composite_transform = CompositeTransform(current_transform)
                        
                        # Add the output of the multiscale transform as a representation module with rank 1000 + i
                        self.add_representation_module(composite_transform, rank=1000 + i)
                        
                        if 'transform_output_shape' in multiscale_args[i]:
                            last_shape = self.transform.add_transform(composite_transform, 
                                                                      **multiscale_args[i])
                        else:
                            last_shape = self.transform.add_transform(composite_transform, 
                                                                     transform_output_shape=last_shape,
                                                                     **multiscale_args[i])
                        current_transform = []       
            else:
                self.transform = CompositeTransform(all_transforms)
        else:
            self.transform = all_transforms[0]
        
        # Setup the base_distribution, set to normal if not specified
        if base_distribution is None:
            self.base_distribution = StandardNormal([dim])
        elif isinstance(base_distribution, dict):
            self.base_distribution = dy.eval(base_distribution['class_path'])(**base_distribution['init_args'])
        elif isinstance(base_distribution, Distribution):
            self.base_distribution = base_distribution

        # if logit transform is true, then add a logit transform to the beginning of the flow
        if self.logit_transform:
            self.transform = CompositeTransform([LogitTransform(), self.transform])
            self.logit_transform = False
            
        # Setup the actual flow underneath
        self._nflow = Flow(
            transform=self.transform,
            distribution=self.base_distribution
        )
    
    def sample(self, n_samples):
        if n_samples == -1:
            with torch.no_grad():
                single_sample = self.sample(1)
                latent_sample = self._nflow.transform_to_noise(single_sample)
                z_0 = torch.zeros_like(latent_sample)
                real_sample, _ = self._nflow._transform.inverse(z_0)
            return real_sample
        else:
            # TODO: batch in parent class
            samples = self._nflow.sample(n_samples)
            return self._inverse_data_transform(samples)

    @batch_or_dataloader()
    def log_prob(self, x):
            
        # NOTE: Careful with log probability when using _data_transform()
        x = self._data_transform(x)
        
        log_prob = self._nflow.log_prob(x)
                        
        if len(log_prob.shape) == 1:
            log_prob = log_prob.unsqueeze(1)

        return log_prob
