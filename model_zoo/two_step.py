# This code is not used at all and is an artifact from the two_step_zoo repo.
# However, I have kept it here for possible future use.

from abc import abstractmethod
import numpy as np
import torch
import torch.nn as nn

from .utils import batch_or_dataloader
import dypy as dy
from dysweep import dysweep_run_resume
import typing as th

class TwoStepDensityEstimator(nn.Module):

    def __init__(self, generalized_autoencoder, density_estimator):
        super().__init__()
        self.generalized_autoencoder = generalized_autoencoder
        self.density_estimator = density_estimator
        self.model_type = ("two_step_"
                            f"{generalized_autoencoder.model_type}_"
                            f"{density_estimator.model_type}")

    def sample(self, n_samples):
        return self.generalized_autoencoder.decode(self.density_estimator.sample(n_samples))

    @batch_or_dataloader()
    def low_dim_log_prob(self, x):
        with torch.no_grad():
            encodings = self.generalized_autoencoder.encode(x)
            return self.density_estimator.log_prob(encodings)

    @batch_or_dataloader()
    def log_prob(self, x):
        with torch.no_grad():
            low_dim_log_prob = self.low_dim_log_prob(x)
            log_det_jtj = self.generalized_autoencoder.log_det_jtj(x)
            return low_dim_log_prob - 0.5*log_det_jtj

    def rec_error(self, *args, **kwargs):
        return self.generalized_autoencoder.rec_error(*args, **kwargs)

    @property
    def device(self):
        return self.generalized_autoencoder.device

    def __getattribute__(self, attr):
        """Redirect other attributes to GAE"""
        gae_attributes = ("encode", "encode_transformed", "decode", "decode_to_transformed",
                          "encoder", "decoder", "rec_error", "latent_dim", "data_min", "data_max",
                          "data_shape")

        if attr in gae_attributes:
            return getattr(self.generalized_autoencoder, attr)
        else:
            return super().__getattribute__(attr)

def mask_and_replace(x, rate, 
                     unique_values: th.Optional[torch.Tensor] = None):
    """
    This implements the background perturbations introduced in Ren et al.
    https://arxiv.org/pdf/1906.02845.pdf
    
    Mask elements in tensor x with probability 'rate' and then replace them with a random value 
    from the unique values in the tensor.

    Args:
        x (torch.Tensor): input tensor.
        rate (float): probability of masking each element in the tensor.

    Returns:
        torch.Tensor: tensor after masking and replacement.
    """
    
    # Create a binary mask of the same shape as x with 1's at the locations to be masked
    mask = (torch.rand(x.shape) < rate).float().to(x.device)

    if unique_values is None:
        # Get unique values from x (no gradients)
        unique_values = x.detach().unique()
    unique_values = unique_values.to(x.device)
    
    # For each masked location, select a random value from unique_vals for replacement
    rand_indices = torch.randint(0, len(unique_values), size=x.shape)
    replacements = unique_values[rand_indices]

    # Use the mask to combine the original tensor with replacements
    result = x * (1 - mask) + replacements * mask
    
    return result

class TwoStepComponent(nn.Module):
    """Superclass for the GeneralizedAutoencoder and DensityEstimator"""
    _OPTIMIZER_MAP = {
        "sgd": torch.optim.SGD,
        "adam": torch.optim.Adam,
        "rmsprop": torch.optim.RMSprop,
    }
    _MIN_WHITEN_STDDEV = 1e-6

    def __init__(
            self,
            flatten=False,
            data_shape=None,
            denoising_sigma=None,
            dequantize=False,
            scale_data=False,
            scale_data_to_n1_1=False,
            whitening_transform=False,
            background_augmentation: th.Optional[float] = None,
            logit_transform=False,
            clamp_samples=False,
        ):
        super().__init__()

        assert not (scale_data and whitening_transform), \
            "Cannot use both a scaling and a whitening transform"
        assert not (scale_data and scale_data_to_n1_1), \
            "Cannot use both a scaling and a scaling to (-1,1) transform"
        assert not (scale_data_to_n1_1 and whitening_transform), \
            "Cannot use both a scaling to (-1,1) and a whitening transform"


        self.flatten = flatten
        self.data_shape = data_shape
        self.denoising_sigma = denoising_sigma
        self.dequantize = dequantize
        self.scale_data = scale_data
        self.scale_data_to_n1_1 = scale_data_to_n1_1
        self.whitening_transform = whitening_transform
        self.logit_transform = logit_transform
        self.background_augmentation = background_augmentation
        self.clamp_samples = clamp_samples

        # NOTE: Need to set buffers to specific amounts or else they will not be loaded by state_dict
        self.register_buffer("data_min", torch.tensor(0.))
        self.register_buffer("data_max", torch.tensor(1.))

        if whitening_transform:
            whiten_dims = self._get_whiten_dims()

            self.register_buffer("whitening_sigma", torch.ones(*whiten_dims))
            self.register_buffer("whitening_mu", torch.zeros(*whiten_dims))

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Required attribute indicating general model type; e.g., 'vae' or 'nf'"""
        pass

    @property
    def device(self):
        # Consider the model's device to be that of its first parameter
        # (there's no great way to define the `device` of a whole model)
        first_param = next(self.parameters(), None)
        if first_param is not None:
            return first_param.device
        else:
            return None

    def loss(self, x, **kwargs):
        raise NotImplementedError(
            "Implement loss function in child class"
        )

    def train_batch(self, x, max_grad_norm=None, trainer=None, **kwargs):
        self.optimizer.zero_grad()
        
        loss = self.loss(x, **kwargs)
        loss.backward()

        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)

        self.optimizer.step()
        
        if trainer is not None:
            # look for an edge in the trainer's epoch and if an edge
            # occures call the step function of your lr_scheduler
            if not hasattr(self.optimizer, 'epoch'):
                self.optimizer.epoch = trainer.epoch
            
            if self.optimizer.epoch != trainer.epoch:
                self.lr_scheduler.step()
                self.optimizer.epoch = trainer.epoch
        else:
            self.lr_scheduler.step()
            
        return {
            "loss": loss
        }

    def set_optimizer(self, cfg):
        """
        cfg format:
        {
            'class_path': the class of the optimizer
            'init_args': the arguments to pass to the optimizer
            {
                'lr': learning rate
            }
            'lr_scheduler': if it is null or non-existant, then no lr scheduler is used
            {
                class_path: the class of the lr scheduler
                init_args: the arguments to pass to the lr scheduler
            }
        }
        """
        self.optimizer: torch.optim.Optimizer = dy.eval(cfg['class_path'])(
            self.parameters(), **cfg['init_args']
        )
        self.num_optimizers = 1

        if 'lr_scheduler' not in cfg:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=self.optimizer,
                lr_lambda=lambda step: 1.
            )
        else:
            init_args = {}
            if 'init_args' in cfg['lr_scheduler'] and cfg['lr_scheduler']['init_args'] is not None:
                init_args = cfg['lr_scheduler']['init_args']
            self.lr_scheduler: torch.optim.lr_scheduler.LRScheduler = dy.eval(cfg['lr_scheduler']['class_path'])(
                optimizer=self.optimizer, **init_args)
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                raise NotImplementedError("ReduceLROnPlateau is not integrated yet!")
        
    def set_whitening_params(self, mu, sigma):
        self.whitening_mu = torch.reshape(mu, self._get_whiten_dims())
        self.whitening_sigma = torch.reshape(sigma, self._get_whiten_dims())
        self.whitening_sigma[self.whitening_sigma < self._MIN_WHITEN_STDDEV] = self._MIN_WHITEN_STDDEV

    def _get_whiten_dims(self):
        if self.flatten:
            return (1, np.prod(self.data_shape))
        else:
            return (1, *self.data_shape)

    def _data_transform(self, data):
        if self.flatten:
            data = data.flatten(start_dim=1)
        if self.denoising_sigma is not None and self.training:
            data = data + torch.randn_like(data) * self.denoising_sigma
        if self.dequantize:
            data = data + torch.rand_like(data)
        if self.scale_data:
            data = data / (self.abs_data_max + self.dequantize)
        elif self.scale_data_to_n1_1:
            data = 2. * (data - self.data_min) / (self.data_range + self.dequantize) - 1.
        elif self.whitening_transform:
            data = data - self.whitening_mu
            data = data / self.whitening_sigma
        if self.logit_transform:
            data = torch.logit(data)
        if self.background_augmentation is not None:
            rate = self.background_augmentation
            if data.max() <= 1.:
                data = mask_and_replace(data, rate, unique_values=torch.linspace(data.min(), data.max(), 256))
            else:
                data = mask_and_replace(data, rate, unique_values=torch.arange(256))
            
        return data

    def _inverse_data_transform(self, data):
        if self.logit_transform:
            data = torch.sigmoid(data)
        if self.scale_data:
            if self.dequantize:
                data = data * (self.abs_data_max + 1.0)
            else:
                data = data * self.abs_data_max
        elif self.scale_data_to_n1_1:
            data = 0.5 * (data + 1.) * (self.data_range + self.dequantize) + self.data_min
        elif self.whitening_transform:
            data = data * self.whitening_sigma
            data = data + self.whitening_mu
        if self.dequantize:
            data = torch.floor(data)
        if self.clamp_samples:
            data.clamp_(self.data_min, self.data_max)
        if self.flatten:
            data = data.reshape((-1, *self.data_shape))
        return data

    @property
    def abs_data_max(self):
        return max(self.data_min.abs(), self.data_max.abs())

    @property
    def data_range(self):
        return self.data_max - self.data_min