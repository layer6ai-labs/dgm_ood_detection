"""
An abstraction for all OOD detection methods.
They take in a loader to perform OOD detection and they are only
permitted to use the likelihood model and the training data loader.
"""
from abc import ABC, abstractmethod
import typing as th
import torch


class OODBaseMethod(ABC):
    def __init__(
        self,
        likelihood_model: torch.nn.Module,
        x_loader: th.Optional[torch.utils.data.DataLoader] = None,
        in_distr_loader: th.Optional[torch.utils.data.DataLoader] = None,
        checkpoint_dir: th.Optional[str] = None,
    ) -> None:
        """
        Args:
            likelihood_model: The likelihood generative model to be used for OOD detection.
            x_loader: The loader for the OOD data.
            in_distr_loader: The loader for the in-distribution data.
            checkpoint_dir: The directory to save the checkpoints to, if needed.
        """
        super().__init__()
        self.likelihood_model = likelihood_model
        self.x_loader = x_loader
        self.in_distr_loader = in_distr_loader
        self.checkpoint_dir = checkpoint_dir
        
    @abstractmethod
    def run(self):
        '''The main method to run for OOD detection which might log important information on W&B'''
        raise NotImplementedError("run method not implemented!")
    
