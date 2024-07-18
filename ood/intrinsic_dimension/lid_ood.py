"""
Our main method in the paper, this method performs LID computation for all the datapoints
and plots a scatterplot of LID and likelihoods for all the datapoints that were considered
for the OOD datapoints.

Finally, these scatterplots can be later used for evaluation and inference.
"""
from ..base import OODBaseMethod
import torch
from typing import Optional, Callable, Literal
from lid.base import ModelBasedLID
import time
from ..utils import buffer_loader
import os
import json
import numpy as np
from ..wandb_visualization import visualize_scatterplots
import dypy as dy
from tqdm import tqdm
from lid.base import ScaleType

class LID_OOD(OODBaseMethod):
    def __init__(
        self,
        likelihood_model: torch.nn.Module,
        # the LID calculator that takes in the likelihood model and constructs itself
        lid_calculator_class: str,
        # likelihood computation args
        likelihood_computation_args: Optional[dict] = None,
        lid_calculator_args: Optional[dict] = None,
        x_loader: Optional[torch.utils.data.DataLoader] = None,
        in_distr_loader: Optional[torch.utils.data.DataLoader] = None,
        # verbosity
        verbose: int = 0,
        # checkpointing args for longer runs
        checkpoint_dir: Optional[str] = None,
        checkpointing_buffer: Optional[int] = None,
        # Hyper-parameters relating to the scale parameter that is being computed
        scale_selection_algorithm: Literal['find_plateau', 'fixed', 'given_from_model_free'] = 'fixed',
        scale_selection_algorithm_args: Optional[dict] = None,
        training_buffer_size: int = 16,    
    ) -> None:
        """
        Args:
            likelihood_model: The likelihood generative model to be used for OOD detection.
            
            x_loader: The loader for the OOD data.
            
            in_distr_loader: The loader for the in-distribution data.
            
            likelihood_computation_args:
                The arguemnts that are given to the model.log_prob function, defaults to nothing!
                
            checkpoint_dir: 
                The OOD method stores LID estimates and likelihood values.
                Sometimes the OOD dataloader is huge, therefore, the method might need to checkpoint
                the LID estimates and likelihood values to the disk. This is the directory where the
                checkpointing will be done.
                
            checkpointing_buffer: 
                This is a buffer size of LID and likelihood values to be stored before checkpointing.
                When the buffer is full, the checkpointing will happen, otherwise, a lot of time
                will be spent for the algorithm to move every datapoint's estimated values to the disk.
                
            lid_calculator: 
                The constructor function of a LID calculator (you can use a partial on the LID class and pass it here)
                this class has an estimate_lid method that takes in the datapoints and performs model-based LID computation.
                
            scale_selection_algorithm: 
                The algorithm to select the scale parameter. All of these algorithms (except the fixed one) use a subset of 
                the training data to represent the in-distribution data, and then play around with the scale parameter
                such that training data has a particular LID estimte.
                
                1. 'find_plateau': This algorithm will plot the LID estimates of the subsampled training data and then
                    find a scale in which a large plateau occures in the LID estimates.
                
                2. 'fixed': This algorithm will use the base_scale as the scale parameter for the LID computation.
                
                3. 'given_from_model_free': This algorithm will use the model_free_dimension, which might come from
                    another intrinsic dimension algorithm such as LPCA, then it will choose the scale parameter
                    such that the LID estimates of the training data is almost equal to the model_free_dimension.
            
            training_buffer_size: 
                The number of batches from the in_distr_dataloader to be considered as training data representatives.
        """
        super().__init__(
            likelihood_model = likelihood_model,
            x_loader=x_loader,
            in_distr_loader=in_distr_loader,
        )
        self.verbose = verbose
        
        # construct the LID calculator model and call its fit function
        self.lid_calculator: ModelBasedLID = dy.eval(lid_calculator_class)(model=likelihood_model, **(lid_calculator_args or {}))
        self.lid_calculator.fit()
        
        # No need to train the model anymore, so turn off all the gradients
        # and set it to evaluation mode for faster computation
        self.likelihood_model.eval()
        for param in self.likelihood_model.parameters():
            param.requires_grad = False
        
        # set the scale selection parameters
        self.scale_selection_algorithm = scale_selection_algorithm
        self.model_selection_algorithm_args = scale_selection_algorithm_args or {}
        self.training_buffer_size = training_buffer_size
        
        # set the checkpointing parameters
        self.checkpoint_dir = checkpoint_dir
        self.checkpointing_buffer = checkpointing_buffer
        
        self.likelihood_computation_args = likelihood_computation_args or {}
    
    def select_scale_fixed(
        self,
        training_data_loader: torch.utils.data.DataLoader,
        base_scale: ScaleType,
    ):
        return base_scale
    
    def select_scale_find_plateau(
        self,
        training_data_loader: torch.utils.data.DataLoader,
        num_bins: int = 100,
        l_scale: float = -20,
        r_scale: float = 20,
        tolerance: float = 1.0,
        eps: float = 1e-6,
    ) -> float:
        """
        This method selects the scale parameter by finding a plateau in the LID estimates
        of the training data.
        
        Complexity: O(lid_buffering_time + num_bins * (log(num_bins) + lid_computation_time))
        
        Args:
            training_data_loader: 
                The training data loader to be used for scale selection.
            num_bins: 
                The number of sampled scales to be used for searching the Plateau.
            l_scale: 
                The smallest scale to consider.
                This should typically give a very large LID estimate for all the points in the training data, such as the ambient dimension.
            r_scale: 
                The largest scale to consider.
                This should typically give a very small LID estimate for all the points in the training data, such as 0.
            tolerance: 
                The tolerance to be used for the plateau detection. If the difference between the maximum and minimum
                LID estimates in a contiguous interval is less than this tolerance, then it is considered a plateau.
            eps: 
                A small number to be used for the plateau detection. If the maximum and minimum LID estimates in a contiguous
                interval are less than this number, then it is considered a plateau.
        Returns:
            The scale parameter that is to be used for LID estimation.
        """
        self.lid_calculator.buffer_data(training_data_loader)
        
        lid_trend = []
        search_space = np.linspace(l_scale, r_scale, num_bins)
        for scale in search_space:
            all_lid = self.lid_calculator.compute_lid_buffer(scale)
            mean_lid = np.mean(np.concatenate([x.cpu().numpy().flatten() for x in all_lid]))
            lid_trend.append(mean_lid)
        ambient_dim = self.lid_calculator.ambient_dim
        
        # find the largest contiguous interval in ambient_dim which is not equal to ambient_dim or zero
        # this is the plateau. Use a two pointer approach to find the largest contiguous interval
        lid_set = set()
        best_scale = None
        best_interval_length = 0
        pnt = 0
        for i in range(len(lid_trend)):
            while True:
                # get the min of lid_set
                current_min = min(lid_set) if len(lid_set) > 0 else lid_trend[i]
                current_max = max(lid_set) if len(lid_set) > 0 else lid_trend[i]
                current_min = min(current_min, lid_trend[i])
                current_max = max(current_max, lid_trend[i])
                if current_max - current_min < tolerance:
                    pnt += 1
                    lid_set.add(lid_trend[i])
                else:
                    break
            if len(lid_set) > best_interval_length and max(lid_set) < ambient_dim - eps and min(lid_set) > eps:
                best_interval_length = len(lid_set)
                best_scale = search_space[(pnt + i) // 2]
            
            # remove the element 'i' for the next step
            lid_set.remove(lid_trend[i])
        
        return best_scale
    
    
    def select_scale_given_from_model_free(
        self,
        training_data_loader: torch.utils.data.DataLoader,
        model_free_dimension: float,
        l_scale: float = -20,
        r_scale: float = 20,
        bin_search_steps: int = 20,
    ):
        """
        This method selects the scale parameter by finding a scale parameter
        such that the LID estimates of the training data is almost equal to the model_free_dimension.
        
        Complexity: O(lid_buffering_time + bin_search_steps * lid_computation_time)
        
        Args:
            training_data_loader: 
                The training data loader to be used for scale selection.
            l_scale: 
                The smallest scale to consider.
                This should typically give a very large LID estimate for all the points in the training data, such as the ambient dimension.
            r_scale: 
                The largest scale to consider.
                This should typically give a very small LID estimate for all the points in the training data, such as 0.
            model_free_dimension:
                The dimensionality of the data as estimated by a model-free dimensionality estimator.
            bin_search_steps:
                The number of steps to be used for the binary search, the binary search finds a scale parameter that
                produces an average LID estimate that is almost equal to the model_free_dimension.
        Returns:
            The scale parameter that is to be used for LID estimation.
            
        """
        self.lid_calculator.buffer_data(training_data_loader)
        
        # binary search to find the scale parameter
        l = l_scale
        r = r_scale
        bin_search_rng = range(bin_search_steps) if self.verbose == 0 else tqdm(range(bin_search_steps), desc="Binary search to find the best scale")
        for _ in bin_search_rng:
            mid = (l + r) / 2
            all_lid = self.lid_calculator.compute_lid_buffer(mid)
            mean_lid = np.mean(np.concatenate([x.cpu().numpy().flatten() for x in all_lid]))
            if mean_lid < model_free_dimension:
                r = mid
            else:
                l = mid
        return (l + r) / 2
    
    def run(self):
        """
        The main method to run for OOD detection which might log important information on W&B.
        
        The way this function works is that it first subsamples a small amount of the training data
        for scale selection. When that is done, it uses the same scale to log all the LID estimates 
        and likelihood values per datapoint in self.x_loader.
        """
        buffer = buffer_loader(self.in_distr_loader, self.training_buffer_size, limit=1)
        for _ in buffer:
            inner_loader = _
            break
        
        progress_dict = {}
        if self.checkpoint_dir is not None and os.path.exists(os.path.join(self.checkpoint_dir, 'progress.json')):
            with open(os.path.join(self.checkpoint_dir, 'progress.json'), 'r') as file:
                progress_dict = json.load(file)

        # Select the scale parameter using the scale selection algorithm
        # if it is already computed, it is stored in the progress_dict
        # which was written to the disk, otherwise, compute it and then store
        # it in the progress_dict and write it to the disk.
        if 'chosen_scale' in progress_dict:
            chosen_scale = progress_dict['chosen_scale']
        elif self.scale_selection_algorithm == 'fixed':
            chosen_scale = self.select_scale_fixed(inner_loader)
        elif self.scale_selection_algorithm == 'find_plateau':
            chosen_scale = self.select_scale_find_plateau(inner_loader, **self.model_selection_algorithm_args)
        elif self.scale_selection_algorithm == 'given_from_model_free':
            chosen_scale = self.select_scale_given_from_model_free(inner_loader, **self.model_selection_algorithm_args)
        else:
            raise ValueError(f"Invalid scale selection algorithm: {self.scale_selection_algorithm}")
        progress_dict['chosen_scale'] = chosen_scale
        with open(os.path.join(self.checkpoint_dir, 'progress.json'), 'w') as file:
            json.dump(progress_dict, file)
        
        # log the scale parameter if verbose
        if self.verbose > 0:
            print("running with scale:", chosen_scale)
        
        
        # All dimensionalities and all likelihoods update
        all_dimensionalities = None
        all_likelihoods = None
        if os.path.exists(os.path.join(self.checkpoint_dir, 'all_likelihoods.npy')):
            all_likelihoods = np.load(os.path.join(self.checkpoint_dir, 'all_likelihoods.npy'), allow_pickle=True)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'all_dimensionalities.npy')):
            all_dimensionalities = np.load(os.path.join(self.checkpoint_dir, 'all_dimensionalities.npy'), allow_pickle=True)
        
        buffer_progress = 0
        for inner_loader in buffer_loader(self.x_loader, self.checkpointing_buffer):
            buffer_progress += 1
            if 'buffer_progress' in progress_dict:
                if buffer_progress <= progress_dict['buffer_progress']:
                    continue
            if self.verbose > 0:
                print(f"Working with buffer [{buffer_progress}]")
            
           
            if self.verbose > 0:
                print("Computing dimensionalities ... ", end='')
            # compute dimensionalities
            lid_estimates = self.lid_calculator.estimate_lid(inner_loader, scale=chosen_scale)
            all_buffer_dimensionalities = np.concatenate([x.cpu().numpy().flatten() for x in lid_estimates])
            if self.verbose > 0:
                print("done!")
            
            # compute and add likelihoods
            if self.verbose > 0:
                print("Computing likelihoods ... ", end='')
            all_buffer_likelihoods = None
            for x in inner_loader:
                with torch.no_grad():
                    likelihoods = self.likelihood_model.log_prob(x, **self.likelihood_computation_args).cpu().numpy().flatten()
                    all_buffer_likelihoods = np.concatenate([all_buffer_likelihoods, likelihoods]) if all_buffer_likelihoods is not None else likelihoods
            if self.verbose > 0:
                print("done!")
            
            # add the likelihood and dimensionalities obtained here to all the estimates
            all_dimensionalities = np.concatenate([all_dimensionalities, all_buffer_dimensionalities]) if all_dimensionalities is not None else all_buffer_dimensionalities
            all_likelihoods = np.concatenate([all_likelihoods, all_buffer_likelihoods]) if all_likelihoods is not None else all_buffer_likelihoods
            
            progress_dict['buffer_progress'] = buffer_progress
            
            np.save(os.path.join(self.checkpoint_dir, 'all_likelihoods.npy'), all_likelihoods)
            np.save(os.path.join(self.checkpoint_dir, 'all_dimensionalities.npy'), all_dimensionalities)
            
            with open(os.path.join(self.checkpoint_dir, 'progress.json'), 'w') as file:
                json.dump(progress_dict, file)
            
            
        # wandb logging
        visualize_scatterplots(
            scores = np.stack([all_likelihoods, all_dimensionalities]).T,
            column_names=["log-likelihood", "LID"],
        )