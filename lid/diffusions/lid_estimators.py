
from ..base import ModelBasedLID, ScaleType, LIDInputType
from einops import repeat   
import torch
from model_zoo.density_estimator.diffusions import ScoreBasedDiffusion
from typing import Optional, Iterable
from tqdm import tqdm
import math

class NormalBundleLIDEstimator(ModelBasedLID):
    '''
    The intrinsic dimension estimator described by Stanczuk et al. (2023).
    
    See the paper (specifically algorithm 1, as of version 5) for details:
    https://arxiv.org/abs/2212.12611.
    
    Please note: the paper assumes the diffusion model is variance-exploding.
    but we have adapted the code to work with variance preserving as well.
    We have also implemented a multi-scale version of it which takes in
    the scale parameter as an input instead of setting it automatically.
    '''
    
    def __init__(
        self,
        *args,
        noise_time: float = 1e-4, 
        num_scores: Optional[int] =None, 
        chunk_size: int = 128,
        verbose: int = 0,
        **kwargs,
        
    ):
        super().__init__(*args, **kwargs)
        # check if the type of the self.model is not a diffusion model, then raise an error
        if not isinstance(self.model, ScoreBasedDiffusion):
            raise ValueError("The model input to the constructor should be a ScoreBasedDiffusion model (from the model_zoo)")
        
        self.model: ScoreBasedDiffusion
        self.sde = self.model
        self.noise_time = noise_time
        # The number of samples required for estimating LID in each case
        self.num_scores = num_scores if num_scores is not None else 4 * self.ambient_dim
        self.chunk_size = chunk_size
        self.verbose = verbose
        
    def infer_dim(
        self, 
        x: torch.Tensor,):
        '''Infer the ambient dimension from a batch of data'''
        self.ambient_dim = x.shape[1:].numel()

    def buffer_data(
        self, 
        x: LIDInputType,
    ):
        # check if x is iterable or not and if so, store it somewhere for when returning the LID
        self.data_is_iterable = True
        if not isinstance(x, Iterable):
            self.buffered_x = [x]
            self.data_is_iterable = False
        else:
            self.buffered_x = x
        
        # Create a list of all the singular values for the score of each batch of data
        self.singular_vals = []
        buffered_x_wrapper = tqdm(self.buffered_x, desc="Buffering data for LID estimation") if self.verbose > 0 else self.buffered_x
        for x in buffered_x_wrapper:
            # if the model has data transform, perform it first
            if hasattr(self.model, '_data_transform'):
                x = self.model._data_transform(x)
                
            d = x.numel() // x.shape[0]
            batch_size = x.shape[0]
            x = x.to(self.device)
            
            # get the slightly noised out points and store them in x_eps
            if not isinstance(self.noise_time, torch.Tensor):
                self.noise_time = torch.tensor(self.noise_time)
            sigma2_t, sigma_t = self.model._get_sigma(self.noise_time)
            sigma2_t, sigma_t = sigma2_t.to(self.device).float(), sigma_t.to(self.device).float()
            noise = torch.randn((batch_size*self.num_scores, *x.shape[1:])).to(self.device) 
            x_repeated = repeat(x, 'b ... -> (b c) ...', c=self.num_scores)
            x_eps = self.model._get_center(x_repeated, self.noise_time) + sigma_t * noise
            
            scores = []
            progress = 0
            ex_eps_splitted = x_eps.split(self.chunk_size)
            for batch in ex_eps_splitted:
                if self.verbose > 0:
                    progress += 1
                    buffered_x_wrapper.set_description(f"Buffering data for LID estimation [{progress}/{len(ex_eps_splitted)}]")
                batch = batch.to(self.device)
                beta_eps = self.model._get_beta(self.noise_time)
                drift = self.model._get_drift(batch, self.noise_time).cpu()
                diffus = self.model.get_true_score(batch, self.noise_time).cpu() * beta_eps
                scores.append((drift - diffus).detach().cpu())
            # Get the singular values of the score to compute the normal space
            scores = torch.cat(scores)
            scores = scores.reshape((batch_size, self.num_scores, d))
            self.singular_vals.append(torch.linalg.svdvals(scores.to(self.device)).cpu())
            
    def compute_lid_buffer(
        self,
        scale: Optional[ScaleType] = None,
    ):
        # count the number of singular values that are more than the threshold
        threshold = math.exp(- 2 * scale)
        all_lids = []
        for singular_vals in self.singular_vals:
            singular_vals = singular_vals.to(self.device)
            
            if scale is None:
                normal_dim = (singular_vals[:,:-1] - singular_vals[:,1:]).argmax(dim=1) + 1
                all_lids.append((self.ambient_dim - normal_dim).cpu())
            else:
                all_lids.append((singular_vals < threshold).sum(dim=1).cpu())
            
        return all_lids if self.data_is_iterable else all_lids[0]
        
    
    