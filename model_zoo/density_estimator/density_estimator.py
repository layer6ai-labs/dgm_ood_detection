from ..two_step import TwoStepComponent
import typing as th
import functools

class DensityEstimator(TwoStepComponent):

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
        clamp_samples=False
        
    ):
        super().__init__(
            flatten,
            data_shape, 
            denoising_sigma, 
            dequantize, 
            scale_data, 
            scale_data_to_n1_1,
            whitening_transform, 
            background_augmentation, 
            logit_transform, 
            clamp_samples
        )
            
        self._mid_level_representations = []
        self._representation_modules = []
        self._rank_to_index = {}
        self._save_representation = False
    
    def get_all_ranks(self):
        return list(self._rank_to_index.keys())
    
    def toggle_save_representation(self, save: bool):
        self._save_representation = save
          
    def add_representation_module(self, module, rank: int = 0):
        # keep a list of sorted modules according to rank in self._representation_modules
        # and register forward hooks on them
        
        self._representation_modules.append((rank, module))
        
        self._representation_modules = sorted(self._representation_modules, key=lambda x: x[0])
        
        # find the index of the module in the sorted list
        index = 0
        for i in range(len(self._representation_modules)):
            if self._representation_modules[i][1] == module:
                index = i
                break
        self._rank_to_index[rank] = index
        
        # remove all the hooks from the modules
        if hasattr(self, 'handles'):
            for handle in self.handles:
                handle.remove()
        
        # define the following forward hook
        def forward_hook(module, args, output, rank):
            repr_output = self.representation_hook(module, args, output, rank)
            if self._save_representation:
                self._mid_level_representations.append(repr_output)
            return output
        self.handles = []
        # When registering the modules in order of their increasing
        # rank, we can be sure that the forward hooks will be called
        # in the same order as the modules are registered
        for module in self._representation_modules:
            self.handles.append(module[1].register_forward_hook(functools.partial(forward_hook, rank=module[0])))
    
    
    def representation_hook(self, module, args, output, module_rank):
        """
        This is a function that gets called on each of the hooked modules
        When registering the hook for every single module, we pass it into
        this function to get the representation output of the module.
        
        Here, this function simply returns the output of the module, in default.
        However, this function can be overridden to return a different representation.
        """
        return output
    
    def clear_representation(self):
        """
        Clears the buffer of representations.
        """
        self._mid_level_representations.clear()
    
    def get_representation_module(self, rank: int):
        """
        Returns the exact module that was registered with the given rank.
        """
        if rank not in self._rank_to_index:
            raise ValueError(f"Rank {rank} not found.")
        ind = self._rank_to_index[rank]
        return self._representation_modules[ind][1]
    
    def get_representation(self, rank: th.Optional[int] = None):
        """
        Either returns the set of all representations, or the representation
        at the given index.
        
        You can call this function after calling the forward pass of the model
        which adds all the representations to the buffer.
        
        Args:
            ind (th.Optional[int], optional): Defaults to None.

        Returns:
            The representation at the given index, or the set of all representations.
        """
        if len(self._mid_level_representations) == 0:
            raise RuntimeError("No representations found. Make sure you call the forward pass of the model before calling this function.")
        if not hasattr(self, '_saved_length'):
            self._saved_length = len(self._mid_level_representations)
        if len(self._mid_level_representations) != self._saved_length:
            raise RuntimeError("The number of representations has changed!\n"
                               "Make sure you call the clear_representations before calling this function.")
        if rank is not None:
            if rank not in self._rank_to_index:
                raise ValueError(f"Rank {rank} not found.")
            ind = self._rank_to_index[rank]
            return self._mid_level_representations[ind]
        return self._mid_level_representations
                 
    def sample(self, n_samples):
        """
        n_sample can also take value "-1" in which case we
        will return the sample with the most probability.
        """
        raise NotImplementedError("sample not implemented")

    def log_prob(self, x, **kwargs):
        raise NotImplementedError("log_prob not implemented")

    def loss(self, x, **kwargs): 
        # This is used when training
        # for training purposes, we don't want to save the representations
        # therefore, we toggle the save_representation flag to False
        self.toggle_save_representation(False)
        ret = -self.log_prob(x, **kwargs).mean()
        self.toggle_save_representation(True)
        return ret