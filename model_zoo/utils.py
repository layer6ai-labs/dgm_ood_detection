import torch
from torch.utils.data import DataLoader
import functools
import dypy as dy
import typing as th
import yaml
import json
import os
from pprint import pprint

def batch_or_dataloader(agg_func=torch.cat):
    def decorator(batch_fn):
        """
        Decorator for methods in which the first arg (after `self`) can either be
        a batch or a dataloader.

        The method should be coded for batch inputs. When called, the decorator will automatically
        determine whether the first input is a batch or dataloader and apply the method accordingly.
        """
        @functools.wraps(batch_fn)
        def batch_fn_wrapper(ref, batch_or_dataloader, **kwargs):
            if isinstance(batch_or_dataloader, DataLoader): # Input is a dataloader
                list_out = [batch_fn(ref, batch, **kwargs)
                            for batch, _, _ in batch_or_dataloader]

                if list_out and type(list_out[0]) in (list, tuple):
                    # Each member of list_out is a tuple/list; re-zip them and output a tuple
                    return tuple(agg_func(out) for out in zip(*list_out))
                else:
                    # Output is not a tuple
                    return agg_func(list_out)

            else: # Input is a batch
                return batch_fn(ref, batch_or_dataloader, **kwargs)

        return batch_fn_wrapper

    return decorator

def load_model_with_checkpoints(
    config: dict,
    device: th.Optional[str] = None,
    eval_mode: bool = True,
    root: str ='.',
):
    """
    This is a function that takes in a loading configuration and then returns a model
    with a loaded dictionary. It is used for loading any checkpoint e.g. peforming
    OOD detection on pretrained models, performing checkpoint resuming at training time, etc.

    Args:
        config (dict): This is a dictionary that eventually contains **all the necessary variables for
            instantiating a model. 
        
        device (th.Optional[str], optional): The device we want the model to be.
        eval_mode (bool, optional): If set to TRUE, then the model.eval() function would be called.
        root (str, optional): This is the root to look for when checking the checkpoint_dir
        json_guide (th.Optional[th.List[str]], optional): _description_. Defaults to None.

    Example of a config dictionary:
        config_dir: <a path to a yaml that contains the instantiation of the model somewhere: ext.yaml>
        config_guide: ['path', 'to', 'model_config']
        class_path_proxy: sth
        init_args_proxy: sth_else
        device: 'cuda:0',
        eval_mode: True,
        root: 1
        
    config_guide (list): This is an optional parameter that guides the configuration. In many instances,
        the configuration that is given as input here might contain a lot of extra information about the
        run that trained that specific model. 
        config_guide is a list of strings that when you iterate on that you end up with the actual parameters
        required to instantiate a model.
    Now ext.yaml would look something like this:
        path:
            to:
                model_config:
                    arg1: val1
                    arg2: val2
                    .
                    .
                    .
        extra_args: ...
    """
    model_conf = None
    
            
    # load the model and load the corresponding checkpoint
    if 'config_dir' in config:
        filename = config['config_dir']
        extension = filename.split('.')[-1]
        
        if 'yaml' in extension or 'yml' in extension:
            with open(os.path.join(root, filename), 'r') as f:
                model_conf = yaml.load(f, Loader=yaml.FullLoader)
        elif 'json' in extension:
            # load the json file into a dictionary
            with open(os.path.join(root, filename), 'r') as f:
                model_conf = json.load(f)
        
        config_guide = config.get('config_guide', None)
        class_path_proxy = config.get('class_path_proxy', 'class_path')
        init_args_proxy = config.get('init_args_proxy', 'init_args')
        
        if config_guide is not None:
            for step in config_guide:
                if step not in model_conf:
                    raise ValueError(f"After following the guide, failed at [{step}]! Please check the guide and the config.")
                model_conf = model_conf[step]
                
        # if it is an entire training configuration, then it contains
        # model configuration as a child node:
        if 'model' in model_conf:
            model_conf = model_conf['model']
    elif 'model' in config:     
        # if the model is directly given in the yaml, then overwrite the model_conf
        model_conf = config['model']
        
    # if the config is still None, then raise an error   
    if model_conf is None:
        raise ValueError("model configuration should be either given in the yaml or in the config_dir")
    
    # Instantiate the model
    # change the device of the model to device
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = dy.eval(model_conf[class_path_proxy])(**model_conf[init_args_proxy])
    
    # load the model weights from the checkpoint
    if config['checkpoint_dir'] is not None:
        model.load_state_dict(torch.load(os.path.join(root, config['checkpoint_dir']), map_location='cpu')['module_state_dict'])
    
    if eval_mode:
        # set to evaluation mode to get rid of any randomness happening in the 
        # architecture such as dropout
        # also remove all the training stuff
        model.eval()
    else:
        raise NotImplementedError("load_training_info is not implemented yet")
        
    return model.to(device)