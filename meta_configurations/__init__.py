"""
This code are the codes used for creating the dysweep and operate on a meta_configuration level to logging onto W&B.

No need to understand the code here as it performs some configuration-level process to change the names of the W&B runs that we do not need to understand.
"""
import typing as th
from pprint import pprint

def get_coupling(conf):
    """
    This function takes in a configuration and returns the coupling transform
    """
    if isinstance(conf, list):
        for c in conf:
            ret = get_coupling(c)
            if ret is not None:
                return ret
        return None
    
    if isinstance(conf, dict):
        for k, v in conf.items():
            if k == 'coupling_transform_cls':
                return v.split('.')[-1]
            else:
                ret = get_coupling(v)
                if ret is not None:
                    return ret
        return None
    
    return None

def change_name(conf, run_name):
    if conf['trainer']['writer']['tag_group'] == 'flow':
        coupling_type = get_coupling(conf)
        return f"{coupling_type}_{conf['data']['dataset']}_{run_name}"
    elif conf['trainer']['writer']['tag_group'] == 'diffusion':
        model_class = conf['model']['class_path'].split('.')[-1]
        return f"diffusion_{model_class}_{conf['data']['dataset']}_{run_name}"
    else:
        return f"{conf['data']['dataset']}_{run_name}"
  
def change_coupling_layers(conf, coupling_name: str, additional_args: th.Optional[dict] = None):
    """
    This function takes in an entire configuration
    and replaces all the coupling transforms to the one specified.
    If the coupling transform is in need of additional arguments,
    this function will add them to the configuration.
    """
    if isinstance(conf, list):
        ret = []
        for c in conf:
            ret.append(change_coupling_layers(c, coupling_name=coupling_name, additional_args=additional_args))
        return ret
    
    if isinstance(conf, dict):
        ret = {}
        for k, v in conf.items():
            if k == 'coupling_transform_cls':
                ret = {}
                for k_, v_ in conf.items():
                    if k_ in ['net', 'mask']:
                        ret[k_] = v_
                ret['coupling_transform_cls'] = f'nflows.transforms.{coupling_name}'
                if additional_args:
                    ret.update(additional_args)
                return ret
            else:   
                ret[k] = change_coupling_layers(v, coupling_name=coupling_name, additional_args=additional_args)
        return ret
    
    return conf

def get_generated_config(conf: dict):
    """
    Returns the configurations required for the model generated datasets.
    """
    return {
        'dataset': 'dgm-generated',
        'dgm_args': {
            'model_loading_config': conf['base_model'],
            'seed': 10,
            'length': 1000,
        }
    }

def ood_run_name_changer(conf, run_name):
    """
    This run changer takes a look at the configuration and sets a name for that
    run appropriately.
    The scheme here is [trained_dataset]_vs_[ood_dataset]_[test_or_train_split]
    """
    first = conf['data']['in_distribution']['dataloader_args']['dataset']
    if first == 'medmnist': # TODO: fix this!
        first = f"{conf['data']['in_distribution']['dataloader_args']['additional_dataset_args']['subclass']}-{'align' if conf['data']['in_distribution']['dataloader_args']['additional_dataset_args'].get('align', True) else 'noalign'}"
    ret = first
    ret += '_vs_'
    second = conf['data']['out_of_distribution']['dataloader_args']['dataset']
    if second == 'medmnist': # TODO: fix this!
        second = f"{conf['data']['out_of_distribution']['dataloader_args']['additional_dataset_args']['subclass']}-{'align' if conf['data']['out_of_distribution']['dataloader_args']['additional_dataset_args'].get('align', True) else 'noalign'}"
    ret += second
    ret += f"_{conf['data']['out_of_distribution']['pick_loader']}"
    ret += f"_{run_name}"
    return ret

# NOTE: obsolete!
def extras_run_name_changer(conf, run_name):
    """
    This run changer takes a look at the configuration and sets a name for that
    run appropriately.
    The scheme here is [trained_dataset]_vs_[ood_dataset]_[test_or_train_split]
    """
    ret = conf['data']['in_distribution']['dataloader_args']['dataset']
    ret += '_vs_'
    ret += conf['data']['out_of_distribution']['dataloader_args']['dataset']
    ret += f"_{conf['data']['out_of_distribution']['pick_loader']}"
    ret += f"_num_samples_{conf['ood']['method_args']['log_prob_kwargs']['trace_calculation_kwargs']['sample_count']}"
    ret += f"_num_steps_{conf['ood']['method_args']['log_prob_kwargs']['steps']}"
    ret += f"_{run_name}"
    return ret

def intrinsic_dimension_run_name_changer(conf, run_name):
    """
    Change the run appropriately for the dataset under consideration
    """
    ret = conf['data']['dataset']
    if 'additional_dataset_args' in conf['data']:
        map = conf['data']['additional_dataset_args']
        for key in map:
            if isinstance(map[key], int) or isinstance(map[key], str):
                ret += f'_{key}_{map[key]}'
    ret += f"_{run_name}"
    return ret

def recursive_replace_with(conf, trigger, replacement):
    """
    Change the run appropriately for the dataset under consideration
    """
    if isinstance(conf, dict):
        for key in conf.keys():
            if conf[key] == trigger:
                conf[key] = replacement
            elif isinstance(conf, dict) or isinstance(conf, list):
                conf[key] = recursive_replace_with(conf[key], trigger, replacement)
    elif isinstance(conf, list):
        for i, val in enumerate(conf):
            if val == trigger:
                conf[i] = replacement
            elif isinstance(val, dict) or isinstance(val, list):
                conf[i] = recursive_replace_with(conf[i], trigger, replacement)
    return conf
