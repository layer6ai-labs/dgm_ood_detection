"""
The main file used for OOD detection.
"""
import copy
from PIL import Image
import io
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import torch
import dypy as dy
from jsonargparse import ArgumentParser, ActionConfigFile
import wandb
from dataclasses import dataclass
from random_word import RandomWords
from model_zoo.datasets import get_loaders
import traceback
import typing as th
from model_zoo.utils import load_model_with_checkpoints
from dotenv import load_dotenv
import os
from tqdm import tqdm
from math import sqrt
import datetime

# Needed for log_prob
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


@dataclass
class OODConfig:
    base_model: dict
    data: dict
    ood: dict
    logger: dict


def plot_likelihood_ood_histogram(
    model: torch.nn.Module,
    data_loader_in: torch.utils.data.DataLoader,
    data_loader_out: torch.utils.data.DataLoader,
    limit: th.Optional[int] = None,
    log_prob_kwargs: th.Optional[dict] = None,
):
    """
    Run the model on the in-distribution and out-of-distribution data
    and then plot the histogram of the log likelihoods of the models to show
    the pathologies if it exists.
    
    Args:
        model (torch.nn.Module): The likelihood model that contains a log_prob method
        data_loader_in (torch.utils.data.DataLoader): A dataloader for the in-distribution data
        data_loader_out (torch.utils.data.DataLoader): A dataloader for the out-of-distribution data
        limit (int, optional): The limit of number of datapoints to consider for the histogram.
                            Defaults to None => no limit.
    """
    # create a function that returns a list of all the likelihoods when given
    # a dataloader
    model.eval()
    def list_all_scores(dloader: torch.utils.data.DataLoader, description: str):
        log_probs = []
        for x in tqdm(dloader, desc=f"Calculating likelihoods for {description}"):
            with torch.no_grad():
                t = model.log_prob(x, **(log_prob_kwargs or {})).cpu()
            # turn t into a list of floats
            t = t.flatten()
            t = t.tolist()
            log_probs += t
            if limit is not None and len(log_probs) > limit:
                break
        return log_probs

    # List the likelihoods for both dataloaders
    in_distr_scores = list_all_scores(data_loader_in, "in distribution")
    out_distr_scores = list_all_scores(data_loader_out, "out of distribution")
    
    # plot using matplotlib and then visualize the picture 
    # using W&B media.
    try:
        # return an image of the histogram
        plt.hist(in_distr_scores, density=True, bins=100,
                 alpha=0.5, label="in distribution")
        plt.hist(out_distr_scores, density=True, bins=100,
                 alpha=0.5, label="out distribution")
        plt.title("Histogram of log likelihoods")
        plt.legend(loc="upper right")
        buf = io.BytesIO()
        # Save your plot to the buffer
        plt.savefig(buf, format="png")

        # Use PIL to convert the BytesIO object to an image object
        buf.seek(0)
        img = Image.open(buf)
    finally:
        plt.close()

    return np.array(img)

def standardize_sample_visualizing_format(sample):
    new_sample = sample
    if len(new_sample.shape) < 2 or new_sample.shape[-1] != new_sample.shape[-2]:
        # If the shape is not of an image, create a square image and do zero padding
        # to visualize and compare
        sqr_root = int(sqrt(new_sample.shape[-1]))
        if sqr_root * sqr_root < new_sample.shape[-1]:
            sqr_root += 1
        new_sample = torch.nn.functional.pad(
            input=new_sample, 
            pad=(0, sqr_root ** 2 - new_sample.shape[-1]),
            mode='constant', 
            value=0.0
        )
        new_shape = [sqr_root, sqr_root]
        if len(new_sample.shape) > 1:
            new_shape = list(new_sample.shape[:-1]) + new_shape
        new_sample = new_sample.reshape(new_shape)
        new_sample = new_sample.unsqueeze(-3)
        mn = new_sample.min()
        mx = new_sample.max()
        return (new_sample - mn) / (mx - mn)
    return sample

def run_ood(config: dict, gpu_index: int = 0, checkpoint_dir: th.Optional[str] = None):
    """
    Check the docs to see how the config dictionary looks like.
    This is the dictionary obtained after parsing the YAML file using jsonargparse.
    """
    ###################
    # (1) Model setup #
    ###################
    load_dotenv(override=True)
    
    if 'MODEL_DIR' in os.environ:
        model_root = os.environ['MODEL_DIR']
    else:
        model_root = './runs'
    
    device = f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu"
    
    model = load_model_with_checkpoints(config=config['base_model'], root=model_root, device=device)
    
    
    model.to(device)
    
    ##################
    # (1) Data setup #
    ##################
    # Load the environment variables
    
    # Set the data directory if it is specified in the environment
    # variables, otherwise, set to './data'
    if 'DATA_DIR' in os.environ:
        data_root = os.environ['DATA_DIR']
    else:
        data_root = './data'
        
    in_train_loader, _, in_test_loader = get_loaders(
        **config["data"]["in_distribution"]["dataloader_args"],
        device=device,
        shuffle=False,
        data_root=data_root,
        unsupervised=True,
    )
    ood_train_loader, _, ood_test_loader = get_loaders(
        **config["data"]["out_of_distribution"]["dataloader_args"],
        device=device,
        shuffle=False,
        data_root=data_root,
        unsupervised=True,
    )
    
    
    # in_loader is the loader that is used for the in-distribution data
    if not 'pick_loader' in config['data']['in_distribution']:
        print("pick_loader for in-distribution not in config, setting to test")
        config['data']['in_distribution']['pick_loader'] = 'test'
    
    if config['data']['in_distribution']['pick_loader'] == 'test':
        in_loader = in_test_loader
    elif config['data']['in_distribution']['pick_loader'] == 'train':
        in_loader = in_train_loader
    
    # out_loader is the loader that is used for the out-of-distribution data
    if not 'pick_loader' in config['data']['out_of_distribution']:
        print("pick_loader for ood not in config, setting to test")
        config['data']['out_of_distribution']['pick_loader'] = 'test'
        
    if config['data']['out_of_distribution']['pick_loader'] == 'test':
        out_loader = ood_test_loader
    elif config['data']['out_of_distribution']['pick_loader'] == 'train':
        out_loader = ood_train_loader


    ############################################################
    # (3) Log model samples and in/out of distribution samples #
    ############################################################
    
    # print out a sample ood and in distribution image onto the wandb logger
    if "seed" in config["data"]:
        np.random.seed(config["data"]["seed"])

    # you can set to visualize or bypass the visualization for speedup!
    if 'bypass_visualization' not in config['ood'] or not config['ood']['bypass_visualization']:
        
        if not config['ood'].get('bypass_dataset_visualization', False):
            # get 9 random samples from the in distribution dataset
            sample_set = np.random.randint(len(in_loader.dataset), size=9)
            in_samples = []
            for s in sample_set:
                in_samples.append(standardize_sample_visualizing_format(in_loader.dataset[s]))
            sample_set = np.random.randint(len(out_loader.dataset), size=9)
            out_samples = []
            for s in sample_set:
                out_samples.append(standardize_sample_visualizing_format(out_loader.dataset[s]))
            in_samples = torch.stack(in_samples)
            out_samples = torch.stack(out_samples)

            in_samples = torchvision.utils.make_grid(in_samples, nrow=3)
            out_samples = torchvision.utils.make_grid(out_samples, nrow=3)
            
            wandb.log({"data/in_distribution_samples": [wandb.Image(
                in_samples, caption="in distribution_samples")]})
            wandb.log({"data/out_of_distribution samples": [wandb.Image(
                out_samples, caption="out of distribution samples")]})
        
        # generate 9 samples from the model if bypass sampling is not set to True
        if 'samples_visualization' in config['ood']:
            if config['ood']['samples_visualization'] > 0:
                # with torch.no_grad():
                def log_samples():
                    samples = model.sample(9, **config['ood'].get('sampling_kwargs', {}))
                    # samples = standardize_sample_visualizing_format(samples)
                    samples = torchvision.utils.make_grid(samples, nrow=3)
                    wandb.log(
                        {"data/model_generated": [wandb.Image(samples, caption="model generated")]})
                # set torch seed for reproducibility
                if config["ood"]["seed"] is not None:
                    if device.startswith("cuda"):
                        torch.cuda.manual_seed(config["ood"]["seed"])
                    torch.manual_seed(config["ood"]["seed"])
                    log_samples()
                else:
                    log_samples()
                        
            if config['ood']['samples_visualization'] > 1:
                wandb.log({"data/most_probable": 
                    [
                        wandb.Image(
                            standardize_sample_visualizing_format(model.sample(-1).squeeze()), 
                            caption="max likelihood"
                        )
                    ]
                })
        
        def log_histograms():
            limit = config['ood'].get('histogram_limit', None)
            img_array = plot_likelihood_ood_histogram(
                model,
                in_loader,
                out_loader,
                limit=limit,
                log_prob_kwargs=config['ood'].get('log_prob_kwargs', {}),
            )
            wandb.log({"likelihood_ood_histogram": [wandb.Image(
                img_array, caption="Histogram of log likelihoods")]})
        
        if "bypass_visualize_histogram" not in config['ood'] or not config['ood']['bypass_visualize_histogram']:
            if config["ood"]["seed"] is not None:  
                if device.startswith("cuda"):
                    torch.cuda.manual_seed(config["ood"]["seed"])
                    torch.manual_seed(config["ood"]["seed"])
                log_histograms()
            else:
                log_histograms()
                
    
    #########################################
    # (4) Instantiate an OOD solver and run #
    #########################################
    
    # For dummy runs that you just use for visualization
    if "method_args" not in config["ood"] or "method" not in config["ood"]:
        print("No ood method available! Exiting...")
        return
    
    method_args = copy.deepcopy(config["ood"]["method_args"])
    method_args["likelihood_model"] = model

    # pick a random batch with seed for reproducibility
    if config["ood"]["seed"] is not None:
        np.random.seed(config["ood"]["seed"])
    idx = np.random.randint(len(out_loader))    
    for _ in range(idx + 1):
        x = next(iter(out_loader))

    if config["ood"].get("pick_single", False):
        # pick a single image the selected batch
        method_args["x_loader"] = [x[np.random.randint(x.shape[0])].unsqueeze(0)]
    elif config["ood"].get("use_dataloader", False):
        method_args["x_loader"] = out_loader
        if config["ood"].get("pick_count", 0) > 0:
            t = min(config['ood']['pick_count'], len(out_loader))
            method_args["x_loader"] = []
            iterable_ = iter(out_loader)
            for _ in range(t):
                method_args["x_loader"].append(next(iterable_))
    elif "pick_count" not in config["ood"]:
        raise ValueError("pick_count not in config when pick_single=False")
    else:
        # pass in the entire batch
        r = min(config["ood"]["pick_count"], x.shape[0])
        method_args["x_loader"] = [x[:r]]
    
    method_args["in_distr_loader"] = in_train_loader
    method_args["checkpoint_dir"] = checkpoint_dir
    
    if device.startswith("cuda"):
        torch.cuda.manual_seed(config["ood"]["seed"])
    torch.manual_seed(config["ood"]["seed"])
    method = dy.eval(config["ood"]["method"])(**method_args)
    
    
    # Call the run function of the given method
    method.run()

def dysweep_compatible_run(config, checkpoint_dir, gpu_index: int = 0):
    """
    Function compatible with dysweep
    """
    try:
        run_ood(config, gpu_index=gpu_index, checkpoint_dir=checkpoint_dir)
    except Exception as e:
        print("Exception:\n", e)
        print(traceback.format_exc())
        print("-----------")
        raise e

if __name__ == "__main__":
    # create a jsonargparse that gets a config file
    parser = ArgumentParser()
    parser.add_class_arguments(
        OODConfig,
        fail_untyped=False,
        sub_configs=True,
    )
    # add an argument to the parser pertaining to the gpu index
    parser.add_argument(
        '--gpu-index',
        type=int,
        help="The index of GPU being used",
        default=0,
    )
    parser.add_argument(
        "--config",
        action=ActionConfigFile,
        help="Path to the config file",
    )
    
    
    args = parser.parse_args()
    
    print("Running on gpu index", args.gpu_index)
    
    conf = {
        "base_model": args.base_model,
        "data": args.data,
        "ood": args.ood,
    }
    if "name" in args.logger:
        # add a random word to the name
        r = RandomWords()
        args.logger["name"] += f"-{r.get_random_word()}"

    wandb.init(config=conf, **args.logger)

    # set the checkpoint_dir to the dotenv variable if it exists
    load_dotenv(override=True)
    if 'MODEL_DIR' in os.environ:
        checkpoint_dir = os.environ['MODEL_DIR']
    else:
        checkpoint_dir = './runs'
    
    timestamp = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    checkpoint_dir = os.path.join(checkpoint_dir, timestamp)
    # make the directories if they do not exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    run_ood(conf, gpu_index=args.gpu_index, checkpoint_dir=checkpoint_dir)
