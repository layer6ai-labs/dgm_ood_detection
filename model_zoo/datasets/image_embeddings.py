import os
from pathlib import Path
from typing import Any, Tuple
import pandas as pd
import PIL
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from model_zoo.datasets.supervised_dataset import SupervisedDataset
import numpy as np
import glob
import typing as th
import torchvision.models as models
from math import inf
from tqdm import tqdm

# TODO: fix the size of the embeddings
EMBEDDING_SIZE_MAPPING = {
    'dinov2_vits14': torch.Size([1000]),
    'dinov2_vitb14': torch.Size([1000]),
    'dinov2_vitl14': torch.Size([1000]),
    'dinov2_vitg14': torch.Size([1000]),
    'resnet50': torch.Size([1000]),
    'nvidia_efficientnet_b0': torch.Size([1000]),
}

def get_dinov2_encoder(architecture):
    VALID_ARCHITECTURES = [
        'dinov2_vits14',
        'dinov2_vitb14',
        'dinov2_vitl14',
        'dinov2_vitg14',
    ]
    assert architecture in VALID_ARCHITECTURES
    return torch.hub.load('facebookresearch/dinov2', architecture)


class EmbeddingTransform:
    """
    This is a one-time transform that performs embeddings on the input data.
    The embedding is obtained from a conventional image embedding architecture, e.g., 
    Resnet, DinoV2, EfficientNet.
    """
    def __init__(
        self,
        model_name: str,
        device: str,
    ):
        try:
            if model_name == 'resnet50':
                self.network_input_size = (224, 224)
                self.model = models.resnet50(pretrained=True)
            elif model_name.startswith('dinov2'):
                self.network_input_size = (224, 224)
                self.model = get_dinov2_encoder(model_name)
            elif 'efficientnet' in model_name:
                self.network_input_size = (224, 224)
                self.model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', model_name, pretrained=True)
        except Exception as e:
            raise Exception(f"{model_name} is not supported!") from e

        self.model = self.model.to(device)
        self.model.eval()
        self.device = device
    
    def __call__(self, img) -> Any:
        
        with torch.no_grad():
            # Define a transformation to resize and normalize the input tensor
            preprocess = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.network_input_size),  # Resize to match the network input size
                transforms.ToTensor(),
            ])
            
            # Check if the input tensor is grayscale (1 channel) or RGB (3 channels)
            if img.size(0) == 1:
                # If grayscale, convert to RGB by repeating the single channel 3 times
                img = img.repeat(3, 1, 1)
                # Normalize each channel independently (grayscale normalization)
                img[0] = (img[0] - 0.485) / 0.229
                img[1] = (img[1] - 0.456) / 0.224
                img[2] = (img[2] - 0.406) / 0.225
            else:
                # If RGB, ensure it has 3 channels
                if img.size(0) != 3:
                    raise ValueError("Input tensor should have either 1 or 3 channels (grayscale or RGB).")

            # Preprocess the input tensor
            img = preprocess(img).to(self.device)
            
            # Add a batch dimension (unsqueeze) to match the model's input shape
            img = img.unsqueeze(0)
            # Forward pass through the model to obtain the embedding
            embedding = self.model(img)
        
        return embedding
    
class EmbeddingWrapper(Dataset):
    def __init__(
        self,
        dset: Dataset,
        embedding_network: str,
        device: str,
        save_directory: str,
    ):
        super().__init__()
        self.cached_values = {}
        self.dset = dset
        self.embedding_network = embedding_network
        self.embedding_transform = EmbeddingTransform(model_name=embedding_network, device=device)
        self.save_directory = save_directory
        self.device = device
        
        # Check if the directory already contains all we want or not
        try:
            # List all the files in the directory
            files_in_directory = os.listdir(save_directory)
            
            # Create the desired file names based on N
            desired_files = [f"{i}.npy" for i in range(len(self.dset))]
            
            # Check if files_in_directory contains only the desired_files
            if set(files_in_directory) != set(desired_files):
                raise Exception("directory content is different!")
            
            print("Found cached embeddings!")
            
        except Exception as e:
            # If at any point an exception occured, this means we have to run it again
            if os.path.exists(save_directory):
                os.rmdir(save_directory)
            os.makedirs(save_directory)
            
            for i in tqdm(range(len(self.dset)), desc=f"embedding data in {os.path.basename(save_directory)}"):
                item = self.dset[i]
                if isinstance(item, tuple):
                    img = item[0]
                else:
                    img = item
                img = (img - self.dset.get_data_min()) / self.dset.get_data_max()
                img = img.to(device)
                img = self.embedding_transform(img).cpu().numpy()
                np.save(os.path.join(self.save_directory, f"{i}.npy"), img)

    def _load_embedding(self, i: int):
        return np.load(os.path.join(self.save_directory, f"{i}.npy"))
    
    def __len__(self):
        return len(self.dset)

    def to(self, device):
        if device != self.device:
            self.cached_values = {}
        self.device = device
        return self
    
    def get_data_min(self):
        return -inf
    
    def get_data_max(self):
        return inf
    
    def get_data_shape(self):
        return EMBEDDING_SIZE_MAPPING[self.embedding_network]
    
    def __getitem__(self, idx):
        if idx not in self.cached_values:
            embedding = self._load_embedding(idx)
            embedding_tensor = torch.from_numpy(embedding).to(dtype=torch.get_default_dtype()).to(self.device).squeeze()
            item = self.dset[idx]
            if isinstance(item, tuple):
                item = tuple([embedding_tensor if i == 0 else x for i, x in enumerate(item)])
            else:
                item = embedding_tensor
            self.cached_values[idx] = item
            
        return self.cached_values[idx]
