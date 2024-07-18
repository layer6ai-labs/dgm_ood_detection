import os
from pathlib import Path
from typing import Any, Tuple
import pandas as pd
import PIL
import torch
from torch.utils.data import Dataset
import torchvision.datasets
import torchvision.transforms as transforms
from model_zoo.datasets.supervised_dataset import SupervisedDataset
import numpy as np
import glob
import typing as th
from math import inf
from tqdm import tqdm
from .image_embeddings import EmbeddingWrapper
from typing import Literal
import medmnist
from medmnist import PneumoniaMNIST

class CelebACropped(Dataset):
    """
    Dataset taken from 
    
    Kist, Andreas M. (2021, October 11). 
    CelebA Dataset cropped with Haar-Cascade face detector. 
    Zenodo. https://doi.org/10.5281/zenodo.5561092
    """
    def __init__(
        self, 
        root: str, 
        role: str = 'train', 
        valid_fraction: float = 0.1, 
        seed: int = 0,
        device: str = "cpu",
        resize_image: th.Optional[Tuple[int, int]] = None,
    ):
        self.resize_image = resize_image if resize_image is not None else (32, 32)
        self.root = root 
        
        if not os.path.exists(root):
            print("Downloading CelebA dataset (clean and small version) ...")
            url = 'https://zenodo.org/record/5561092/files/celeba_5000_clean.zip?download=1'
            
            # Send HTTP request to the specified URL and save the response from server in a response object called r
            import requests
            r = requests.get(url)
            file_path = "celeba_5000_clean.zip"
            # Open the zip file in the write-binary mode
            with open(file_path, 'wb') as f:
                f.write(r.content)
                
            # now unzip
            from zipfile import ZipFile
            # Use ZipFile to extract
            with ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(self.root)
            os.remove(file_path)
        
        self.image_paths = []
        # list all the files with .jpg extension
        self.all_image_files = glob.glob(os.path.join(self.root, 'celeba_5000_clean', '*.jpg'))
        
        np.random.seed(seed)
        randperm = np.random.permutation(len(self.all_image_files))
        train_len = int(0.9 * (1 - valid_fraction) * len(self.all_image_files))
        valid_len = int(0.9 * valid_fraction * len(self.all_image_files))
        if role == 'train':
            self.all_image_files = [self.all_image_files[i] for i in randperm[:train_len]]
        elif role == 'valid':
            self.all_image_files = [self.all_image_files[i] for i in randperm[train_len:train_len + valid_len]]
        elif role == 'test':
            self.all_image_files = [self.all_image_files[i] for i in randperm[train_len + valid_len:]]
        else:
            raise ValueError(f"Unknown role {role}")
        
        self.transform = transforms.Compose([
            transforms.Resize(self.resize_image, antialias=True),
            transforms.ToTensor(),
        ])
        
        self.cached_values = {}
        
        self.device = device
        
    def __len__(self):
        return len(self.all_image_files)

    def to(self, device):
        if device != self.device:
            self.cached_values = {}
        self.device = device
        return self
    
    def get_data_min(self):
        return 0.0
    
    def get_data_max(self):
        return 255.0
    
    def get_data_shape(self):
        return (3, self.resize_image[0], self.resize_image[1])
    
    def __getitem__(self, idx):
        if idx not in self.cached_values:
            img_name = self.all_image_files[idx]
            image = PIL.Image.open(img_name).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
            self.cached_values[idx] = torch.clamp(image.to(self.device) * 255, min=0, max=255)
        return self.cached_values[idx]
    
class CelebA(Dataset):
    """
    CelebA PyTorch dataset
    """

    def __init__(
        self, 
        root: str, 
        role: str = "train",
        valid_fraction: float = 0.1, 
        seed: int = 0,
        device: str = "cpu",
        resize_image: th.Optional[Tuple[int, int]] = None,
    ):
        self.resize_image = resize_image if resize_image is not None else (32, 32)
        self.celeba_dataset = torchvision.datasets.CelebA(
            root=root,
            split=role,
            download=True,
            transform = transforms.Compose([
                transforms.Resize(self.resize_image, antialias=True),
                transforms.ToTensor(),
            ])
        )
                
        self.root = Path(root)
        self.role = role
        
        self.device = device
        
        self.transform = transforms.Compose([
            transforms.Resize(self.resize_image, antialias=True),
        ])
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        return self.transform(255 * self.celeba_dataset[index][0].to(self.device)), self.celeba_dataset[index][1].to(self.device)

    def __len__(self) -> int:
        return len(self.celeba_dataset)
    
    def get_data_min(self):
        return 0.0
    
    def get_data_max(self):
        return 255.0
    
    def get_data_shape(self):
        return (3, self.resize_image[0], self.resize_image[1])
    
    def to(self, device):
        self.device = device
        return self


class EMNIST(Dataset):
    def __init__(
        self, 
        root: str, 
        role: str = "train", 
        valid_fraction: float = 0.1, 
        seed: int = 0,
        classes_to_ignore: int = 0, 
        device: str = "cpu",
        resize_image: th.Optional[Tuple[int, int]] = None,
    ):
        self.resize_image = resize_image if resize_image is not None else (28, 28)
        self.device = device
        self.classes_to_ignore = classes_to_ignore
        # Download and load the EMNIST dataset
        self.emnist_dataset = torchvision.datasets.EMNIST(
            root, 
            split='byclass', 
            train=True, 
            download=True,
        )

        # Get the indices of examples that are not digits (i.e., their labels are not between 0 and 9)
        self.indices = [i for i in range(len(self.emnist_dataset)) if self.emnist_dataset[i][1] >= self.classes_to_ignore]
        
        # shuffle indices with a fixed seed
        np.random.seed(seed)
        self.indices = np.random.permutation(np.array(self.indices))
        test_size = int(0.1 * len(self.indices))
        if role == 'test':
            self.indices = self.indices[:test_size]
        else:
            self.indices = self.indices[test_size:]
            valid_size = int(valid_fraction * len(self.indices))
            if role == 'train':
                self.indices = self.indices[valid_size:]
            else:
                self.indices = self.indices[:valid_size]
        
        self.transforms = transforms.Compose([
            transforms.Resize(self.resize_image, antialias=True),
            transforms.ToTensor(),
        ])
            
    def get_data_min(self):
        return 0.0
    
    def get_data_max(self):
        return 255.0
    
    def get_data_shape(self):
        return (1, self.resize_image[0], self.resize_image[1])
          
    def __getitem__(self, index):
        # Get the image and label from the original EMNIST dataset for the given index
        image, label = self.emnist_dataset[self.indices[index]]
        image = 255.0 * self.transforms(image)
        # Subtract 10 from the label, so that labels start from 0 (not a necessary step, but often useful)
        return image.to(self.device), label - self.classes_to_ignore

    def __len__(self):
        return len(self.indices)
    
    def to(self, device):
        self.device = device
        return self

class EMNISTMinusMNIST(EMNIST):
    def __init__(
        self, 
        root: str, 
        role: str = "train", 
        valid_fraction: float = 0.1, 
        seed: int = 0,
        device: str = "cpu",
        resize_image: th.Optional[Tuple[int, int]] = None,
    ):
        super().__init__(
            root=root,
            role=role,
            valid_fraction=valid_fraction,
            seed=seed,
            classes_to_ignore=10,
            device=device,
            resize_image=resize_image,
        )
    

class Omniglot(Dataset):
    def __init__(self, root, role, valid_fraction, seed: int = 0, device: str = "cpu", resize_image: th.Optional[Tuple[int, int]] = None):
        self.omniglot = torchvision.datasets.Omniglot(
            root=root, 
            background=True, 
            download=True
        )
        self.resize_image = resize_image if resize_image is not None else (28, 28)
        self.device = device
        # shuffle the dataset deterministically according to the splitting seed
        
        torch.manual_seed(seed)
        perm = torch.randperm(len(self.omniglot))
        test_len = int(len(self.omniglot) * 0.1)
        valid_len = int(len(self.omniglot) * valid_fraction)
        test_indices = perm[:test_len]
        valid_indices = perm[test_len:valid_len+test_len]
        train_indices = perm[valid_len+test_len:]
        if role == "train":
            self.omniglot = torch.utils.data.Subset(self.omniglot, train_indices)
        elif role == 'valid':
            self.omniglot = torch.utils.data.Subset(self.omniglot, valid_indices)
        elif role == 'test':
            self.omniglot = torch.utils.data.Subset(self.omniglot, test_indices)
        else:
            raise ValueError(f"Unknown role {role}")
        
        self.cached_values = {}
            
        self.transform = transforms.Compose([
            transforms.Resize(self.resize_image, antialias=True),
            transforms.ToTensor(),
        ])  
        
    
    def get_data_min(self):
        return 0.0
    
    def get_data_max(self):
        return 255.0
    
    def get_data_shape(self):
        return (1, self.resize_image[0], self.resize_image[1])
           
    def __len__(self):
        return len(self.omniglot)

    def __getitem__(self, index) -> Any:
        if index not in self.cached_values:
            img, label = self.omniglot[index]
            img = self.transform(img)
            
            # turn label from an int to a tensor
            label = torch.tensor(label)
            
            # TODO: look into why I have to apply a x255 here!
            img = 255 * (1.0 - img).clamp(0.0, 1.0)
            
            self.cached_values[index] = img.to(self.device), label.to(self.device)
        return self.cached_values[index]
    
    def to(self, device):
        if device != self.device:
            self.cached_values = {}
        self.device = device
        return self



class TinyImageNet(Dataset):
    def __init__(
        self, 
        root, 
        role, 
        valid_fraction, 
        seed: int = 0, 
        device: str = "cpu",
        resize_image: th.Optional[Tuple[int, int]] = None,
    ):
        #self, root_dir, n_images, train_or_test, transform=None, ):
        """
        Args:
            text_file(string): path to text file
            root_dir(string): directory with all train images
        """
        self.resize_image = resize_image if resize_image is not None else (32, 32)
        self.root_dir = os.path.join(root, 'tiny-imagenet')
        if not os.path.exists(self.root_dir):
            # download the dataset from http://cs231n.stanford.edu/tiny-imagenet-200.zip
            # create the root directory and extract the zip file in it
            # then erase the zip file
            
            os.makedirs(self.root_dir)
            # now download
            # URL of the dataset
            print("Downloading dataset ...")
            
            url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

            # Send HTTP request to the specified URL and save the response from server in a response object called r
            import requests
            r = requests.get(url)
            file_path = "tiny-imagenet-200.zip"
            # Open the zip file in the write-binary mode
            with open(file_path, 'wb') as f:
                f.write(r.content)
                
            # now unzip
            from zipfile import ZipFile
            # Use ZipFile to extract
            with ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(self.root_dir)
            os.remove(file_path)
            
        self.transform = transforms.Compose([
            transforms.Resize(self.resize_image, antialias=True),
            transforms.ToTensor(),
        ])
        if role in ['train', 'valid']:
            folds = ('train', 'val',)
        else:
            folds = ('test',)

        all_image_files = []
        for fold in folds:
            if fold == 'train':
                subfolders = glob.glob(os.path.join(self.root_dir, "tiny-imagenet-200", fold, '*'))
                for f in subfolders:
                    image_files = glob.glob(os.path.join(f, 'images/', '*.JPEG'))
                    all_image_files.extend(image_files)
            else:
                image_files = glob.glob(os.path.join(self.root_dir, "tiny-imagenet-200", fold, 'images', '*.JPEG'))
                
                all_image_files.extend(image_files)
        self.all_image_files = all_image_files
        
        np.random.seed(seed)
        randperm = np.random.permutation(len(self.all_image_files))
        train_split = int((1 - valid_fraction) * len(self.all_image_files))
        if role == 'train':
            self.all_image_files = [self.all_image_files[i] for i in randperm[:train_split]]
        elif role == 'valid':
            self.all_image_files = [self.all_image_files[i] for i in randperm[train_split:]]
            
        self.cached_values = {}
        
        self.device = device

    def __len__(self):
        return len(self.all_image_files)

    def to(self, device):
        if device != self.device:
            self.cached_values = {}
        self.device = device
        return self
    
    def get_data_min(self):
        return 0.0
    
    def get_data_max(self):
        return 255.0
    
    def get_data_shape(self):
        return (3, self.resize_image[0], self.resize_image[1])
    
    def __getitem__(self, idx):
        if idx not in self.cached_values:
            
            img_name = self.all_image_files[idx]
            image = PIL.Image.open(img_name).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
            self.cached_values[idx] = torch.clamp(image.to(self.device) * 255, min=0, max=255)
            
        return self.cached_values[idx]

class MNIST(SupervisedDataset):
    
    def __init__(
        self,
        root, 
        role, 
        valid_fraction, 
        seed: int = 0, 
        device: str = "cpu",
        resize_image: th.Optional[Tuple[int, int]] = None,
    ):
        self.resize_image = resize_image if resize_image is not None else (28, 28)
        self.dataset = torchvision.datasets.MNIST(root=root, train=role in ['train', 'valid'], download=True)
        self.images = self.dataset.data.unsqueeze(1).to(torch.uint8)
        self.labels = self.dataset.targets.to(torch.uint8)
        
        np.random.seed(seed)
        perm = np.random.permutation(np.arange(len(self.images)))
        shuffled_images = self.images[perm]
        shuffled_labels = self.labels[perm]

        valid_size = int(valid_fraction * self.images.shape[0])
        
        if role == 'train':
            images = shuffled_images[valid_size:]
            labels = shuffled_labels[valid_size:]
        elif role == 'valid':
            images = shuffled_images[:valid_size]
            labels = shuffled_labels[:valid_size]
        elif role == 'test':
            images = shuffled_images
            labels = shuffled_labels
           
        images = images.to(dtype=torch.get_default_dtype())
        labels = labels.long()
        
        super().__init__("mnist", role, x=transforms.Resize(self.resize_image, antialias=True)(images), y=labels)
        
        self.to(device)

class MedMNIST(SupervisedDataset):
    
    def __init__(
        self, 
        root, 
        subclass: Literal['ChestMNIST', 'PneumoniaMNIST', 'OrganAMNIST', 'OrganCMNIST', 'OrganSMNIST'],
        role, 
        valid_fraction, 
        seed: int = 0, 
        device: str = "cpu",
        resize_image: th.Optional[Tuple[int, int]] = None,
        align: bool = True,
    ):
        
        self.resize_image = resize_image if resize_image is not None else (28, 28)
        # import the argument 'subclass' from 'medmnist' using import_module
        # if root doesn't exist create it
        if not os.path.exists(root):
            os.makedirs(root)
        self._dataset = getattr(medmnist, subclass)(root=root, split="val" if role == "valid" else role, download=True)
        
        np.random.seed(seed)
        perm = np.random.permutation(np.arange(len(self._dataset)))
        valid_size = int(valid_fraction * len(self._dataset))
        
        if role == 'train':
            indices = perm[valid_size:]
        elif role == 'valid':
            indices = perm[:valid_size]
        elif role == 'test':
            indices = perm
        
        if len(indices) == 0:
            indices = torch.IntTensor([perm[0]])
        
        images = torch.stack([transforms.ToTensor()(self._dataset[i][0]) for i in indices]) * 255
        labels = torch.tensor([self._dataset[i][1][0] for i in indices])
        
        images = images.to(dtype=torch.get_default_dtype())
        if align and subclass == 'ChestMNIST':
            images = transforms.CenterCrop((27, 18))(images) # center crop the chest x-ray images to align with the PneumoniaMNIST images
        labels = labels.long()
    
        super().__init__(f"medmnist-{subclass}", role, x=transforms.Resize(self.resize_image, antialias=True)(images), y=labels)

        self.to(device)
        

class FMNIST(SupervisedDataset):
    
    def __init__(
        self, 
        root, 
        role, 
        valid_fraction, 
        seed: int = 0, 
        device: str = "cpu",
        resize_image: th.Optional[Tuple[int, int]] = None,
    ):
        self.resize_image = resize_image if resize_image is not None else (28, 28)
        self.dataset = torchvision.datasets.FashionMNIST(root=root, train=role in ['train', 'valid'], download=True)
        self.images = self.dataset.data.unsqueeze(1).to(torch.uint8)
        self.labels = self.dataset.targets.to(torch.uint8)
        
        np.random.seed(seed)
        perm = np.random.permutation(np.arange(len(self.images)))
        shuffled_images = self.images[perm]
        shuffled_labels = self.labels[perm]

        valid_size = int(valid_fraction * self.images.shape[0])
        
        if role == 'train':
            images = shuffled_images[valid_size:]
            labels = shuffled_labels[valid_size:]
        elif role == 'valid':
            images = shuffled_images[:valid_size]
            labels = shuffled_labels[:valid_size]
        elif role == 'test':
            images = shuffled_images
            labels = shuffled_labels
           
        images = images.to(dtype=torch.get_default_dtype())
        labels = labels.long()
        
        super().__init__("fashion-mnist", role, x=transforms.Resize(self.resize_image, antialias=True)(images), y=labels)

        self.to(device)
        

class CIFAR10(Dataset):

    def __init__(
        self, 
        root, 
        role, 
        valid_fraction, 
        seed: int = 0, 
        device: str = "cpu",
        resize_image: th.Optional[Tuple[int, int]] = None,
    ):
        self.cached_values = {}
        self.dataset = torchvision.datasets.CIFAR10(root=root, train=role in ['train', 'valid'], download=True)
        self.images = self.dataset.data
        self.labels = np.array(self.dataset.targets)
        self.device = device
        
        np.random.seed(seed)
        perm = np.random.permutation(np.arange(len(self.images)))
        shuffled_images = self.images[perm]
        shuffled_labels = self.labels[perm]

        valid_size = int(valid_fraction * self.images.shape[0])
        
        if role == 'train':
            images = shuffled_images[valid_size:]
            labels = shuffled_labels[valid_size:]
        elif role == 'valid':
            images = shuffled_images[:valid_size]
            labels = shuffled_labels[:valid_size]
        elif role == 'test':
            images = shuffled_images
            labels = shuffled_labels
           
        self.images = images
        self.labels = labels
        self.resize_image = resize_image if resize_image is not None else (32, 32)
        # Resize the images using self.resize_image
        self.transform = transforms.Compose([
            transforms.Resize(self.resize_image, antialias=True),
        ])
        
        
    def __len__(self):
        return len(self.labels)

    def to(self, device):
        if device != self.device:
            self.cached_values = {}
        self.device = device
        return self
    
    def get_data_min(self):
        return 0.0
    
    def get_data_max(self):
        return 255.0
    
    def get_data_shape(self):
        return (3, self.resize_image[0], self.resize_image[1])
    
    def __getitem__(self, idx):
        if idx not in self.cached_values:
            
            img = torch.from_numpy(self.images[idx]).to(dtype=torch.get_default_dtype()).to(self.device).permute((2, 0, 1))
            self.cached_values[idx] = self.transform(img), self.labels[idx]
            
        return self.cached_values[idx]
        
class CIFAR100(Dataset):
    
    def __init__(self, root, role, valid_fraction, seed: int = 0, device: str = "cpu", resize_image: th.Optional[Tuple[int, int]] = None):
        self.resize_image = resize_image if resize_image is not None else (32, 32)
        self.cached_values = {}
        self.dataset = torchvision.datasets.CIFAR100(root=root, train=role in ['train', 'valid'], download=True)
        self.images = self.dataset.data
        self.labels = np.array(self.dataset.targets)
        self.device = device
        
        np.random.seed(seed)
        perm = np.random.permutation(np.arange(len(self.images)))
        shuffled_images = self.images[perm]
        shuffled_labels = self.labels[perm]

        valid_size = int(valid_fraction * self.images.shape[0])
        
        if role == 'train':
            images = shuffled_images[valid_size:]
            labels = shuffled_labels[valid_size:]
        elif role == 'valid':
            images = shuffled_images[:valid_size]
            labels = shuffled_labels[:valid_size]
        elif role == 'test':
            images = shuffled_images
            labels = shuffled_labels
           
        self.images = images
        self.labels = labels
        self.resize_image = resize_image if resize_image is not None else (32, 32)
        # Resize the images using self.resize_image
        self.transform = transforms.Compose([
            transforms.Resize(self.resize_image, antialias=True),
        ])
        
    def __len__(self):
        return len(self.labels)

    def to(self, device):
        if device != self.device:
            self.cached_values = {}
        self.device = device
        return self
    
    def get_data_min(self):
        return 0.0
    
    def get_data_max(self):
        return 255.0
    
    def get_data_shape(self):
        return (3, self.resize_image[0], self.resize_image[1])
    
    def __getitem__(self, idx):
        if idx not in self.cached_values:
            
            img = torch.from_numpy(self.images[idx]).to(dtype=torch.get_default_dtype()).to(self.device).permute((2, 0, 1))
            img = self.transform(img)
            self.cached_values[idx] = img, self.labels[idx]
            
        return self.cached_values[idx]

class SVHN(Dataset):
    def __init__(self, root, role, valid_fraction, seed: int = 0, device: str = "cpu", resize_image: th.Optional[Tuple[int, int]] = None):
        self.resize_image = resize_image if resize_image is not None else (32, 32)
        self.cached_values = {}
        self.dataset = torchvision.datasets.SVHN(root=root, split="train" if role in ['valid' or 'train'] else "test", download=True)
        self.images = self.dataset.data
        self.labels = np.array(self.dataset.labels)
        self.device = device
        
        np.random.seed(seed)
        perm = np.random.permutation(np.arange(len(self.images)))
        shuffled_images = self.images[perm]
        shuffled_labels = self.labels[perm]

        valid_size = int(valid_fraction * self.images.shape[0])
        
        if role == 'train':
            images = shuffled_images[valid_size:]
            labels = shuffled_labels[valid_size:]
        elif role == 'valid':
            images = shuffled_images[:valid_size]
            labels = shuffled_labels[:valid_size]
        elif role == 'test':
            images = shuffled_images
            labels = shuffled_labels
           
        self.images = images
        self.labels = labels
        
        self.resize_image = resize_image if resize_image is not None else (32, 32)
        # Resize the images using self.resize_image
        self.transform = transforms.Compose([
            transforms.Resize(self.resize_image, antialias=True),
        ])
        
    def __len__(self):
        return len(self.labels)

    def to(self, device):
        if device != self.device:
            self.cached_values = {}
        self.device = device
        return self
    
    def get_data_min(self):
        return 0.0
    
    def get_data_max(self):
        return 255.0
    
    def get_data_shape(self):
        return (3, self.resize_image[0], self.resize_image[1])
    
    def __getitem__(self, idx):
        if idx not in self.cached_values:
            
            img = torch.from_numpy(self.images[idx]).to(dtype=torch.get_default_dtype()).to(self.device)
            img = self.transform(img)
            self.cached_values[idx] = img, self.labels[idx]
            
        return self.cached_values[idx]
    

def get_image_datasets_by_class(
    dataset_name, 
    data_root, 
    valid_fraction, 
    seed: int = 0, 
    device: str = "cpu",
    additional_dataset_args: th.Optional[dict] = None,    
):
    data_dir = os.path.join(data_root, dataset_name)

    data_class = {
        'celeba': CelebA,
        'celeba-small': CelebACropped,
        'omniglot': Omniglot,
        'emnist-minus-mnist': EMNISTMinusMNIST,
        'emnist': EMNIST,
        'tiny-imagenet': TinyImageNet,
        'mnist': MNIST,
        'fashion-mnist': FMNIST,
        'cifar10': CIFAR10,
        'cifar100': CIFAR100,
        'svhn': SVHN,
        'medmnist': MedMNIST,
    }[dataset_name]

    
    additional_dataset_args = additional_dataset_args or {}
    train_dset = data_class(root=data_dir, role="train", valid_fraction=valid_fraction, seed=seed, device=device, **additional_dataset_args)
    valid_dset = data_class(root=data_dir, role="valid", valid_fraction=valid_fraction, seed=seed, device=device, **additional_dataset_args)
    test_dset = data_class(root=data_dir, role="test", valid_fraction=valid_fraction, seed=seed, device=device, **additional_dataset_args)

    return train_dset, valid_dset, test_dset


def get_image_datasets(
    dataset_name, 
    data_root, 
    make_valid_dset, 
    embedding_network: th.Optional[str],
    device: str = "cpu",
    **additional_dataset_args,
):
    # Currently hardcoded; could make configurable
    valid_fraction = 0.1 if make_valid_dset else 0
    
    # torchvision_datasets = ["mnist", "fashion-mnist", "svhn", "cifar10", "cifar100"]
    
    # get_datasets_fn = get_torchvision_datasets if dataset_name in torchvision_datasets else get_image_datasets_by_class
    
    train_dset, valid_dset, test_dset = get_image_datasets_by_class(
        dataset_name, 
        data_root, 
        valid_fraction, 
        device=device, 
        additional_dataset_args=additional_dataset_args
    )

    if embedding_network is not None:
        train_dset = EmbeddingWrapper(
            dset=train_dset, 
            embedding_network=embedding_network, 
            device=device, 
            save_directory=os.path.join(data_root, f"{dataset_name}-{embedding_network}-train"),
        )
        valid_dset = EmbeddingWrapper(
            dset=valid_dset, 
            embedding_network=embedding_network, 
            device=device, 
            save_directory=os.path.join(data_root, f"{dataset_name}-{embedding_network}-valid"),
        )
        test_dset = EmbeddingWrapper(
            dset=test_dset, 
            embedding_network=embedding_network, 
            device=device, 
            save_directory=os.path.join(data_root, f"{dataset_name}-{embedding_network}-test"),
        )
        
    return train_dset, valid_dset, test_dset