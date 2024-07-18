import torch
import typing as th
from typing import Any, Tuple
from torchvision import transforms

class SupervisedDataset(torch.utils.data.Dataset):
    """Generic implementation of torch Dataset"""

    def __init__(
        self, 
        name, 
        role, 
        x, 
        y=None,
    ):
        if y is None:
            y = torch.zeros(x.shape[0]).long()

        assert x.shape[0] == y.shape[0]
        assert role in ["train", "valid", "test"]

        self.name = name
        self.role = role

        self.x = x 
        self.y = y

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        return self.x[index], self.y[index], index

    def to(self, device):
        return SupervisedDataset(
            self.name,
            self.role,
            self.x.to(device),
            self.y.to(device)
        )

    def get_data_max(self):
        return self.x.max()
    
    def get_data_min(self):
        return self.x.min()