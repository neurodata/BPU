import os
import torch
from torchvision import datasets, transforms
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

# Get the project root directory (assuming this script is in src/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

class DataLoader:
    def __init__(self, task='mnist', batch_size=64, test_batch_size=256):
        self.task = task
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        
    def get_transforms(self):
        if self.task == 'mnist':
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        else:
            raise ValueError(f"Unsupported task: {self.task}")

    def load_data(self, data_dir=None):
        """Load the full dataset and create test loader."""
        if data_dir is None:
            data_dir = os.path.join(PROJECT_ROOT, "data")
            
        if self.task == 'mnist':
            transform = self.get_transforms()
            train_set = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
            test_set = datasets.MNIST(data_dir, train=False, transform=transform)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.test_batch_size, shuffle=False)
            return train_set, test_loader
        else:
            raise ValueError(f"Unsupported task: {self.task}")

    def create_fewshot_subset(self, dataset, seed, samples_per_class=60):
        """Create a few-shot subset of the dataset."""
        targets = np.array(dataset.targets)
        train_size = (samples_per_class * 10) / len(targets)
        sss = StratifiedShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
        indices, _ = next(sss.split(np.zeros_like(targets), targets))
        return torch.utils.data.Subset(dataset, indices)

    def get_train_loader(self, full_train_set, epoch=None, fewshot_samples=None, fewshot_batch_size=None):
        """Get the appropriate train loader based on configuration.
        
        Args:
            full_train_set: The complete training dataset
            epoch: Current epoch number (for few-shot learning)
            fewshot_samples: Number of samples per class for few-shot learning
            fewshot_batch_size: Batch size for few-shot learning
            
        Returns:
            torch.utils.data.DataLoader: The training data loader
        """
        if fewshot_samples is not None:
            # Few-shot learning mode
            subset = self.create_fewshot_subset(full_train_set, epoch, fewshot_samples)
            return torch.utils.data.DataLoader(
                subset, 
                batch_size=fewshot_batch_size or self.batch_size, 
                shuffle=True
            )
        else:
            # Regular training mode
            return torch.utils.data.DataLoader(
                full_train_set, 
                batch_size=self.batch_size, 
                shuffle=True
            ) 